#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/deploy_env_config.sh"

# Fix for gcloud IAP warning (some environments need python for IAP helpers).
if command -v python3 &>/dev/null; then
  export CLOUDSDK_PYTHON
  CLOUDSDK_PYTHON="$(command -v python3)"
fi

log(){ echo -e "[\e[36m$(date +'%F %T')\e[0m] $*"; }

require_cmd() {
  if ! command -v "$1" &>/dev/null; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd gcloud
require_cmd curl

# Check ADC credentials (optional but useful for some tooling).
if [ ! -f "${HOME}/.config/gcloud/application_default_credentials.json" ]; then
  log "ADC credentials not found. Running: gcloud auth application-default login"
  gcloud auth application-default login
fi

log "Step 1: Enabling required services..."
gcloud services enable \
  compute.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  cloudresourcemanager.googleapis.com \
  --project="${PROJECT_ID}"

log "Step 2: Ensuring Artifact Registry repo exists (${REPO_NAME})..."
if ! gcloud artifacts repositories describe "${REPO_NAME}" \
  --location="${REGION}" \
  --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Gomoku deploy images" \
    --project="${PROJECT_ID}"
else
  log " -> Repo already exists."
fi

log "Step 3: Ensuring service account exists (${SA_NAME})..."
if ! gcloud iam service-accounts describe "${SA_EMAIL}" \
  --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${SA_NAME}" \
    --display-name="Gomoku Deploy SA" \
    --project="${PROJECT_ID}"
  log " -> Waiting for service account propagation..."
  sleep 10
else
  log " -> Service account already exists."
fi

log "Step 3b: Granting minimal roles to service account..."
SA_ROLES=(
  "roles/artifactregistry.reader"
  "roles/logging.logWriter"
  "roles/monitoring.metricWriter"
)
for role in "${SA_ROLES[@]}"; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${role}" >/dev/null
  log " -> Granted ${role}"
done

log "Step 3c: Allowing user to attach the service account to VMs..."
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --member="user:${USER_EMAIL}" \
  --role="roles/iam.serviceAccountUser" \
  --project="${PROJECT_ID}" >/dev/null || true

log "Step 4: Creating firewall rules (target tag: gomoku-deploy)..."

# Cloudflare publishes these canonical IP lists:
# - https://www.cloudflare.com/ips-v4
# - https://www.cloudflare.com/ips-v6
CF_IPV4_RANGES="$(curl -fsSL https://www.cloudflare.com/ips-v4 | tr '\n' ',' | sed 's/,$//')"
if [ -z "${CF_IPV4_RANGES}" ]; then
  echo "Failed to fetch Cloudflare IPv4 ranges." >&2
  exit 1
fi

WS_RULE="gomoku-deploy-ws-8080"
if ! gcloud compute firewall-rules describe "${WS_RULE}" \
  --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud compute firewall-rules create "${WS_RULE}" \
    --project="${PROJECT_ID}" \
    --direction=INGRESS \
    --priority=1000 \
    --network=default \
    --action=ALLOW \
    --rules=tcp:8080 \
    --source-ranges="${CF_IPV4_RANGES}" \
    --target-tags="gomoku-deploy" \
    --description="Allow gomoku websocket traffic from Cloudflare to port 8080"
else
  log " -> Firewall rule exists: ${WS_RULE}"
fi

SSH_RULE="gomoku-deploy-ssh-iap"
if ! gcloud compute firewall-rules describe "${SSH_RULE}" \
  --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud compute firewall-rules create "${SSH_RULE}" \
    --project="${PROJECT_ID}" \
    --direction=INGRESS \
    --priority=1000 \
    --network=default \
    --action=ALLOW \
    --rules=tcp:22 \
    --source-ranges="35.235.240.0/20" \
    --target-tags="gomoku-deploy" \
    --description="Allow SSH via IAP TCP forwarding"
else
  log " -> Firewall rule exists: ${SSH_RULE}"
fi

log "Step 5: Reserving static external IPs..."
MINIMAX_IP_NAME="${MINIMAX_VM}-ip"
ALPHAZERO_IP_NAME="${ALPHAZERO_VM}-ip"

if ! gcloud compute addresses describe "${MINIMAX_IP_NAME}" \
  --region="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud compute addresses create "${MINIMAX_IP_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}"
else
  log " -> Address exists: ${MINIMAX_IP_NAME}"
fi

if ! gcloud compute addresses describe "${ALPHAZERO_IP_NAME}" \
  --region="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud compute addresses create "${ALPHAZERO_IP_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}"
else
  log " -> Address exists: ${ALPHAZERO_IP_NAME}"
fi

MINIMAX_IP="$(gcloud compute addresses describe "${MINIMAX_IP_NAME}" \
  --region="${REGION}" --project="${PROJECT_ID}" --format='get(address)')"
ALPHAZERO_IP="$(gcloud compute addresses describe "${ALPHAZERO_IP_NAME}" \
  --region="${REGION}" --project="${PROJECT_ID}" --format='get(address)')"

log "Step 6: Creating VMs (Container-Optimized OS + startup script)..."
PLACEHOLDER_IMAGE="docker.io/library/nginx:stable-alpine"
STARTUP_SCRIPT="${SCRIPT_DIR}/02_startup_script.sh"

if [ ! -f "${STARTUP_SCRIPT}" ]; then
  echo "02_startup_script.sh not found: ${STARTUP_SCRIPT}" >&2
  exit 1
fi

if ! gcloud compute instances describe "${MINIMAX_VM}" \
  --zone="${ZONE}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud compute instances create "${MINIMAX_VM}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --machine-type="${MINIMAX_MACHINE}" \
    --image-family=cos-stable \
    --image-project=cos-cloud \
    --tags="gomoku-deploy" \
    --address="${MINIMAX_IP_NAME}" \
    --service-account="${SA_EMAIL}" \
    --scopes="https://www.googleapis.com/auth/cloud-platform" \
    --metadata=enable-oslogin=TRUE,container-image="${PLACEHOLDER_IMAGE}",container-port=8080,container-env="MINIMAX_PORT=8080" \
    --metadata-from-file=startup-script="${STARTUP_SCRIPT}"
else
  log " -> Instance exists: ${MINIMAX_VM}"
fi

if ! gcloud compute instances describe "${ALPHAZERO_VM}" \
  --zone="${ZONE}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud compute instances create "${ALPHAZERO_VM}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --machine-type="${ALPHAZERO_MACHINE}" \
    --image-family=cos-stable \
    --image-project=cos-cloud \
    --tags="gomoku-deploy" \
    --address="${ALPHAZERO_IP_NAME}" \
    --service-account="${SA_EMAIL}" \
    --scopes="https://www.googleapis.com/auth/cloud-platform" \
    --metadata=enable-oslogin=TRUE,container-image="${PLACEHOLDER_IMAGE}",container-port=8080,container-env="" \
    --metadata-from-file=startup-script="${STARTUP_SCRIPT}"
else
  log " -> Instance exists: ${ALPHAZERO_VM}"
fi

log "Setup complete. Static external IPs:"
echo "  MINIMAX_IP=${MINIMAX_IP}"
echo "  ALPHAZERO_IP=${ALPHAZERO_IP}"

