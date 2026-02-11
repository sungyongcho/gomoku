#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/deploy_env_config.sh"

log(){ echo -e "[\e[36m$(date +'%F %T')\e[0m] $*"; }

YES=false
DELETE_SA=false
DELETE_REPO=false
PURGE_ALL_IMAGES=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [--yes] [--delete-sa] [--delete-repo] [--purge-all-images]

Deletes deploy resources created by deploy/setup.sh:
  - VMs (${MINIMAX_VM}, ${ALPHAZERO_VM})
  - Static external IPs (${MINIMAX_VM}-ip, ${ALPHAZERO_VM}-ip)
  - Firewall rules (gomoku-deploy-ws-8080, gomoku-deploy-ssh-iap)
  - Artifact Registry images (default: gomoku-minimax, gomoku-alphazero)

Options:
  --yes               Skip interactive confirmation
  --delete-sa         Also delete the deploy service account (${SA_EMAIL})
  --delete-repo       Also delete the Artifact Registry repository (${REPO_NAME})
  --purge-all-images  Delete ALL images in the Artifact Registry repo (not just gomoku-minimax/alphazero)
  -h, --help          Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes) YES=true; shift ;;
    --delete-sa) DELETE_SA=true; shift ;;
    --delete-repo) DELETE_REPO=true; shift ;;
    --purge-all-images) PURGE_ALL_IMAGES=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

confirm_or_exit() {
  local prompt="$1"
  if [[ "${YES}" == true ]]; then
    return 0
  fi
  read -r -p "${prompt} [y/N]: " ans
  case "${ans}" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 1 ;;
  esac
}

if ! command -v gcloud &>/dev/null; then
  echo "Missing required command: gcloud" >&2
  exit 1
fi

log "Target project: ${PROJECT_ID}"
log "Target region/zone: ${REGION} / ${ZONE}"
log "Target deploy repo: ${REPO_NAME}"
log "Target VMs: minimax=${MINIMAX_VM}, alphazero=${ALPHAZERO_VM}"
log "Delete service account: ${DELETE_SA}"
log "Delete Artifact Registry repo: ${DELETE_REPO}"
log "Purge all images: ${PURGE_ALL_IMAGES}"

confirm_or_exit "This will delete GCP deploy resources. Continue?"

log "Step 1: Deleting VMs (if present)..."
gcloud compute instances delete "${MINIMAX_VM}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --quiet >/dev/null 2>&1 || true
gcloud compute instances delete "${ALPHAZERO_VM}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --quiet >/dev/null 2>&1 || true

log "Step 2: Releasing static external IPs (if present)..."
MINIMAX_IP_NAME="${MINIMAX_VM}-ip"
ALPHAZERO_IP_NAME="${ALPHAZERO_VM}-ip"
gcloud compute addresses delete "${MINIMAX_IP_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --quiet >/dev/null 2>&1 || true
gcloud compute addresses delete "${ALPHAZERO_IP_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --quiet >/dev/null 2>&1 || true

log "Step 3: Deleting firewall rules (if present)..."
for fw in gomoku-deploy-ws-8080 gomoku-deploy-ssh-iap; do
  gcloud compute firewall-rules delete "${fw}" \
    --project "${PROJECT_ID}" \
    --quiet >/dev/null 2>&1 || true
done

log "Step 4: Deleting Artifact Registry images..."
if [[ "${PURGE_ALL_IMAGES}" == true ]]; then
  # Delete everything in the repo.
  gcloud artifacts docker images list "${ARTIFACT_REGISTRY}" --format="value(package)" 2>/dev/null \
    | sort -u \
    | xargs -r -I {} gcloud artifacts docker images delete {} --quiet --delete-tags >/dev/null 2>&1 || true
else
  # Delete only the deploy packages we create by default.
  for pkg in gomoku-minimax gomoku-alphazero; do
    gcloud artifacts docker images delete "${ARTIFACT_REGISTRY}/${pkg}" \
      --quiet --delete-tags >/dev/null 2>&1 || true
  done
fi

if [[ "${DELETE_REPO}" == true ]]; then
  log "Step 5: Deleting Artifact Registry repository..."
  gcloud artifacts repositories delete "${REPO_NAME}" \
    --project "${PROJECT_ID}" \
    --location "${REGION}" \
    --quiet >/dev/null 2>&1 || true
fi

if [[ "${DELETE_SA}" == true ]]; then
  log "Step 6: Deleting service account..."
  gcloud iam service-accounts delete "${SA_EMAIL}" \
    --project "${PROJECT_ID}" \
    --quiet >/dev/null 2>&1 || true
fi

log "Teardown complete."

