#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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
require_cmd docker

# Flags (match training infra style)
DO_DELETE="${DO_DELETE:-false}"   # Purge images in Artifact Registry (dangerous)
DO_BUILD="${DO_BUILD:-true}"      # Build & push images
DO_PRUNE="${DO_PRUNE:-true}"      # Prune old docker artifacts on VMs before update (best-effort)
DO_UPDATE="${DO_UPDATE:-true}"    # Update VM containers
DO_VERIFY="${DO_VERIFY:-true}"    # SSH in and verify docker status (best-effort)

# Tagging
if command -v git &>/dev/null && git -C "${REPO_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
  DEFAULT_TAG="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || true)"
else
  DEFAULT_TAG=""
fi
DEFAULT_TAG="${DEFAULT_TAG:-$(date +'%Y%m%d-%H%M%S')}"
IMAGE_TAG="${IMAGE_TAG:-${DEFAULT_TAG}}"

PKG_MINIMAX="${PKG_MINIMAX:-gomoku-minimax}"
PKG_ALPHAZERO="${PKG_ALPHAZERO:-gomoku-alphazero}"

IMAGE_MINIMAX="${IMAGE_MINIMAX:-${ARTIFACT_REGISTRY}/${PKG_MINIMAX}:${IMAGE_TAG}}"
IMAGE_ALPHAZERO="${IMAGE_ALPHAZERO:-${ARTIFACT_REGISTRY}/${PKG_ALPHAZERO}:${IMAGE_TAG}}"

# Dockerfiles
MINIMAX_DOCKERFILE="${MINIMAX_DOCKERFILE:-${REPO_ROOT}/minimax/Dockerfile.prod}"
if [ ! -f "${MINIMAX_DOCKERFILE}" ]; then
  echo "minimax prod Dockerfile not found: ${MINIMAX_DOCKERFILE}" >&2
  exit 1
fi

if [ -n "${ALPHAZERO_DOCKERFILE:-}" ]; then
  :
elif [ -f "${REPO_ROOT}/alphazero/infra/image/Dockerfile.prod" ]; then
  ALPHAZERO_DOCKERFILE="${REPO_ROOT}/alphazero/infra/image/Dockerfile.prod"
else
  echo "alphazero prod Dockerfile not found. Expected:" >&2
  echo "  - alphazero/infra/image/Dockerfile.prod" >&2
  exit 1
fi

# ---------------------------------------------------------

log "Config: Project=${PROJECT_ID}, Region=${REGION}, Zone=${ZONE}"
log "Images:"
log "  minimax:   ${IMAGE_MINIMAX}"
log "  alphazero: ${IMAGE_ALPHAZERO}"
log "Flags: DO_DELETE=${DO_DELETE}, DO_BUILD=${DO_BUILD}, DO_PRUNE=${DO_PRUNE}, DO_UPDATE=${DO_UPDATE}, DO_VERIFY=${DO_VERIFY}"

log "Step 1: Authenticate Docker to Artifact Registry..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

if [ "${DO_DELETE}" = true ]; then
  log "Step 2: Purging deploy packages from Artifact Registry (dangerous)..."
  for pkg in "${PKG_MINIMAX}" "${PKG_ALPHAZERO}"; do
    # Delete all tags/versions under the package (ignore NOT_FOUND).
    gcloud artifacts docker images delete "${ARTIFACT_REGISTRY}/${pkg}" \
      --quiet --delete-tags >/dev/null 2>&1 || true
    log "  -> Purged ${pkg}"
  done
else
  log "Step 2: Skipping registry purge"
fi

log "Step 3: Prepare AlphaZero checkpoint in build context..."
CKPT_SOURCE="${CKPT_SOURCE:-${REPO_ROOT}/alphazero/runs/elo1800-gcp-v4/ckpt/champion.pt}"
CKPT_DEST="${CKPT_DEST:-${REPO_ROOT}/alphazero/models/champion.pt}"
if [ ! -f "${CKPT_SOURCE}" ]; then
  echo "Checkpoint source not found: ${CKPT_SOURCE}" >&2
  exit 1
fi
rm -f "${CKPT_DEST}"
cp -L "${CKPT_SOURCE}" "${CKPT_DEST}"

if [ "${DO_BUILD}" = true ]; then
  log "Step 4: Build & push minimax..."
  docker buildx build \
    --platform linux/amd64 \
    --output "type=image,compression=zstd,force-compression=true,push=true" \
    -t "${IMAGE_MINIMAX}" \
    -f "${MINIMAX_DOCKERFILE}" \
    "${REPO_ROOT}/minimax"

  log "Step 5: Build & push alphazero..."
  docker buildx build \
    --platform linux/amd64 \
    --output "type=image,compression=zstd,force-compression=true,push=true" \
    -t "${IMAGE_ALPHAZERO}" \
    -f "${ALPHAZERO_DOCKERFILE}" \
    "${REPO_ROOT}/alphazero"
else
  log "Step 4-5: Skipping build/push"
fi

if [ "${DO_UPDATE}" = true ]; then
  log "Step 6: Update container metadata on VMs and restart..."

  if [ "${DO_PRUNE}" = true ]; then
    log " -> Pruning old docker images/artifacts on VMs (best-effort)..."
    for vm in "${MINIMAX_VM}" "${ALPHAZERO_VM}"; do
      set +e
      gcloud compute ssh "${vm}" \
        --project="${PROJECT_ID}" \
        --zone="${ZONE}" \
        --command "sudo docker container prune -f >/dev/null 2>&1 || true; sudo docker image prune -af >/dev/null 2>&1 || true; sudo docker builder prune -af >/dev/null 2>&1 || true; sudo docker system df || true" \
        >/dev/null 2>&1
      rc=$?
      set -e
      if [ "${rc}" -ne 0 ]; then
        log "   -> ${vm}: prune skipped (ssh unavailable or docker not ready)"
      else
        log "   -> ${vm}: prune completed"
      fi
    done
  else
    log " -> Skipping VM docker prune"
  fi

  STARTUP_SCRIPT="${SCRIPT_DIR}/docker_container_startup_script.sh"
  if [ ! -f "${STARTUP_SCRIPT}" ]; then
    echo "docker_container_startup_script.sh not found: ${STARTUP_SCRIPT}" >&2
    exit 1
  fi

  log " -> minimax: ${MINIMAX_VM}"
  gcloud compute instances add-metadata "${MINIMAX_VM}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --metadata=container-image="${IMAGE_MINIMAX}",container-port=8080,container-name="${MINIMAX_VM}",container-env="MINIMAX_PORT=8080"
  gcloud compute instances add-metadata "${MINIMAX_VM}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --metadata-from-file=startup-script="${STARTUP_SCRIPT}"

  # Defaults match the Dockerfile; env overrides are optional.
  # Use DEPLOY_ALPHAZERO_* to avoid conflicts with training env vars.
  ALPHAZERO_CONFIG_VALUE="${DEPLOY_ALPHAZERO_CONFIG:-configs/deploy.yaml}"
  ALPHAZERO_DEVICE_VALUE="${DEPLOY_ALPHAZERO_DEVICE:-cpu}"
  ALPHAZERO_THREADS_VALUE="${DEPLOY_ALPHAZERO_THREADS:-}"
  ALPHAZERO_OMP_THREADS_VALUE="${DEPLOY_ALPHAZERO_OMP_THREADS:-${ALPHAZERO_THREADS_VALUE}}"
  ALPHAZERO_MKL_THREADS_VALUE="${DEPLOY_ALPHAZERO_MKL_THREADS:-${ALPHAZERO_THREADS_VALUE}}"
  ALPHAZERO_TORCH_THREADS_VALUE="${DEPLOY_ALPHAZERO_TORCH_THREADS:-${ALPHAZERO_THREADS_VALUE}}"
  ALPHAZERO_CPUS_VALUE="${DEPLOY_ALPHAZERO_CPUS:-}"
  ALPHAZERO_CPUSET_VALUE="${DEPLOY_ALPHAZERO_CPUSET:-}"
  AZ_ENV="ALPHAZERO_CONFIG=${ALPHAZERO_CONFIG_VALUE},ALPHAZERO_DEVICE=${ALPHAZERO_DEVICE_VALUE}"
  if [ -n "${ALPHAZERO_OMP_THREADS_VALUE}" ]; then
    AZ_ENV="${AZ_ENV},OMP_NUM_THREADS=${ALPHAZERO_OMP_THREADS_VALUE}"
  fi
  if [ -n "${ALPHAZERO_MKL_THREADS_VALUE}" ]; then
    AZ_ENV="${AZ_ENV},MKL_NUM_THREADS=${ALPHAZERO_MKL_THREADS_VALUE}"
  fi
  if [ -n "${ALPHAZERO_TORCH_THREADS_VALUE}" ]; then
    AZ_ENV="${AZ_ENV},TORCH_NUM_THREADS=${ALPHAZERO_TORCH_THREADS_VALUE}"
  fi
  if [ -n "${ALPHAZERO_INFER_BACKEND}" ]; then
    AZ_ENV="${AZ_ENV},ALPHAZERO_INFER_BACKEND=${ALPHAZERO_INFER_BACKEND}"
  fi
  AZ_META="container-image=${IMAGE_ALPHAZERO},container-port=8080,container-name=${ALPHAZERO_VM}"
  if [ -n "${ALPHAZERO_CPUS_VALUE}" ]; then
    AZ_META="${AZ_META},container-cpus=${ALPHAZERO_CPUS_VALUE}"
  fi
  if [ -n "${ALPHAZERO_CPUSET_VALUE}" ]; then
    AZ_META="${AZ_META},container-cpuset=${ALPHAZERO_CPUSET_VALUE}"
  fi

  log " -> alphazero: ${ALPHAZERO_VM}"
  gcloud compute instances add-metadata "${ALPHAZERO_VM}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --metadata="${AZ_META}"
  REMOVE_AZ_KEYS=()
  if [ -z "${ALPHAZERO_CPUS_VALUE}" ]; then
    REMOVE_AZ_KEYS+=("container-cpus")
  fi
  if [ -z "${ALPHAZERO_CPUSET_VALUE}" ]; then
    REMOVE_AZ_KEYS+=("container-cpuset")
  fi
  if [ "${#REMOVE_AZ_KEYS[@]}" -gt 0 ]; then
    AZ_REMOVE_KEYS_CSV="$(IFS=,; echo "${REMOVE_AZ_KEYS[*]}")"
    gcloud compute instances remove-metadata "${ALPHAZERO_VM}" \
      --project="${PROJECT_ID}" \
      --zone="${ZONE}" \
      --keys="${AZ_REMOVE_KEYS_CSV}" >/dev/null 2>&1 || true
  fi
  # container-env contains commas which conflict with gcloud's --metadata delimiter.
  # Pass it via --metadata-from-file to avoid the issue.
  AZ_ENV_FILE="$(mktemp)"
  echo "${AZ_ENV}" > "${AZ_ENV_FILE}"
  gcloud compute instances add-metadata "${ALPHAZERO_VM}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --metadata-from-file=container-env="${AZ_ENV_FILE}"
  rm -f "${AZ_ENV_FILE}"
  gcloud compute instances add-metadata "${ALPHAZERO_VM}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --metadata-from-file=startup-script="${STARTUP_SCRIPT}"

  log " -> Resetting VMs to re-run startup script..."
  gcloud compute instances reset "${MINIMAX_VM}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" --quiet
  gcloud compute instances reset "${ALPHAZERO_VM}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" --quiet

  log " -> Waiting 30s for VMs to boot..."
  sleep 30
else
  log "Step 6: Skipping VM updates"
fi

if [ "${DO_VERIFY}" = true ]; then
  log "Step 7: Verify containers via SSH (best-effort)..."

  set +e
  gcloud compute ssh "${MINIMAX_VM}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --command "sudo docker ps --format 'minimax: {{.Image}} | {{.Status}}'" 2>/dev/null
  gcloud compute ssh "${ALPHAZERO_VM}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --command "sudo docker ps --format 'alphazero: {{.Image}} | {{.Status}}'" 2>/dev/null
  set -e
else
  log "Step 7: Skipping verification"
fi

log "Deploy complete."
