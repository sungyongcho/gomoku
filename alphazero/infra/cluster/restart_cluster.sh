#!/usr/bin/env bash
set -Eeuo pipefail

# Resolve paths relative to this script so the command works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ALPHAZERO_ROOT="$(cd "${INFRA_ROOT}/.." && pwd)"
REPO_ROOT="$(cd "${ALPHAZERO_ROOT}/.." && pwd)"

# Load .env defaults (can be overridden by exported env variables).
DOTENV_PATH="${DOTENV_PATH:-${REPO_ROOT}/.env}"
if [ ! -f "${DOTENV_PATH}" ] && [ -f "${ALPHAZERO_ROOT}/.env" ]; then
  DOTENV_PATH="${ALPHAZERO_ROOT}/.env"
fi
if [ -f "${DOTENV_PATH}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${DOTENV_PATH}"
  set +a
fi

# Fix for GCP IAP Warning (requires numpy)
if command -v python3 &>/dev/null; then
    export CLOUDSDK_PYTHON=$(which python3)
fi

# ===== Configuration (required from env) =====
PROJECT="${GCP_PROJECT:?GCP_PROJECT is required}"
REGION="${GCP_REGION:?GCP_REGION is required}"
ZONE="${GCP_ZONE:?GCP_ZONE is required}"
REPO="${GCP_REPO:?GCP_REPO is required}"
CLUSTER_NAME="${GCP_CLUSTER_NAME:?GCP_CLUSTER_NAME is required}"
SSH_USER="${GCP_SSH_USER:?GCP_SSH_USER is required}"
CONTAINER_NAME="${GCP_CONTAINER_NAME:?GCP_CONTAINER_NAME is required}"
GPU_TAG="${GCP_GPU_TAG:?GCP_GPU_TAG is required}"
CPU_TAG="${GCP_CPU_TAG:?GCP_CPU_TAG is required}"
SA_NAME="${GCP_SA_NAME:?GCP_SA_NAME is required}"
HEAD_RESERVATION="${GCP_HEAD_RESERVATION:?GCP_HEAD_RESERVATION is required}"
SSH_PRIVATE_KEY="${GCP_SSH_PRIVATE_KEY:?GCP_SSH_PRIVATE_KEY is required}"

# Needed by the Python renderer in render_cluster_yaml().
export PROJECT REGION ZONE REPO CLUSTER_NAME SSH_USER CONTAINER_NAME
export GPU_TAG CPU_TAG SA_NAME HEAD_RESERVATION SSH_PRIVATE_KEY

CLUSTER_YAML_SOURCE="${CLUSTER_YAML_SOURCE:-${SCRIPT_DIR}/cluster_elo1800.yaml}"
CLUSTER_YAML="${CLUSTER_YAML:-${SCRIPT_DIR}/.cluster_elo1800.resolved.yaml}"

# Image URIs
IMAGE_GPU="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/gmk-ray:${GPU_TAG}"
IMAGE_CPU="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/gmk-ray:${CPU_TAG}"

# Flags
DO_DELETE="${DO_DELETE:-true}"   # Delete old tags in Artifact Registry
DO_BUILD="${DO_BUILD:-true}"     # Build & Push
DO_RESTART="${DO_RESTART:-true}" # Ray up (re-deploy)

log(){ echo -e "[\e[36m$(date +'%F %T')\e[0m] $*"; }

render_cluster_yaml() {
  if [ ! -f "${CLUSTER_YAML_SOURCE}" ]; then
    echo "cluster source not found: ${CLUSTER_YAML_SOURCE}" >&2
    exit 1
  fi
  python3 - "${CLUSTER_YAML_SOURCE}" "${CLUSTER_YAML}" <<'PY'
import os
import pathlib
import re
import sys

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
text = source.read_text(encoding="utf-8")

project = os.environ["PROJECT"]
region = os.environ["REGION"]
zone = os.environ["ZONE"]
repo = os.environ["REPO"]
cluster_name = os.environ["CLUSTER_NAME"]
ssh_user = os.environ["SSH_USER"]
container_name = os.environ["CONTAINER_NAME"]
ssh_private_key = os.environ["SSH_PRIVATE_KEY"]
sa_name = os.environ["SA_NAME"]
head_reservation = os.environ["HEAD_RESERVATION"]
gpu_tag = os.environ["GPU_TAG"]
cpu_tag = os.environ["CPU_TAG"]

new_gpu_image = f"{region}-docker.pkg.dev/{project}/{repo}/gmk-ray:{gpu_tag}"
new_cpu_image = f"{region}-docker.pkg.dev/{project}/{repo}/gmk-ray:{cpu_tag}"

def _extract(pattern: str) -> str | None:
    m = re.search(pattern, text, flags=re.M)
    return m.group(1) if m else None

old_project = _extract(r"^\s*project_id:\s*([^\s#]+)")
old_head_reservation = _extract(r"^\s*-\s*projects/[^/]+/reservations/([^\s#]+)")
if (not old_project) or old_project.startswith("${"):
    m = re.search(r"projects/([^/\s#]+)/", text)
    if m:
        old_project = m.group(1)

if old_project and (not old_project.startswith("${")) and old_project != project:
    text = text.replace(old_project, project)
if old_head_reservation and old_head_reservation != head_reservation:
    text = text.replace(old_head_reservation, head_reservation)

# Replace explicit ${GCP_*} placeholders if present in the template.
placeholder_map = {
    "${GCP_PROJECT}": project,
    "${GCP_REGION}": region,
    "${GCP_ZONE}": zone,
    "${GCP_REPO}": repo,
    "${GCP_CLUSTER_NAME}": cluster_name,
    "${GCP_SSH_USER}": ssh_user,
    "${GCP_SSH_PRIVATE_KEY}": ssh_private_key,
    "${GCP_CONTAINER_NAME}": container_name,
    "${GCP_SA_NAME}": sa_name,
    "${GCP_HEAD_RESERVATION}": head_reservation,
    "${GCP_GPU_TAG}": gpu_tag,
    "${GCP_CPU_TAG}": cpu_tag,
}
for k, v in placeholder_map.items():
    text = text.replace(k, v)

text = re.sub(r"^(\s*cluster_name:\s*).*$", rf"\1{cluster_name}", text, flags=re.M)
text = re.sub(r"^(\s*region:\s*).*$", rf"\1{region}", text, flags=re.M)
text = re.sub(r"^(\s*availability_zone:\s*).*$", rf"\1{zone}", text, flags=re.M)
text = re.sub(r"^(\s*project_id:\s*).*$", rf"\1{project}", text, flags=re.M)
text = re.sub(r"^(\s*ssh_user:\s*).*$", rf"\1{ssh_user}", text, flags=re.M)
text = re.sub(
    r"^(\s*ssh_private_key:\s*).*$",
    rf"\1{ssh_private_key}",
    text,
    flags=re.M,
)
text = re.sub(
    r"^(\s*container_name:\s*).*$",
    rf"\1{container_name}",
    text,
    flags=re.M,
)

text = re.sub(
    r"^(\s*head_image:\s*).*$",
    rf"\1{new_gpu_image}",
    text,
    flags=re.M,
)
text = re.sub(
    r"^(\s*worker_image:\s*).*$",
    rf"\1{new_cpu_image}",
    text,
    flags=re.M,
)
text = re.sub(
    r"^(\s*-\s*email:\s*)[^\s#]+",
    rf"\1{sa_name}@{project}.iam.gserviceaccount.com",
    text,
    flags=re.M,
)

target.write_text(text, encoding="utf-8")
PY
}

# ---------------------------------------------------------

# 0. Prep
log "Config: Project=$PROJECT, User=$SSH_USER, Cluster=$CLUSTER_NAME"
log "Render cluster yaml: ${CLUSTER_YAML_SOURCE} -> ${CLUSTER_YAML}"
render_cluster_yaml

# 1. Clean Registry (Optional)
if [ "${DO_DELETE}" = true ]; then
  log "Step 1: Purging all images from Artifact Registry repository: ${REPO}"
  REPO_PATH="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}"

  # List all image packages in the repository and delete them (includes all versions/tags)
  # Silence 2>/dev/null to hide 'NOT_FOUND' if the repo is already empty
  gcloud artifacts docker images list "${REPO_PATH}" --format="value(package)" 2>/dev/null \
    | sort -u \
    | xargs -r -I {} gcloud artifacts docker images delete {} --quiet --delete-tags || true
else
  log "Step 1: Skipping registry deletion"
fi

# 2. Build & Push (Using ZSTD compression for stability)
if [ "${DO_BUILD}" = true ]; then
  log "Step 2: Building & Pushing Images (zstd mode)..."

  log " -> GPU Image: ${IMAGE_GPU}"
  docker buildx build \
    --platform linux/amd64 \
    --output "type=image,compression=zstd,force-compression=true,push=true" \
    -t "${IMAGE_GPU}" \
    -f "${INFRA_ROOT}/image/Dockerfile.py313-cu124" \
    "${ALPHAZERO_ROOT}"

  log " -> CPU Image: ${IMAGE_CPU}"
  docker buildx build \
    --platform linux/amd64 \
    --output "type=image,compression=zstd,force-compression=true,push=true" \
    -t "${IMAGE_CPU}" \
    -f "${INFRA_ROOT}/image/Dockerfile.py313-cpu" \
    "${ALPHAZERO_ROOT}"
else
  log "Step 2: Skipping Build/Push"
fi

# 3. Clean Running Nodes (Docker RM)
log "Step 3: Cleaning up containers on running nodes..."

# Find instances belonging to this cluster
mapfile -t NODE_NAMES < <(
  gcloud compute instances list \
    --project "${PROJECT}" \
    --filter="name~^ray-${CLUSTER_NAME}- AND zone:${ZONE} AND status=RUNNING" \
    --format="value(name)"
)

if [ "${#NODE_NAMES[@]}" -eq 0 ]; then
  log " -> No running nodes found. Skipping cleanup."
else
  # Command to kill/remove container and prune images
  # Using 'sudo' because usually required for docker on GCP DLVM if user not in group.
  CLEANUP_CMD="sudo docker rm -f ${CONTAINER_NAME} 2>/dev/null || true; sudo docker image prune -a -f >/dev/null 2>&1 || true"

  for node in "${NODE_NAMES[@]}"; do
    log " -> Cleaning node: ${node}"
    gcloud compute ssh "${SSH_USER}@${node}" \
      --project "${PROJECT}" \
      --zone "${ZONE}" \
      --tunnel-through-iap \
      --quiet \
      --command="${CLEANUP_CMD}" \
      --ssh-flag="-o StrictHostKeyChecking=no" \
      --ssh-flag="-o UserKnownHostsFile=/dev/null"
  done
  # wait # No need to wait anymore, as SSH is sequential
fi

# 4. Ray Up (Re-deploy)
if [ "${DO_RESTART}" = true ]; then
  log "Step 4: Running 'ray up' to restart containers..."
  # --no-config-cache guarantees it checks the cloud state and updates
  ray up -y "${CLUSTER_YAML}" --no-config-cache
else
  log "Step 4: Skipping ray up"
fi

log "Done."
