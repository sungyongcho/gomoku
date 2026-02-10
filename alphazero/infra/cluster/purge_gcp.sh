#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALPHAZERO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${ALPHAZERO_ROOT}/.." && pwd)"

DOTENV_PATH="${DOTENV_PATH:-${REPO_ROOT}/.env}"
if [ -f "${DOTENV_PATH}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${DOTENV_PATH}"
  set +a
fi

PROJECT="${GCP_PROJECT:-}"
REGION="${GCP_REGION:-}"
ZONE="${GCP_ZONE:-}"
REPO="${GCP_REPO:-}"
CLUSTER_NAME="${GCP_CLUSTER_NAME:-}"
HEAD_RESERVATION="${GCP_HEAD_RESERVATION:-}"
BUCKET="${GCP_BUCKET_NAME:-}"
NETWORK="${NETWORK:-default}"
ROUTER="${ROUTER:-${NETWORK}-router-ew4}"
NAT="${NAT:-${NETWORK}-nat-ew4}"

DELETE_PROJECT=false
YES=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [--delete-project] [--yes]

Default behavior (without --delete-project):
  - Delete Ray cluster instances in GCP_PROJECT
  - Delete reservation (GCP_HEAD_RESERVATION)
  - Delete common Ray firewall rules
  - Delete Cloud NAT + router
  - Delete Artifact Registry repository (GCP_REPO)
  - Delete GCS bucket (GCP_BUCKET_NAME)
  - Revoke local gcloud auth and remove ~/.config/gcloud

Options:
  --delete-project   Also delete the whole GCP project at the end
  --yes              Skip interactive confirmations
  -h, --help         Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --delete-project) DELETE_PROJECT=true; shift ;;
    --yes) YES=true; shift ;;
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

log() {
  printf '[%s] %s\n' "$(date +'%F %T')" "$*"
}

if [[ -z "${PROJECT}" ]]; then
  echo "GCP_PROJECT is required (from .env or env var)." >&2
  exit 1
fi

if [[ -z "${ZONE}" ]]; then
  echo "GCP_ZONE is required (from .env or env var)." >&2
  exit 1
fi

log "Target project: ${PROJECT}"
log "Target region/zone: ${REGION:-<unset>} / ${ZONE}"
log "Target cluster: ${CLUSTER_NAME:-<unset>}"
log "Target repo: ${REPO:-<unset>}"
log "Target bucket: ${BUCKET:-<unset>}"
log "Delete whole project: ${DELETE_PROJECT}"

confirm_or_exit "This will delete GCP resources and local gcloud auth state. Continue?"

if [[ -n "${CLUSTER_NAME}" ]]; then
  log "Deleting Ray cluster instances..."
  mapfile -t nodes < <(
    gcloud compute instances list \
      --project "${PROJECT}" \
      --filter="name~^ray-${CLUSTER_NAME}-" \
      --format="value(name)"
  )
  if [[ ${#nodes[@]} -gt 0 ]]; then
    gcloud compute instances delete "${nodes[@]}" \
      --project "${PROJECT}" \
      --zone "${ZONE}" \
      --quiet || true
  else
    log "No Ray instances found."
  fi
fi

if [[ -n "${HEAD_RESERVATION}" ]]; then
  log "Deleting reservation: ${HEAD_RESERVATION}"
  gcloud compute reservations delete "${HEAD_RESERVATION}" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --quiet || true
fi

log "Deleting common Ray firewall rules..."
for fw in ray-allow-internal ray-allow-head-from-home-v4 ray-allow-head-from-home-v6 ray-allow-iap-ssh; do
  gcloud compute firewall-rules delete "${fw}" \
    --project "${PROJECT}" \
    --quiet || true
done

if [[ -n "${REGION}" ]]; then
  log "Deleting Cloud NAT/router if present..."
  gcloud compute routers nats delete "${NAT}" \
    --project "${PROJECT}" \
    --region "${REGION}" \
    --router "${ROUTER}" \
    --quiet || true

  gcloud compute routers delete "${ROUTER}" \
    --project "${PROJECT}" \
    --region "${REGION}" \
    --quiet || true
fi

if [[ -n "${REPO}" && -n "${REGION}" ]]; then
  log "Deleting Artifact Registry repo: ${REPO}"
  gcloud artifacts repositories delete "${REPO}" \
    --project "${PROJECT}" \
    --location "${REGION}" \
    --quiet || true
fi

if [[ -n "${BUCKET}" ]]; then
  log "Deleting GCS bucket and all objects: gs://${BUCKET}"
  gcloud storage rm --recursive "gs://${BUCKET}/**" --project "${PROJECT}" || true
  gcloud storage buckets delete "gs://${BUCKET}" --project "${PROJECT}" || true
fi

if [[ "${DELETE_PROJECT}" == true ]]; then
  confirm_or_exit "Final confirmation: delete entire project '${PROJECT}'?"
  log "Deleting project: ${PROJECT}"
  gcloud projects delete "${PROJECT}" --quiet || true
fi

log "Revoking local gcloud auth..."
gcloud auth revoke --all --quiet || true
gcloud auth application-default revoke --quiet || true

log "Removing local gcloud config directory..."
rm -rf "${HOME}/.config/gcloud"

log "Done."
