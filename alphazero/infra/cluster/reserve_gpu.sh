#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALPHAZERO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${ALPHAZERO_ROOT}/.." && pwd)"

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

# Reservation parameters (override via env if needed).
PROJECT="${GCP_PROJECT:?GCP_PROJECT is required}"
ZONE="${GCP_ZONE:?GCP_ZONE is required}"
RES="${RES:-${GCP_HEAD_RESERVATION:?GCP_HEAD_RESERVATION is required}}"
MT="${MT:-g2-standard-16}"
ACC_TYPE="${ACC_TYPE:-nvidia-l4}"
ACC_PER_VM="${ACC_PER_VM:-1}"
VM_COUNT="${VM_COUNT:-1}"
BASE_SLEEP="${BASE_SLEEP:-15}"

MODE="create"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -d|--delete) MODE="delete"; shift ;;
    -h|--help)
      cat <<EOF
Usage: $(basename "$0") [--delete]

Creates (default) or deletes (--delete) the specific GPU reservation.

Env inputs:
  GCP_PROJECT (required)
  GCP_ZONE (required)
  GCP_HEAD_RESERVATION (required unless RES is set)
Optional overrides:
  RES MT ACC_TYPE ACC_PER_VM VM_COUNT BASE_SLEEP
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

log() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"
}

if [[ "$MODE" == "delete" ]]; then
  log "Deleting reservation '${RES}' in zone '${ZONE}'..."
  if gcloud compute reservations delete "${RES}" \
      --zone="${ZONE}" \
      --project="${PROJECT}" \
      --quiet; then
    log "Deleted reservation '${RES}'."
    exit 0
  fi

  log "Failed to delete reservation '${RES}' (it may not exist)."
  exit 1
fi

while true; do
  log "Create attempt: ${RES} (${MT}, ${ACC_TYPE} x${ACC_PER_VM})..."

  if gcloud compute reservations create "${RES}" \
      --zone="$ZONE" \
      --project="${PROJECT}" \
      --machine-type="$MT" \
      --accelerator="type=${ACC_TYPE},count=${ACC_PER_VM}" \
      --vm-count="$VM_COUNT" \
      --require-specific-reservation; then
    log "Reservation '${RES}' created."
    exit 0
  fi

  # If it already exists, treat as success.
  if gcloud compute reservations describe "${RES}" \
      --zone="${ZONE}" \
      --project="${PROJECT}" >/dev/null 2>&1; then
    log "Reservation '${RES}' already exists."
    exit 0
  fi

  log "Failed or capacity busy; retrying in ${BASE_SLEEP}s..."
  sleep "$BASE_SLEEP"
done
