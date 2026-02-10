#!/usr/bin/env bash
set -euo pipefail

# Resolve paths and load .env defaults.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ALPHAZERO_ROOT="$(cd "${INFRA_ROOT}/.." && pwd)"
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

# --- 설정 (환경변수로 override 가능) ---
# cluster_elo1800.yaml 의 head_gpu 노드 설정에 맞춤

# 예약 이름 (YAML의 주석된 reservation name과 일치시키거나, 새로 정의)
RES="${RES:-${GCP_HEAD_RESERVATION:?GCP_HEAD_RESERVATION is required}}"
ZONE="${ZONE:-${GCP_ZONE:?GCP_ZONE is required}}"
MT="${MT:-g2-standard-16}"
ACC_TYPE="${ACC_TYPE:-nvidia-l4}"
ACC_PER_VM="${ACC_PER_VM:-1}"
VM_COUNT="${VM_COUNT:-1}"
PROJECT="${PROJECT:-${GCP_PROJECT:?GCP_PROJECT is required}}"

# 재시도 간격(초)
BASE_SLEEP="${BASE_SLEEP:-15}"
# -------------------------------------

GCLOUD_PROJECT_ARGS=()
if [[ -n "${PROJECT}" ]]; then
  GCLOUD_PROJECT_ARGS=(--project="${PROJECT}")
fi

MODE="create"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--delete) MODE="delete"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

if [[ "$MODE" == "delete" ]]; then
    echo "[$(date -u +%H:%M:%S)] Deleting reservation '$RES' in zone '$ZONE'..."
    if gcloud compute reservations delete "$RES" --zone="$ZONE" "${GCLOUD_PROJECT_ARGS[@]}" --quiet; then
        echo "[$(date -u +%H:%M:%S)] Successfully deleted reservation '$RES'."
        exit 0
    else
        echo "[$(date -u +%H:%M:%S)] Failed to delete reservation '$RES' (it might not exist)."
        exit 1
    fi
fi

while true; do
  echo "[$(date -u +%H:%M:%S)] create attempt: $RES ($MT, $ACC_TYPE x$ACC_PER_VM)..."

  # gcloud compute reservations create 명령
  if gcloud compute reservations create "$RES" \
      --zone="$ZONE" \
      "${GCLOUD_PROJECT_ARGS[@]}" \
      --machine-type="$MT" \
      --accelerator="type=${ACC_TYPE},count=${ACC_PER_VM}" \
      --vm-count="$VM_COUNT" \
      --require-specific-reservation; then
    echo "[$(date -u +%H:%M:%S)] success: Reservation '$RES' created."
    exit 0
  fi

  # 이미 존재하면 성공 처리
  if gcloud compute reservations describe "$RES" --zone="$ZONE" "${GCLOUD_PROJECT_ARGS[@]}" >/dev/null 2>&1; then
    echo "[$(date -u +%H:%M:%S)] Reservation '$RES' already exists → done"
    exit 0
  fi

  echo "[$(date -u +%H:%M:%S)] failed/busy, retrying in ${BASE_SLEEP}s..."
  sleep "$BASE_SLEEP"
done
