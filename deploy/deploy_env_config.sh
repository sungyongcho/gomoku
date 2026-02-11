#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DOTENV_PATH="${DOTENV_PATH:-${REPO_ROOT}/.env}"
if [ ! -f "${DOTENV_PATH}" ]; then
  echo "env file not found. Set DOTENV_PATH or create ${REPO_ROOT}/.env" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${DOTENV_PATH}"
set +a

# ===== Required config (from .env) =====
export PROJECT_ID="${DEPLOY_GCP_PROJECT:?DEPLOY_GCP_PROJECT is required}"
export REGION="${DEPLOY_GCP_REGION:?DEPLOY_GCP_REGION is required}"
export ZONE="${DEPLOY_GCP_ZONE:?DEPLOY_GCP_ZONE is required}"
export REPO_NAME="${DEPLOY_GCP_REPO:?DEPLOY_GCP_REPO is required}"

export MINIMAX_VM="${DEPLOY_MINIMAX_VM:?DEPLOY_MINIMAX_VM is required}"
export ALPHAZERO_VM="${DEPLOY_ALPHAZERO_VM:?DEPLOY_ALPHAZERO_VM is required}"

export MINIMAX_MACHINE="${DEPLOY_MINIMAX_MACHINE:?DEPLOY_MINIMAX_MACHINE is required}"
export ALPHAZERO_MACHINE="${DEPLOY_ALPHAZERO_MACHINE:?DEPLOY_ALPHAZERO_MACHINE is required}"

export DEPLOY_DOMAIN="${DEPLOY_DOMAIN:?DEPLOY_DOMAIN is required}"
export SA_NAME="${DEPLOY_SA_NAME:?DEPLOY_SA_NAME is required}"
export USER_EMAIL="${DEPLOY_USER_EMAIL:?DEPLOY_USER_EMAIL is required}"

# ===== Cloudflare Worker config (optional – required only for 04_deploy_cloudflare.sh) =====
export DEPLOY_MINIMAX_IP="${DEPLOY_MINIMAX_IP:-}"
export DEPLOY_ALPHAZERO_IP="${DEPLOY_ALPHAZERO_IP:-}"

# ===== Derived config =====
export SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
export ARTIFACT_REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"

# Threading safeguards (useful when scripts invoke Python/PyTorch locally).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

_mask_len() {
  local s="${1:-}"
  if [ -z "${s}" ]; then
    echo "unset"
  else
    echo "set(len=${#s})"
  fi
}

echo "Loaded Configuration:"
echo "  Project: ${PROJECT_ID}"
echo "  Region: ${REGION} (${ZONE})"
echo "  Domain: ${DEPLOY_DOMAIN}"
echo "  VMs: minimax=${MINIMAX_VM}, alphazero=${ALPHAZERO_VM}"
echo "  Machines: minimax=${MINIMAX_MACHINE}, alphazero=${ALPHAZERO_MACHINE}"
echo "  VM IPs: minimax=${DEPLOY_MINIMAX_IP:-unset}, alphazero=${DEPLOY_ALPHAZERO_IP:-unset}"
echo "  Artifact Registry: ${ARTIFACT_REGISTRY}"
echo "  User: ${USER_EMAIL}"
echo "  Service Account: ${SA_EMAIL}"
echo "  Cloudflare: account_id=$(_mask_len "${CLOUDFLARE_ACCOUNT_ID:-}"), api_token=$(_mask_len "${CLOUDFLARE_API_TOKEN:-}")"
echo "-------------------------------------"

