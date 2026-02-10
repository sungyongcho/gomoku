#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ALPHAZERO_ROOT="$(cd "${INFRA_ROOT}/.." && pwd)"
REPO_ROOT="$(cd "${ALPHAZERO_ROOT}/.." && pwd)"

DOTENV_PATH="${DOTENV_PATH:-${REPO_ROOT}/.env}"
if [ ! -f "${DOTENV_PATH}" ]; then
  echo "env file not found. Set DOTENV_PATH or create ${REPO_ROOT}/.env" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${DOTENV_PATH}"
set +a

export PROJECT_ID="${GCP_PROJECT:?GCP_PROJECT is required}"
export REGION="${GCP_REGION:?GCP_REGION is required}"
export ZONE="${GCP_ZONE:?GCP_ZONE is required}"
export SA_NAME="${GCP_SA_NAME:?GCP_SA_NAME is required}"
export REPO_NAME="${GCP_REPO:?GCP_REPO is required}"
export USER_EMAIL="${GCP_USER_EMAIL:?GCP_USER_EMAIL is required}"
export BUCKET_NAME="${GCP_BUCKET_NAME:?GCP_BUCKET_NAME is required}"

export SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
export ARTIFACT_REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"

echo "Loaded Configuration:"
echo "  Project: ${PROJECT_ID}"
echo "  Region: ${REGION} (${ZONE})"
echo "  User: ${USER_EMAIL}"
echo "  Bucket: ${BUCKET_NAME}"
echo "  Service Account: ${SA_EMAIL}"
echo "-------------------------------------"

# Threading safeguards for shell-invoked Python/Numpy workloads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1
