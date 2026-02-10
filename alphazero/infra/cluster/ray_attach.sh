#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_DIR="${SCRIPT_DIR}"
ALPHAZERO_ROOT="$(cd "${CLUSTER_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${ALPHAZERO_ROOT}/.." && pwd)"
CLUSTER_TEMPLATE="${CLUSTER_DIR}/cluster_elo1800.yaml"
CLUSTER_RESOLVED="${CLUSTER_DIR}/.cluster_elo1800.resolved.yaml"

DOTENV_PATH="${DOTENV_PATH:-${REPO_ROOT}/.env}"
if [ -f "${DOTENV_PATH}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${DOTENV_PATH}"
  set +a
fi

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

export PROJECT REGION ZONE REPO CLUSTER_NAME SSH_USER CONTAINER_NAME
export GPU_TAG CPU_TAG SA_NAME HEAD_RESERVATION SSH_PRIVATE_KEY

if [ ! -f "${CLUSTER_TEMPLATE}" ]; then
  echo "Cluster template not found: ${CLUSTER_TEMPLATE}" >&2
  exit 1
fi

render_cluster_yaml() {
  local source_yaml="$1"
  local target_yaml="$2"
  python3 - "${source_yaml}" "${target_yaml}" <<'PY'
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
text = re.sub(r"^(\s*ssh_private_key:\s*).*$", rf"\1{ssh_private_key}", text, flags=re.M)
text = re.sub(r"^(\s*container_name:\s*).*$", rf"\1{container_name}", text, flags=re.M)
text = re.sub(r"^(\s*head_image:\s*).*$", rf"\1{new_gpu_image}", text, flags=re.M)
text = re.sub(r"^(\s*worker_image:\s*).*$", rf"\1{new_cpu_image}", text, flags=re.M)
text = re.sub(
    r"^(\s*-\s*email:\s*)[^\s#]+",
    rf"\1{sa_name}@{project}.iam.gserviceaccount.com",
    text,
    flags=re.M,
)

target.write_text(text, encoding="utf-8")
PY
}

TMP_RESOLVED="$(mktemp "${CLUSTER_DIR}/.cluster_elo1800.resolved.yaml.tmp.XXXXXX")"
cleanup_tmp() {
  rm -f "${TMP_RESOLVED}" 2>/dev/null || true
}
trap cleanup_tmp EXIT

render_cluster_yaml "${CLUSTER_TEMPLATE}" "${TMP_RESOLVED}"

if [ -f "${CLUSTER_RESOLVED}" ] && cmp -s "${TMP_RESOLVED}" "${CLUSTER_RESOLVED}"; then
  echo "[ray_attach] Resolved yaml is up to date: ${CLUSTER_RESOLVED}"
else
  rm -f "${CLUSTER_RESOLVED}" 2>/dev/null || true
  mv "${TMP_RESOLVED}" "${CLUSTER_RESOLVED}"
  echo "[ray_attach] Generated resolved yaml: ${CLUSTER_RESOLVED}"
fi

cleanup_tmp
trap - EXIT

if [ -x "${ALPHAZERO_ROOT}/.venv/bin/ray" ]; then
  RAY_BIN="${ALPHAZERO_ROOT}/.venv/bin/ray"
elif command -v ray >/dev/null 2>&1; then
  RAY_BIN="$(command -v ray)"
else
  echo "ray command not found. Activate alphazero venv first." >&2
  exit 1
fi

exec "${RAY_BIN}" attach "$@" "${CLUSTER_RESOLVED}"
