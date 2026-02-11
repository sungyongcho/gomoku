#!/bin/bash
# startup_script.sh — runs on every COS VM boot.
# Pulls the container image specified in instance metadata and runs it.

set -euo pipefail

# ---- Helper ----
metadata() {
  curl -fsSL -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null || echo ""
}

# ---- Read metadata ----
CONTAINER_IMAGE="$(metadata container-image)"
CONTAINER_PORT="$(metadata container-port)"
CONTAINER_ENV_JSON="$(metadata container-env)"
CONTAINER_NAME="$(metadata container-name)"

CONTAINER_PORT="${CONTAINER_PORT:-8080}"
CONTAINER_NAME="${CONTAINER_NAME:-gomoku-app}"

if [ -z "${CONTAINER_IMAGE}" ]; then
  echo "ERROR: 'container-image' metadata not set. Nothing to run." >&2
  exit 1
fi

echo "=== Startup script ==="
echo "  Image: ${CONTAINER_IMAGE}"
echo "  Port:  ${CONTAINER_PORT}"
echo "  Name:  ${CONTAINER_NAME}"

# ---- Configure Docker to use writable directory ----
# COS has read-only /root/.docker, so use /tmp
export DOCKER_CONFIG=/tmp/.docker
mkdir -p "${DOCKER_CONFIG}"

# ---- Authenticate to Artifact Registry ----
# Get access token from metadata service (gcloud is not in PATH during startup)
ACCESS_TOKEN="$(curl -fsSL -H 'Metadata-Flavor: Google' \
  'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
  | grep -o '"access_token":"[^"]*"' | sed 's/"access_token":"//;s/"//')"

REGISTRY_HOST="$(echo "${CONTAINER_IMAGE}" | cut -d/ -f1)"
echo "${ACCESS_TOKEN}" | docker login -u oauth2accesstoken --password-stdin "https://${REGISTRY_HOST}"

# ---- Stop & remove any previous container ----
docker stop "${CONTAINER_NAME}" 2>/dev/null || true
docker rm   "${CONTAINER_NAME}" 2>/dev/null || true

# ---- Pull latest image ----
docker pull "${CONTAINER_IMAGE}"

# ---- Build -e flags from JSON metadata ----
ENV_FLAGS=""
if [ -n "${CONTAINER_ENV_JSON}" ]; then
  # Expects comma-separated KEY=VALUE pairs, e.g. "FOO=bar,BAZ=qux"
  IFS=',' read -ra PAIRS <<< "${CONTAINER_ENV_JSON}"
  for pair in "${PAIRS[@]}"; do
    ENV_FLAGS="${ENV_FLAGS} -e ${pair}"
  done
fi

# ---- Run ----
# shellcheck disable=SC2086
docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  -p "${CONTAINER_PORT}:${CONTAINER_PORT}" \
  ${ENV_FLAGS} \
  "${CONTAINER_IMAGE}"

echo "=== Container started ==="
docker ps --filter name="${CONTAINER_NAME}" --format '{{.Image}} | {{.Status}}'
