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

CONTAINER_PORT="${CONTAINER_PORT:-8080}"

if [ -z "${CONTAINER_IMAGE}" ]; then
  echo "ERROR: 'container-image' metadata not set. Nothing to run." >&2
  exit 1
fi

echo "=== Startup script ==="
echo "  Image: ${CONTAINER_IMAGE}"
echo "  Port:  ${CONTAINER_PORT}"

# ---- Authenticate Docker to Artifact Registry ----
# COS ships with Docker pre-installed.  Authenticate via the instance SA.
ACCESS_TOKEN="$(curl -fsSL -H 'Metadata-Flavor: Google' \
  'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])' 2>/dev/null || true)"

if [ -z "${ACCESS_TOKEN}" ]; then
  # COS might not have python3; fallback to grep/sed
  ACCESS_TOKEN="$(curl -fsSL -H 'Metadata-Flavor: Google' \
    'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
    | grep -o '"access_token":"[^"]*"' | sed 's/"access_token":"//;s/"//')"
fi

REGISTRY_HOST="$(echo "${CONTAINER_IMAGE}" | cut -d/ -f1)"
echo "${ACCESS_TOKEN}" | docker login -u oauth2accesstoken --password-stdin "https://${REGISTRY_HOST}"

# ---- Stop & remove any previous container ----
docker stop gomoku-app 2>/dev/null || true
docker rm   gomoku-app 2>/dev/null || true

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
  --name gomoku-app \
  --restart unless-stopped \
  -p "${CONTAINER_PORT}:${CONTAINER_PORT}" \
  ${ENV_FLAGS} \
  "${CONTAINER_IMAGE}"

echo "=== Container started ==="
docker ps --filter name=gomoku-app --format '{{.Image}} | {{.Status}}'
