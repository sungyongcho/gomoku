#!/usr/bin/env bash
set -Eeuo pipefail
# ============================================================
# 02_deploy_alphazero.sh
#
# Sync the AlphaZero source and one checkpoint to the Oracle A1 instance, build the
# image there (native aarch64, no cross-compilation or registry) and (re)start the
# container. Re-run for every engine or model update; the build cache keeps it fast.
#
# The model is chosen in .env:
#   DEPLOY_ALPHAZERO_A1_CONFIG       YAML under alphazero/ (default configs/deploy-v5.yaml)
#   DEPLOY_ALPHAZERO_A1_CHECKPOINT   local checkpoint file that matches it
#
# Flags:
#   DO_BUILD=false   skip the image build (restart only)
#   NO_CACHE=true    force a clean image build
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/alphazero_oracle_env_config.sh"

DO_BUILD="${DO_BUILD:-true}"
NO_CACHE="${NO_CACHE:-false}"
ALPHAZERO_SRC="${REPO_ROOT}/alphazero"

command -v rsync >/dev/null 2>&1 || { echo "Missing required command: rsync" >&2; exit 1; }
[ -f "${ALPHAZERO_SRC}/infra/image/Dockerfile.prod" ] || { echo "alphazero prod Dockerfile not found: ${ALPHAZERO_SRC}/infra/image/Dockerfile.prod" >&2; exit 1; }
[ -f "${ALPHAZERO_SRC}/${A1_CONFIG}" ] || { echo "Config not found: ${ALPHAZERO_SRC}/${A1_CONFIG}" >&2; exit 1; }
[ -n "${A1_CHECKPOINT}" ] || { echo "DEPLOY_ALPHAZERO_A1_CHECKPOINT is required (local checkpoint file for ${A1_CONFIG})" >&2; exit 1; }
[ -f "${A1_CHECKPOINT}" ] || { echo "Checkpoint not found: ${A1_CHECKPOINT}" >&2; exit 1; }

GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "Deploying alphazero ${GIT_SHA} with ${A1_CONFIG} to ${A1_SSH_TARGET}:${A1_DIR}"

log "Step 1: Preparing remote project directory..."
a1_ssh "install -d -m 0755 '${A1_DIR}/src/models'"

log "Step 2: Syncing engine source (training runs, virtualenv and native builds excluded)..."
# Only what Dockerfile.prod needs. Local native builds would be stale x86_64 objects on
# an aarch64 host, and models/ is excluded so the checkpoint below is the only one there.
a1_rsync \
  --exclude '.venv/' --exclude 'runs/' --exclude 'models/' --exclude 'build/' \
  --exclude 'cpp/build/' --exclude '__pycache__/' --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' --exclude '.ruff_cache/' --exclude 'gomoku/cpp_ext/*.so' \
  "${ALPHAZERO_SRC}/" "${A1_SSH_TARGET}:${A1_DIR}/src/"
a1_scp "${A1_CHECKPOINT}" "${A1_SSH_TARGET}:${A1_DIR}/src/models/champion.pt"
a1_scp "${SCRIPT_DIR}/docker-compose.yml" "${A1_SSH_TARGET}:${A1_DIR}/docker-compose.yml"
a1_ssh "printf 'ALPHAZERO_PORT=%s\nALPHAZERO_CONFIG=%s\nALPHAZERO_INFER_BACKEND=%s\nALPHAZERO_THREADS=%s\nGIT_SHA=%s\n' '${A1_PORT}' '${A1_CONFIG}' '${A1_INFER_BACKEND}' '${A1_THREADS}' '${GIT_SHA}' > '${A1_DIR}/.env'"

COMPOSE="sudo docker compose --project-directory '${A1_DIR}'"
if [ "${DO_BUILD}" = true ]; then
  log "Step 3: Building image on the instance (the first build installs PyTorch and compiles the engine, several minutes on 2 cores)..."
  build_flags=""
  [ "${NO_CACHE}" = true ] && build_flags="--no-cache"
  a1_ssh "${COMPOSE} build ${build_flags} alphazero"
else
  log "Step 3: Skipping build"
fi

log "Step 4: Starting container..."
a1_ssh "${COMPOSE} up -d alphazero && ${COMPOSE} ps"

log "Step 5: Waiting for the engine to answer on the instance itself (it calibrates the int8 model at start)..."
a1_ssh "for i in \$(seq 1 40); do curl -fsS --max-time 3 http://127.0.0.1:${A1_PORT}/ >/dev/null 2>&1 && { echo '  port ${A1_PORT} answers'; exit 0; }; sleep 3; done; echo '  port ${A1_PORT} not answering yet' >&2; sudo docker logs --tail 30 gomoku-alphazero; exit 1"

log "Deploy complete. Measure the search budget with 03_bench_search.sh; switch routing by setting"
log "DEPLOY_ALPHAZERO_IP=${A1_HOST} in .env and running: bash ${REPO_ROOT}/deploy/cloudflare/deploy.sh"
