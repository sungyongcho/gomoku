#!/usr/bin/env bash
set -Eeuo pipefail
# ============================================================
# 02_deploy_minimax.sh
#
# Sync the minimax engine source to the Oracle A1 instance, build the image
# there (native aarch64, no cross-compilation or registry) and (re)start the
# container. Re-run for every engine update; the build cache keeps it fast.
#
# Flags:
#   DO_BUILD=false   skip the image build (restart only)
#   NO_CACHE=true    force a clean image build
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/oracle_env_config.sh"

DO_BUILD="${DO_BUILD:-true}"
NO_CACHE="${NO_CACHE:-false}"
MINIMAX_SRC="${REPO_ROOT}/minimax"

command -v rsync >/dev/null 2>&1 || { echo "Missing required command: rsync" >&2; exit 1; }
[ -f "${MINIMAX_SRC}/Dockerfile.prod" ] || { echo "minimax prod Dockerfile not found: ${MINIMAX_SRC}/Dockerfile.prod" >&2; exit 1; }

GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "Deploying minimax ${GIT_SHA} to ${ORACLE_SSH_TARGET}:${ORACLE_MINIMAX_DIR}"

log "Step 1: Preparing remote project directory..."
oracle_ssh "install -d -m 0755 '${ORACLE_MINIMAX_DIR}/src'"

log "Step 2: Syncing engine source (build outputs and binaries excluded)..."
# Only what Dockerfile.prod needs: sources, headers, Makefile. Local build
# artefacts would be stale x86_64 objects on an aarch64 host.
oracle_rsync \
  --exclude 'build/' --exclude 'build_debug/' --exclude '.pytest_cache/' \
  --exclude 'minimax' --exclude 'minimax_debug' --exclude 'doublethree_test' \
  --exclude 'bit_test' --exclude 'search_benchmark' --exclude '*.o' \
  "${MINIMAX_SRC}/" "${ORACLE_SSH_TARGET}:${ORACLE_MINIMAX_DIR}/src/"
oracle_scp "${SCRIPT_DIR}/docker-compose.yml" "${ORACLE_SSH_TARGET}:${ORACLE_MINIMAX_DIR}/docker-compose.yml"
oracle_ssh "printf 'MINIMAX_PORT=%s\nGIT_SHA=%s\n' '${ORACLE_MINIMAX_PORT}' '${GIT_SHA}' > '${ORACLE_MINIMAX_DIR}/.env'"

COMPOSE="sudo docker compose --project-directory '${ORACLE_MINIMAX_DIR}'"
if [ "${DO_BUILD}" = true ]; then
  log "Step 3: Building image on the instance (first build compiles with g++, ~2-4 min on 2 cores)..."
  build_flags=""
  [ "${NO_CACHE}" = true ] && build_flags="--no-cache"
  oracle_ssh "${COMPOSE} build ${build_flags} minimax"
else
  log "Step 3: Skipping build"
fi

log "Step 4: Starting container..."
oracle_ssh "${COMPOSE} up -d minimax && ${COMPOSE} ps"

log "Step 5: Verifying the WebSocket port from the instance itself..."
oracle_ssh "for i in 1 2 3 4 5 6; do (exec 3<>/dev/tcp/127.0.0.1/${ORACLE_MINIMAX_PORT}) 2>/dev/null && { echo '  port ${ORACLE_MINIMAX_PORT} accepts connections'; exit 0; }; sleep 2; done; echo '  port ${ORACLE_MINIMAX_PORT} not reachable yet' >&2; sudo docker logs --tail 20 gomoku-minimax; exit 1"

log "Deploy complete. Cut over with: bash ${SCRIPT_DIR}/03_print_origin.sh"
