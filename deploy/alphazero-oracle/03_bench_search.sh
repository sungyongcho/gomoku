#!/usr/bin/env bash
set -Eeuo pipefail
# ============================================================
# 03_bench_search.sh
#
# Print how many searches the deployed image completes within the move budget on the
# instance, with the same backend, thread count, config and checkpoint the engine serves.
# The number decides which model is worth serving: the search stops on time, so a larger
# network answers just as fast but with fewer searches.
#
# It runs in a throwaway container next to the engine. A game in progress shares the two
# cores with it, so stop the engine first (STOP_ENGINE=true) for a clean number.
#
# Flags:
#   STOP_ENGINE=true   stop the engine for the measurement and start it again after
#   POSITIONS=12       positions to search
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/alphazero_oracle_env_config.sh"

STOP_ENGINE="${STOP_ENGINE:-false}"
POSITIONS="${POSITIONS:-12}"
COMPOSE="sudo docker compose --project-directory '${A1_DIR}'"

if [ "${STOP_ENGINE}" = true ]; then
  log "Stopping the engine for the measurement..."
  a1_ssh "${COMPOSE} stop alphazero"
fi

log "Searching ${POSITIONS} positions with ${A1_CONFIG} (${A1_INFER_BACKEND}, ${A1_THREADS} threads)..."
status=0
a1_ssh "sudo docker run --rm \
  -e ALPHAZERO_INFER_BACKEND='${A1_INFER_BACKEND}' -e ALPHAZERO_INFER_THREADS='${A1_THREADS}' \
  -e OMP_NUM_THREADS='${A1_THREADS}' -e TORCH_NUM_THREADS='${A1_THREADS}' \
  gomoku-alphazero:local python -m gomoku.scripts.bench_search \
  --config '${A1_CONFIG}' --checkpoint models/champion.pt --positions '${POSITIONS}'" || status=$?

if [ "${STOP_ENGINE}" = true ]; then
  log "Starting the engine again..."
  a1_ssh "${COMPOSE} up -d alphazero"
fi
exit "${status}"
