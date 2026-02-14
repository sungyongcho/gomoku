#!/usr/bin/env bash
set -Eeuo pipefail
# ============================================================
# verify_connection.sh
#
# Tests connectivity to the deployed Gomoku backends
# through the Cloudflare Worker reverse proxy.
# ============================================================

log(){ echo -e "[\e[36m$(date +'%F %T')\e[0m] $*"; }

DOMAIN="${DEPLOY_DOMAIN:-sungyongcho.com}"
GOMOKU_PUBLIC_PATH="https://${DOMAIN}/gomoku"

log "Verifying endpoints on ${DOMAIN}..."

# ---- Health check (HTTP) ----
log " -> Health check: https://${DOMAIN}/alphazero/"
HEALTH_RESPONSE="$(curl -fsSL --max-time 10 "https://${DOMAIN}/alphazero/" 2>&1 || true)"
echo "  Response: ${HEALTH_RESPONSE}"
if echo "${HEALTH_RESPONSE}" | grep -q '"status":"ok"'; then
  log "  AlphaZero health check ✅"
else
  log "  AlphaZero health check ❌ (may need a moment to propagate)"
fi

# ---- Gomoku proxy check ----
log " -> Gomoku proxy check: ${GOMOKU_PUBLIC_PATH}"
GOMOKU_HTTP_CODE="$(curl -sS -o /dev/null --max-time 10 -w '%{http_code}' "${GOMOKU_PUBLIC_PATH}" || true)"
GOMOKU_EFFECTIVE_URL="$(curl -sSL -o /dev/null --max-time 10 -w '%{url_effective}' "${GOMOKU_PUBLIC_PATH}" || true)"
echo "  HTTP: ${GOMOKU_HTTP_CODE:-<none>}"
echo "  Effective URL: ${GOMOKU_EFFECTIVE_URL:-<none>}"
if [[ "${GOMOKU_HTTP_CODE}" == 2* || "${GOMOKU_HTTP_CODE}" == 3* ]] && [[ "${GOMOKU_EFFECTIVE_URL}" == "${GOMOKU_PUBLIC_PATH}"* ]]; then
  log "  /gomoku proxy check ✅"
else
  log "  /gomoku proxy check ❌ (URL should stay on ${GOMOKU_PUBLIC_PATH})"
fi

# ---- WebSocket tests ----
log " -> WebSocket test: wss://${DOMAIN}/minimax/ws"
npx -y wscat -c "wss://${DOMAIN}/minimax/ws" --wait 3 -x '{"type":"ping"}' 2>&1 | head -5 || true

log " -> WebSocket test: wss://${DOMAIN}/alphazero/ws"
npx -y wscat -c "wss://${DOMAIN}/alphazero/ws" --wait 3 -x '{"type":"ping"}' 2>&1 | head -5 || true

log "Done 🎉"
