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

# ---- WebSocket tests ----
log " -> WebSocket test: wss://${DOMAIN}/minimax/ws"
npx -y wscat -c "wss://${DOMAIN}/minimax/ws" --wait 3 -x '{"type":"ping"}' 2>&1 | head -5 || true

log " -> WebSocket test: wss://${DOMAIN}/alphazero/ws"
npx -y wscat -c "wss://${DOMAIN}/alphazero/ws" --wait 3 -x '{"type":"ping"}' 2>&1 | head -5 || true

log "Done 🎉"
