#!/usr/bin/env bash
set -Eeuo pipefail
# ============================================================
# 03_print_origin.sh
#
# Print the .env values that move the public routing from GCP to Oracle.
# The Cloudflare Worker and its routes stay untouched: 03_deploy_cloudflare.sh
# only rewrites the minimax-api DNS A record and the DocReview origin vars.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/oracle_env_config.sh" >/dev/null

cat <<EOF
# Minimax now served from the Oracle instance (minimax-api.<domain> A record).
DEPLOY_MINIMAX_IP=${ORACLE_HOST}

# DocReview on the same instance (Caddy on port ${ORACLE_DOCREVIEW_PORT}).
# Keep DEPLOY_DOCREVIEW_SITE_ORIGIN as printed by docreview-rag/deploy/oracle/print_origin.sh.
DEPLOY_DOCREVIEW_IP=${ORACLE_HOST}
DEPLOY_DOCREVIEW_ORIGIN=http://${ORACLE_HOST}:${ORACLE_DOCREVIEW_PORT}
EOF

echo
echo "Apply: update the values above in ${DOTENV_PATH}, then run:" >&2
echo "  bash ${REPO_ROOT}/deploy/03_deploy_cloudflare.sh" >&2
echo "DEPLOY_ALPHAZERO_IP stays on GCP." >&2
