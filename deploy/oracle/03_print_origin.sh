#!/usr/bin/env bash
set -Eeuo pipefail
# ============================================================
# 03_print_origin.sh
#
# Print the .env values that move the public routing from GCP to Oracle.
# 03_deploy_cloudflare.sh updates Gomoku routing only. DocReview uses its own
# Worker and deployment commands in the docreview-rag repository.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/oracle_env_config.sh" >/dev/null

cat <<EOF
# Minimax now served from the Oracle instance (minimax-api.<domain> A record).
DEPLOY_MINIMAX_IP=${ORACLE_HOST}
EOF

echo
echo "Apply: update the values above in ${DOTENV_PATH}, then run:" >&2
echo "  bash ${REPO_ROOT}/deploy/03_deploy_cloudflare.sh" >&2
echo "DEPLOY_ALPHAZERO_IP stays on GCP." >&2
