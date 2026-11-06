#!/usr/bin/env bash
set -Eeuo pipefail
# ============================================================
# 01_host_setup.sh
#
# One-time preparation of the Oracle A1 instance: Docker, swap, host firewall
# ports and deployment directories. Safe to re-run.
# Prerequisite: the instance exists, SSH works as DEPLOY_MINIMAX_SSH_USER, and
# the VCN security list already admits SSH from your IP.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/minimax_env_config.sh"

log "Step 1: Checking SSH access to ${MINIMAX_SSH_TARGET}..."
minimax_ssh 'uname -m; nproc; free -h | sed -n 2p'

log "Step 2: Uploading host_setup.sh..."
remote_stage="$(minimax_ssh 'mktemp -d /tmp/gomoku-oracle.XXXXXXXX')"
minimax_scp "${SCRIPT_DIR}/host_setup.sh" "${MINIMAX_SSH_TARGET}:${remote_stage}/host_setup.sh"

log "Step 3: Running host setup as root (ports ${MINIMAX_PORT}, ${DOCREVIEW_PORT})..."
minimax_ssh -t "sudo bash '${remote_stage}/host_setup.sh' '${MINIMAX_SSH_USER}' '${MINIMAX_PORT}' '${DOCREVIEW_PORT}'; rm -rf '${remote_stage}'"

log "Host setup complete."
log "Reminder: the VCN security list must allow tcp/${MINIMAX_PORT} and tcp/${DOCREVIEW_PORT} from Cloudflare IPv4 ranges only:"
log "  curl -fsSL https://www.cloudflare.com/ips-v4"
