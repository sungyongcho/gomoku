#!/usr/bin/env bash
# Configuration for deploy/minimax. Source it; do not execute it.
#
# The Oracle Cloud host is a single Always Free Ampere A1 instance that serves
# the minimax engine (this repo) and docreview-rag (its own deploy/oracle).
# Values come from the repo .env (or DOTENV_PATH); see .env.example.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export SCRIPT_DIR REPO_ROOT

DOTENV_PATH="${DOTENV_PATH:-${REPO_ROOT}/.env}"
if [ ! -f "${DOTENV_PATH}" ]; then
  echo "env file not found. Set DOTENV_PATH or create ${REPO_ROOT}/.env" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${DOTENV_PATH}"
set +a

# ===== Required (00_security_list.sh runs before the instance has an IP) =====
if [ "${MINIMAX_HOST_OPTIONAL:-0}" = 1 ]; then
  export MINIMAX_HOST="${DEPLOY_MINIMAX_HOST:-0.0.0.0}"
else
  export MINIMAX_HOST="${DEPLOY_MINIMAX_HOST:?DEPLOY_MINIMAX_HOST is required (public IPv4 of the A1 instance)}"
fi

# ===== Optional =====
export MINIMAX_SSH_USER="${DEPLOY_MINIMAX_SSH_USER:-ubuntu}"
export MINIMAX_SSH_KEY="${DEPLOY_MINIMAX_SSH_KEY:-}"
export MINIMAX_PORT="${DEPLOY_MINIMAX_PORT:-8080}"
# Port reserved for docreview-rag's Caddy on the same host; opened by host_setup.sh.
export DOCREVIEW_PORT="${DEPLOY_MINIMAX_DOCREVIEW_PORT:-8880}"
# Remote directories. The compose project lives in the SSH user's home so rsync
# needs no root; containers are managed with sudo docker.
export MINIMAX_DIR="${DEPLOY_MINIMAX_DIR:-/home/${MINIMAX_SSH_USER}/build/gomoku-minimax}"

if [[ ! "${MINIMAX_HOST}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
  echo "DEPLOY_MINIMAX_HOST must be a plain IPv4 address, got: ${MINIMAX_HOST}" >&2
  exit 1
fi

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30)
if [ -n "${MINIMAX_SSH_KEY}" ]; then
  SSH_OPTS+=(-i "${MINIMAX_SSH_KEY}")
fi
export MINIMAX_SSH_TARGET="${MINIMAX_SSH_USER}@${MINIMAX_HOST}"

minimax_ssh() { ssh "${SSH_OPTS[@]}" "${MINIMAX_SSH_TARGET}" "$@"; }
minimax_scp() { scp "${SSH_OPTS[@]}" "$@"; }
minimax_rsync() { rsync -az --delete -e "ssh ${SSH_OPTS[*]}" "$@"; }

log() { echo -e "[\e[36m$(date +'%F %T')\e[0m] $*"; }

echo "Loaded Oracle configuration:"
echo "  Host: ${MINIMAX_SSH_TARGET}$( [ -n "${MINIMAX_SSH_KEY}" ] && echo " (key ${MINIMAX_SSH_KEY})")"
echo "  Ports: minimax=${MINIMAX_PORT}, docreview=${DOCREVIEW_PORT}"
echo "  Minimax dir: ${MINIMAX_DIR}"
echo "-------------------------------------"
