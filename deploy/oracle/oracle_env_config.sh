#!/usr/bin/env bash
# Shared configuration for deploy/oracle. Source it; do not execute it.
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
if [ "${ORACLE_HOST_OPTIONAL:-0}" = 1 ]; then
  export ORACLE_HOST="${DEPLOY_ORACLE_HOST:-0.0.0.0}"
else
  export ORACLE_HOST="${DEPLOY_ORACLE_HOST:?DEPLOY_ORACLE_HOST is required (public IPv4 of the A1 instance)}"
fi

# ===== Optional =====
export ORACLE_SSH_USER="${DEPLOY_ORACLE_SSH_USER:-ubuntu}"
export ORACLE_SSH_KEY="${DEPLOY_ORACLE_SSH_KEY:-}"
export ORACLE_MINIMAX_PORT="${DEPLOY_ORACLE_MINIMAX_PORT:-8080}"
# Port reserved for docreview-rag's Caddy on the same host; opened by host_setup.sh.
export ORACLE_DOCREVIEW_PORT="${DEPLOY_ORACLE_DOCREVIEW_PORT:-8880}"
# Remote directories. The compose project lives in the SSH user's home so rsync
# needs no root; containers are managed with sudo docker.
export ORACLE_MINIMAX_DIR="${DEPLOY_ORACLE_MINIMAX_DIR:-/home/${ORACLE_SSH_USER}/build/gomoku-minimax}"

if [[ ! "${ORACLE_HOST}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
  echo "DEPLOY_ORACLE_HOST must be a plain IPv4 address, got: ${ORACLE_HOST}" >&2
  exit 1
fi

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30)
if [ -n "${ORACLE_SSH_KEY}" ]; then
  SSH_OPTS+=(-i "${ORACLE_SSH_KEY}")
fi
export ORACLE_SSH_TARGET="${ORACLE_SSH_USER}@${ORACLE_HOST}"

oracle_ssh() { ssh "${SSH_OPTS[@]}" "${ORACLE_SSH_TARGET}" "$@"; }
oracle_scp() { scp "${SSH_OPTS[@]}" "$@"; }
oracle_rsync() { rsync -az --delete -e "ssh ${SSH_OPTS[*]}" "$@"; }

log() { echo -e "[\e[36m$(date +'%F %T')\e[0m] $*"; }

echo "Loaded Oracle configuration:"
echo "  Host: ${ORACLE_SSH_TARGET}$( [ -n "${ORACLE_SSH_KEY}" ] && echo " (key ${ORACLE_SSH_KEY})")"
echo "  Ports: minimax=${ORACLE_MINIMAX_PORT}, docreview=${ORACLE_DOCREVIEW_PORT}"
echo "  Minimax dir: ${ORACLE_MINIMAX_DIR}"
echo "-------------------------------------"
