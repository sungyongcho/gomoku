#!/usr/bin/env bash
# Configuration for deploy/alphazero-oracle. Source it; do not execute it.
#
# The host is an Always Free Ampere A1 instance (2 OCPU, 12 GB, aarch64) that runs only
# the AlphaZero engine. Values come from the repo .env (or DOTENV_PATH); see .env.example.
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

# ===== Required =====
export A1_HOST="${DEPLOY_ALPHAZERO_A1_HOST:?DEPLOY_ALPHAZERO_A1_HOST is required (public IPv4 of the A1 instance)}"

# ===== Optional =====
export A1_SSH_USER="${DEPLOY_ALPHAZERO_A1_SSH_USER:-ubuntu}"
export A1_SSH_KEY="${DEPLOY_ALPHAZERO_A1_SSH_KEY:-}"
export A1_PORT="${DEPLOY_ALPHAZERO_A1_PORT:-8080}"
# The model served: a YAML under alphazero/ naming its size and search settings, and the
# checkpoint that goes with it (a local file).
export A1_CONFIG="${DEPLOY_ALPHAZERO_A1_CONFIG:-configs/deploy-v5.yaml}"
export A1_CHECKPOINT="${DEPLOY_ALPHAZERO_A1_CHECKPOINT:-}"
# Static int8 with both cores is what makes the search fit the move budget on this CPU:
# on two Neoverse-N1 cores a 6x96 network completes about 630 searches in 2.5 seconds
# this way, 390 on one core and 200 as plain float32.
export A1_INFER_BACKEND="${DEPLOY_ALPHAZERO_A1_INFER_BACKEND:-onnx-static}"
export A1_THREADS="${DEPLOY_ALPHAZERO_A1_THREADS:-2}"
# Remote directories. The compose project lives in the SSH user's home so rsync needs no
# root; containers are managed with sudo docker.
export A1_DIR="${DEPLOY_ALPHAZERO_A1_DIR:-/home/${A1_SSH_USER}/build/gomoku-alphazero}"

if [[ ! "${A1_HOST}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
  echo "DEPLOY_ALPHAZERO_A1_HOST must be a plain IPv4 address, got: ${A1_HOST}" >&2
  exit 1
fi

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o IdentitiesOnly=yes)
if [ -n "${A1_SSH_KEY}" ]; then
  SSH_OPTS+=(-i "${A1_SSH_KEY}")
fi
export A1_SSH_TARGET="${A1_SSH_USER}@${A1_HOST}"

a1_ssh() { ssh "${SSH_OPTS[@]}" "${A1_SSH_TARGET}" "$@"; }
a1_scp() { scp "${SSH_OPTS[@]}" "$@"; }
a1_rsync() { rsync -az --delete -e "ssh ${SSH_OPTS[*]}" "$@"; }

log() { echo -e "[\e[36m$(date +'%F %T')\e[0m] $*"; }

echo "Loaded Oracle configuration:"
echo "  Host: ${A1_SSH_TARGET}$( [ -n "${A1_SSH_KEY}" ] && echo " (key ${A1_SSH_KEY})")"
echo "  Port: ${A1_PORT}"
echo "  Model: ${A1_CONFIG} (${A1_INFER_BACKEND}, ${A1_THREADS} threads)"
echo "  AlphaZero dir: ${A1_DIR}"
echo "-------------------------------------"
