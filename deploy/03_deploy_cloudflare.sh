#!/usr/bin/env bash
set -Eeuo pipefail
# ============================================================
# 03_deploy_cloudflare.sh
#
# 1. Updates Cloudflare DNS A records for the backend subdomains
# 2. Deploys the Cloudflare Worker that routes WebSocket traffic
#
# Required .env variables:
#   CLOUDFLARE_ACCOUNT_ID  – Cloudflare account ID
#   CLOUDFLARE_API_TOKEN   – API token (needs Workers + DNS edit)
#   DEPLOY_MINIMAX_IP      – external IP of the minimax VM
#   DEPLOY_ALPHAZERO_IP    – external IP of the alphazero VM
#   DEPLOY_DOMAIN          – root domain (e.g. sungyongcho.com)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/deploy_env_config.sh"

log(){ echo -e "[\e[36m$(date +'%F %T')\e[0m] $*"; }
require_cmd() {
  if ! command -v "$1" &>/dev/null; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

cf_request() {
  local method="$1"
  local url="$2"
  local payload="${3:-}"
  local response

  if [ -n "${payload}" ]; then
    if ! response="$(curl -fsSL --connect-timeout 10 --max-time 30 "${CF_AUTH[@]}" -X "${method}" "${url}" -d "${payload}" 2>&1)"; then
      echo "Cloudflare API request failed (${method} ${url})." >&2
      echo "${response}" >&2
      echo "Hint: check DNS/network egress and CLOUDFLARE_API_TOKEN permissions." >&2
      exit 1
    fi
  else
    if ! response="$(curl -fsSL --connect-timeout 10 --max-time 30 "${CF_AUTH[@]}" -X "${method}" "${url}" 2>&1)"; then
      echo "Cloudflare API request failed (${method} ${url})." >&2
      echo "${response}" >&2
      echo "Hint: check DNS/network egress and CLOUDFLARE_API_TOKEN permissions." >&2
      exit 1
    fi
  fi

  printf '%s' "${response}"
}

cf_success() {
  local response="$1"
  if command -v jq &>/dev/null; then
    [ "$(printf '%s' "${response}" | jq -r '.success // false')" = "true" ]
  else
    printf '%s' "${response}" | grep -Eq '"success"[[:space:]]*:[[:space:]]*true'
  fi
}

cf_first_result_id() {
  local response="$1"
  if command -v jq &>/dev/null; then
    printf '%s' "${response}" | jq -r '.result[0].id // empty'
  else
    printf '%s' "${response}" \
      | grep -Eo '"id"[[:space:]]*:[[:space:]]*"[^"]+"' \
      | head -1 \
      | cut -d'"' -f4
  fi
}

cf_error_summary() {
  local response="$1"
  if command -v jq &>/dev/null; then
    printf '%s' "${response}" | jq -r '[.errors[]? | (.message // ("code=" + (.code|tostring)))] | join("; ")'
  fi
}

ensure_cf_success() {
  local response="$1"
  local action="$2"
  if cf_success "${response}"; then
    return 0
  fi

  local errors
  errors="$(cf_error_summary "${response}")"
  if [ -n "${errors}" ]; then
    echo "Cloudflare API error during ${action}: ${errors}" >&2
  else
    echo "Cloudflare API error during ${action}." >&2
    echo "Response: ${response}" >&2
  fi
  exit 1
}

# ---- Validate ----
: "${CLOUDFLARE_ACCOUNT_ID:?CLOUDFLARE_ACCOUNT_ID is required in .env}"
: "${CLOUDFLARE_API_TOKEN:?CLOUDFLARE_API_TOKEN is required in .env}"
: "${DEPLOY_MINIMAX_IP:?DEPLOY_MINIMAX_IP is required in .env}"
: "${DEPLOY_ALPHAZERO_IP:?DEPLOY_ALPHAZERO_IP is required in .env}"
: "${DEPLOY_DOMAIN:?DEPLOY_DOMAIN is required in .env}"

# ---- Backend subdomains (DNS only, grey cloud) ----
MINIMAX_SUBDOMAIN="minimax-api.${DEPLOY_DOMAIN}"
ALPHAZERO_SUBDOMAIN="alphazero-api.${DEPLOY_DOMAIN}"

MINIMAX_ORIGIN="http://${MINIMAX_SUBDOMAIN}:8080"
ALPHAZERO_ORIGIN="http://${ALPHAZERO_SUBDOMAIN}:8080"
GOMOKU_PROXY_ORIGIN="https://omoku.netlify.app/"

log "Cloudflare deploy"
log "  Account:            ${CLOUDFLARE_ACCOUNT_ID}"
log "  Domain:             ${DEPLOY_DOMAIN}"
log "  Minimax DNS:        ${MINIMAX_SUBDOMAIN} -> ${DEPLOY_MINIMAX_IP}"
log "  AlphaZero DNS:      ${ALPHAZERO_SUBDOMAIN} -> ${DEPLOY_ALPHAZERO_IP}"
log "  Minimax origin:     ${MINIMAX_ORIGIN}"
log "  AlphaZero origin:   ${ALPHAZERO_ORIGIN}"
log "  Gomoku proxy:       /gomoku -> ${GOMOKU_PROXY_ORIGIN}"

# ---- Check tools ----
require_cmd curl
require_cmd npx

# ===========================================================
# Step 1: Update DNS A records via Cloudflare API
# ===========================================================
CF_API="https://api.cloudflare.com/client/v4"
CF_AUTH=(-H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" -H "Content-Type: application/json")

# Get zone ID for the domain
log "Step 1: Fetching zone ID for ${DEPLOY_DOMAIN}..."
ZONE_RESPONSE="$(cf_request GET "${CF_API}/zones?name=${DEPLOY_DOMAIN}&status=active")"
ensure_cf_success "${ZONE_RESPONSE}" "zone lookup for ${DEPLOY_DOMAIN}"
ZONE_ID="$(cf_first_result_id "${ZONE_RESPONSE}")"

if [ -z "${ZONE_ID}" ]; then
  echo "Could not find zone ID for ${DEPLOY_DOMAIN}" >&2
  echo "Response: ${ZONE_RESPONSE}" >&2
  exit 1
fi
log "  Zone ID: ${ZONE_ID}"

# Upsert DNS A record (create or update)
upsert_dns_record() {
  local name="$1"
  local ip="$2"

  # Check if record already exists
  local existing
  existing="$(cf_request GET "${CF_API}/zones/${ZONE_ID}/dns_records?type=A&name=${name}")"
  ensure_cf_success "${existing}" "DNS lookup for ${name}"

  local record_id
  record_id="$(cf_first_result_id "${existing}")"

  local payload="{\"type\":\"A\",\"name\":\"${name}\",\"content\":\"${ip}\",\"ttl\":1,\"proxied\":false}"

  if [ -n "${record_id}" ]; then
    # Update existing record
    local update_response
    update_response="$(cf_request PUT "${CF_API}/zones/${ZONE_ID}/dns_records/${record_id}" "${payload}")"
    ensure_cf_success "${update_response}" "DNS update for ${name}"
    log "  Updated: ${name} -> ${ip}"
  else
    # Create new record
    local create_response
    create_response="$(cf_request POST "${CF_API}/zones/${ZONE_ID}/dns_records" "${payload}")"
    ensure_cf_success "${create_response}" "DNS create for ${name}"
    log "  Created: ${name} -> ${ip}"
  fi
}

log "Step 2: Upserting DNS A records (DNS only / grey cloud)..."
upsert_dns_record "${MINIMAX_SUBDOMAIN}" "${DEPLOY_MINIMAX_IP}"
upsert_dns_record "${ALPHAZERO_SUBDOMAIN}" "${DEPLOY_ALPHAZERO_IP}"

# ===========================================================
# Step 3: Deploy Worker
# ===========================================================
log "Step 3: Deploying Cloudflare Worker..."
export CLOUDFLARE_API_TOKEN
export CLOUDFLARE_ACCOUNT_ID

npx wrangler deploy \
  --config "${SCRIPT_DIR}/wrangler.toml" \
  --var "MINIMAX_ORIGIN:${MINIMAX_ORIGIN}" \
  --var "ALPHAZERO_ORIGIN:${ALPHAZERO_ORIGIN}"

log "Cloudflare Worker deployed successfully ✅"
log "Running connection verification..."
bash "${SCRIPT_DIR}/verify_connection.sh"
