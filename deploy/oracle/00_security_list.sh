#!/usr/bin/env bash
set -Eeuo pipefail
# ============================================================
# 00_security_list.sh
#
# Add the VCN security list ingress rules for the Oracle A1 origin with the OCI
# CLI instead of typing 15 Cloudflare CIDRs into the console:
#   - tcp/22 from DEPLOY_ORACLE_SSH_SOURCE_CIDR (default: your current public IP /32)
#   - tcp/<minimax port> and tcp/<docreview port> from every Cloudflare IPv4 range
# Existing rules are preserved; missing ones are appended. Idempotent.
#
# Requires: oci CLI configured (oci setup config), jq, and in .env:
#   DEPLOY_ORACLE_SECURITY_LIST_OCID=ocid1.securitylist.oc1...
# Find it: Networking > Virtual cloud networks > <vcn> > Security lists.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Only the security list OCID is needed here; DEPLOY_ORACLE_HOST may not exist yet.
export ORACLE_HOST_OPTIONAL=1
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/oracle_env_config.sh"

for cmd in oci jq curl python3; do
  command -v "${cmd}" >/dev/null 2>&1 || { echo "Missing required command: ${cmd}" >&2; exit 1; }
done
SECURITY_LIST_OCID="${DEPLOY_ORACLE_SECURITY_LIST_OCID:?DEPLOY_ORACLE_SECURITY_LIST_OCID is required in .env}"

SSH_SOURCE_CIDR="${DEPLOY_ORACLE_SSH_SOURCE_CIDR:-}"
if [ -z "${SSH_SOURCE_CIDR}" ]; then
  my_ip="$(curl -fsSL --max-time 10 https://api.ipify.org)"
  [[ "${my_ip}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || { echo "Could not detect the current public IPv4." >&2; exit 1; }
  SSH_SOURCE_CIDR="${my_ip}/32"
fi

log "Fetching Cloudflare IPv4 ranges..."
CF_RANGES="$(curl -fsSL --max-time 15 https://www.cloudflare.com/ips-v4)"
[ -n "${CF_RANGES}" ] || { echo "Failed to fetch Cloudflare IPv4 ranges." >&2; exit 1; }

log "Reading current security list ${SECURITY_LIST_OCID}..."
current="$(oci network security-list get --security-list-id "${SECURITY_LIST_OCID}" --query 'data."ingress-security-rules"' --raw-output)"

merged="$(SSH_SOURCE_CIDR="${SSH_SOURCE_CIDR}" CF_RANGES="${CF_RANGES}" \
  MINIMAX_PORT="${ORACLE_MINIMAX_PORT}" DOCREVIEW_PORT="${ORACLE_DOCREVIEW_PORT}" \
  python3 -c '
import json, os, sys
rules = json.loads(sys.stdin.read() or "[]")

def rule(cidr, port, desc):
    return {"protocol": "6", "source": cidr, "source-type": "CIDR_BLOCK", "is-stateless": False,
            "description": desc, "tcp-options": {"destination-port-range": {"min": port, "max": port}}}

def key(r):
    tcp = r.get("tcp-options") or {}
    rng = tcp.get("destination-port-range") or {}
    return (r.get("protocol"), r.get("source"), rng.get("min"), rng.get("max"))

wanted = [rule(os.environ["SSH_SOURCE_CIDR"], 22, "SSH from operator")]
for cidr in os.environ["CF_RANGES"].split():
    for name in ("MINIMAX_PORT", "DOCREVIEW_PORT"):
        wanted.append(rule(cidr, int(os.environ[name]), f"Cloudflare origin {name.split(chr(95))[0].lower()}"))
existing = {key(r) for r in rules}
added = [r for r in wanted if key(r) not in existing]
rules.extend(added)
print(json.dumps({"rules": rules, "added": len(added)}))
' <<< "${current}")"

added="$(jq -r '.added' <<< "${merged}")"
if [ "${added}" = "0" ]; then
  log "Security list already contains every rule. Nothing to do."
  exit 0
fi

rules_file="$(mktemp)"
trap 'rm -f "${rules_file}"' EXIT
jq '.rules' <<< "${merged}" > "${rules_file}"
log "Adding ${added} ingress rules (SSH from ${SSH_SOURCE_CIDR}; ${ORACLE_MINIMAX_PORT}/${ORACLE_DOCREVIEW_PORT} from Cloudflare)..."
oci network security-list update \
  --security-list-id "${SECURITY_LIST_OCID}" \
  --ingress-security-rules "file://${rules_file}" \
  --force \
  --query 'data."ingress-security-rules" | length(@)' --raw-output \
  | sed 's/^/  ingress rules now: /'
log "Done. Egress rules were not touched."
