#!/usr/bin/env bash
# Runs ON the Oracle A1 instance as root (01_host_setup.sh copies and invokes it).
# Idempotent: prepares Docker, swap, host firewall and deployment directories for
# the minimax engine and docreview-rag. Ubuntu 24.04 aarch64 (Oracle image).
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "run this host setup script as root" >&2
  exit 1
fi

SSH_USER="${1:-ubuntu}"
shift || true
# Remaining arguments: TCP ports to open in the host firewall (default 8080 8880).
PORTS=("$@")
if [ "${#PORTS[@]}" -eq 0 ]; then
  PORTS=(8080 8880)
fi

log() { echo "[$(date +'%F %T')] $*"; }

log "Installing Docker, compose, rsync and Python..."
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl rsync python3 docker.io docker-compose-v2 iptables-persistent
systemctl enable --now docker
if id "${SSH_USER}" >/dev/null 2>&1; then
  usermod -aG docker "${SSH_USER}"
fi

# ---- Swap: 12 GB RAM is plenty, but a swap file protects builds from OOM kills ----
if [[ ! -f /swapfile ]]; then
  log "Creating 2 GB swap file..."
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
else
  log "Swap file exists."
fi

# ---- Host firewall ----
# Oracle's Ubuntu image ships /etc/iptables/rules.v4 with a final REJECT in INPUT.
# Docker-published ports are routed through FORWARD, but an explicit INPUT accept
# keeps host-level listeners (and future non-Docker services) reachable too.
# Source restriction to Cloudflare IPv4 ranges is enforced by the VCN security list.
for port in "${PORTS[@]}"; do
  if iptables -C INPUT -p tcp -m state --state NEW --dport "${port}" -j ACCEPT 2>/dev/null; then
    log "Firewall: port ${port} already open."
  else
    # Insert before the trailing REJECT rule (position 6 mirrors Oracle's default layout;
    # fall back to appending after the established/SSH rules if the chain is shorter).
    if ! iptables -I INPUT 6 -p tcp -m state --state NEW --dport "${port}" -j ACCEPT 2>/dev/null; then
      iptables -I INPUT -p tcp -m state --state NEW --dport "${port}" -j ACCEPT
    fi
    log "Firewall: opened tcp/${port}."
  fi
done
netfilter-persistent save >/dev/null

# ---- Deployment roots ----
install -d -m 0750 /opt/docreview /var/lib/docreview
install -d -m 0755 -o "${SSH_USER}" -g "${SSH_USER}" "/home/${SSH_USER}/build"

log "Host ready: $(uname -m), $(nproc) cores, $(free -h | awk '/Mem:/ {print $2}') RAM, docker $(docker --version | awk '{print $3}' | tr -d ,)"
log "Next: 02_deploy_minimax.sh from the gomoku repo, deploy/oracle/deploy_backend.sh from docreview-rag."
