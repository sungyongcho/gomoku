#!/usr/bin/env bash
set -Eeuo pipefail

YES=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [--yes]

Purges LOCAL Google Cloud login/config state:
  - gcloud accounts, tokens, configs, and ADC
  - legacy boto/gsutil auth files
  - Docker gcloud credential helpers for Artifact Registry/GCR
  - default GCE SSH keys created by gcloud

This does NOT delete remote GCP resources and does NOT uninstall gcloud.

Options:
  --yes       Skip interactive confirmation
  -h, --help  Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes) YES=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

log(){ echo -e "[\e[36m$(date +'%F %T')\e[0m] $*"; }

confirm_or_exit() {
  [[ "${YES}" == true ]] && return 0
  read -r -p "Purge all LOCAL gcloud account/config state? [y/N]: " ans
  case "${ans}" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 1 ;;
  esac
}

confirm_or_exit

if command -v gcloud &>/dev/null; then
  log "Revoking gcloud auth and ADC tokens..."
  gcloud auth revoke --all --quiet >/dev/null 2>&1 || true
  gcloud auth application-default revoke --quiet >/dev/null 2>&1 || true
fi

log "Removing gcloud, boto, and gsutil local config..."
rm -rf "${HOME}/.config/gcloud" "${HOME}/.boto" "${HOME}/.gsutil"
if [[ -n "${CLOUDSDK_CONFIG:-}" && "${CLOUDSDK_CONFIG}" != "${HOME}/.config/gcloud" ]]; then
  rm -rf "${CLOUDSDK_CONFIG}"
fi

DOCKER_CONFIG_FILE="${HOME}/.docker/config.json"
if [[ -f "${DOCKER_CONFIG_FILE}" ]]; then
  if command -v jq &>/dev/null; then
    log "Removing Docker gcloud credential helpers..."
    tmp_file="$(mktemp)"
    jq '
      def is_gcloud_registry:
        (.value == "gcloud") and
        ((.key | endswith("docker.pkg.dev")) or
         (.key | endswith(".gcr.io")) or
         (.key == "gcr.io") or
         (.key == "marketplace.gcr.io"));
      if (.credHelpers | type) == "object" then
        .credHelpers |= with_entries(select(is_gcloud_registry | not))
      else . end
      | if (.credHelpers? == {}) then del(.credHelpers) else . end
    ' "${DOCKER_CONFIG_FILE}" > "${tmp_file}"
    mv "${tmp_file}" "${DOCKER_CONFIG_FILE}"
  else
    log "Skipping Docker config cleanup: jq is not installed."
  fi
fi

log "Removing default gcloud/GCP SSH keys..."
rm -f \
  "${HOME}/.ssh/google_compute_engine" \
  "${HOME}/.ssh/google_compute_engine.pub" \
  "${HOME}/.ssh/id_rsa_gcp" \
  "${HOME}/.ssh/id_rsa_gcp.pub"

for var_name in GOOGLE_APPLICATION_CREDENTIALS CLOUDSDK_CONFIG CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE; do
  if [[ -n "${!var_name:-}" ]]; then
    log "Note: ${var_name} is still set in this shell. Remove it from your shell profile if needed."
  fi
done

log "Done. Local gcloud account/config state has been purged."
