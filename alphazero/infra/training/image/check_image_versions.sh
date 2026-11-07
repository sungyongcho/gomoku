#!/usr/bin/env bash
# Fail when the GPU head image and the CPU worker image would ship different versions.
#
# Builds only the `versions` stage of each Dockerfile (it shares the build cache with the full
# images) and compares the files image_versions.py wrote: Python patch version, Ray, torch
# release and every shared Python package. Ray rejects a worker whose Python patch version
# differs from the head's, so run this before pushing either image.
#
# Usage: check_image_versions.sh [output_dir]
set -Eeuo pipefail

IMAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALPHAZERO_ROOT="$(cd "${IMAGE_DIR}/../../.." && pwd)"
OUT_DIR="${1:-$(mktemp -d)}"
GPU_DOCKERFILE="${GPU_DOCKERFILE:-${IMAGE_DIR}/Dockerfile.py313-cu134}"
CPU_DOCKERFILE="${CPU_DOCKERFILE:-${IMAGE_DIR}/Dockerfile.py313-cpu}"

build_versions() {
  local dockerfile="$1" dest="$2"
  docker buildx build \
    --platform linux/amd64 \
    --target versions \
    --output "type=local,dest=${dest}" \
    -f "${dockerfile}" \
    "${ALPHAZERO_ROOT}"
}

build_versions "${GPU_DOCKERFILE}" "${OUT_DIR}/gpu"
build_versions "${CPU_DOCKERFILE}" "${OUT_DIR}/cpu"

echo "GPU image: $(head -n 3 "${OUT_DIR}/gpu/image-versions.txt" | paste -sd ' ')"
echo "CPU image: $(head -n 3 "${OUT_DIR}/cpu/image-versions.txt" | paste -sd ' ')"

if ! diff -u --label gpu --label cpu \
  "${OUT_DIR}/gpu/image-versions.txt" "${OUT_DIR}/cpu/image-versions.txt"; then
  echo "ERROR: the GPU and CPU images disagree on the versions above." >&2
  exit 1
fi
echo "Images agree on $(wc -l < "${OUT_DIR}/cpu/image-versions.txt") versions (${OUT_DIR})."
