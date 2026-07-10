#!/usr/bin/env bash
set -euo pipefail

# Build script for naja GPU sampler

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"

# GPU architecture to build for. Default `native` auto-detects the build host's
# GPU (A100->80, H100->90, B300->100/103), so no edits are needed per cluster.
# For a portable fat binary that runs on all of them (CUDA 13):
#   NAJA_CUDA_ARCH='80;90;100;103' ./scripts/build.sh
NAJA_CUDA_ARCH="${NAJA_CUDA_ARCH:-native}"

echo "project     :: ${PROJECT_DIR}"
echo "build       :: ${BUILD_DIR}"
echo "cuda arch   :: ${NAJA_CUDA_ARCH}"
echo

if [[ "${1:-}" == "clean" ]]; then
    echo "> cleaning"
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Prefer Ninja when available (faster incremental CUDA builds); else default make.
GENERATOR=()
if command -v ninja >/dev/null 2>&1; then
    GENERATOR=(-G Ninja)
fi

echo "> cmake configure"
cmake "${PROJECT_DIR}" \
    "${GENERATOR[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${NAJA_CUDA_ARCH}"

echo
echo "> building"
cmake --build . -j"$(nproc)"

echo
echo "executable  :: ${BUILD_DIR}/naja"
