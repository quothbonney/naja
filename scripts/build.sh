#!/usr/bin/env bash
set -euo pipefail

# Build script for naja GPU sampler

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"

echo "project     :: ${PROJECT_DIR}"
echo "build       :: ${BUILD_DIR}"
echo

if [[ "${1:-}" == "clean" ]]; then
    echo "> cleaning"
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "> cmake configure"
cmake "${PROJECT_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=80

echo
echo "> building"
cmake --build . -j$(nproc)

echo
echo "executable  :: ${BUILD_DIR}/naja"
