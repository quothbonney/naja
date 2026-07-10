#!/usr/bin/env bash
# Build naja inside a CUDA container (NGC PyTorch sqsh, or the image from
# docker/Dockerfile.naja). Installs the minimal toolchain the build needs, then
# invokes scripts/build.sh. Intended to be run *inside* the container, e.g. via
# a Slurm+enroot step:
#
#   srun --container-image=<cuda-image> --container-mounts=$PWD:$PWD \
#        --container-workdir=$PWD --container-remap-root --no-container-entrypoint \
#        bash -c 'NAJA_CUDA_ARCH=native ./scripts/build_container.sh'
#
# Pass-through args go to scripts/build.sh (e.g. `clean`). NAJA_CUDA_ARCH selects
# the GPU target (default `native`; use '80;90;100;103' for a portable binary).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== naja container build ==="
echo "project :: ${PROJECT_DIR}"
echo "arch    :: ${NAJA_CUDA_ARCH:-native}"

echo "> nvcc"
nvcc --version | tail -2 || { echo "ERROR: nvcc not found on PATH (need a CUDA -devel image)"; exit 1; }

# CMake >= 3.30 knows Blackwell sm_100/103 and the `native` keyword; install via
# pip so the version is controlled regardless of the base image's apt cmake.
echo "> installing cmake>=3.30 + ninja (pip)"
python3 -m pip install --quiet --upgrade "cmake>=3.30" ninja \
  || pip install --quiet --upgrade "cmake>=3.30" ninja
hash -r
echo "cmake   :: $(cmake --version | head -1)"
echo "ninja   :: $(ninja --version 2>/dev/null || echo 'n/a')"

# zlib dev headers for find_package(ZLIB) (src/npz.cpp). Only apt-install if the
# header isn't already reachable by the compiler.
if ! echo '#include <zlib.h>' | "${CC:-cc}" -E - >/dev/null 2>&1; then
  echo "> installing zlib1g-dev (apt)"
  (apt-get update && apt-get install -y --no-install-recommends zlib1g-dev) \
    || echo "WARN: could not apt-install zlib1g-dev; relying on preinstalled zlib"
else
  echo "zlib    :: already available"
fi

echo "> building"
cd "${PROJECT_DIR}"
NAJA_CUDA_ARCH="${NAJA_CUDA_ARCH:-native}" ./scripts/build.sh "$@"

echo
echo "=== done -> ${PROJECT_DIR}/build/naja ==="
ls -la "${PROJECT_DIR}/build/naja" 2>/dev/null || { echo "ERROR: build/naja not produced"; exit 1; }
