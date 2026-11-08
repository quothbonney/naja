#!/usr/bin/env bash
set -euo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"
"$B/scripts/enter.sh" cmake -S "$B/experimental/naja-barrier" \
  -B "$B/build-h100-barrier-profile" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DNAJA_ENABLE_GPU_TESTS=ON \
  -DNAJA_NVTX_INCLUDE="$B/tools/nsight/extracted/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nvtx/include" \
  -DNAJA_EIGEN_ROOT="$B/../naja-test-jack/deps/eigen-3.4.0" \
  -DNAJA_GUROBI_ROOT="$B/gurobi1302/linux64"
"$B/scripts/enter.sh" cmake --build "$B/build-h100-barrier-profile" -j 8
