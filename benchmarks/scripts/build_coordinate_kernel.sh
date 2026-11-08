#!/usr/bin/env bash
set -euo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"
"$B/scripts/enter.sh" cmake -S "$B/experimental/naja-coordinate" \
  -B "$B/build-h100-coordinate" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DNAJA_ENABLE_GPU_TESTS=ON \
  -DCMAKE_CUDA_FLAGS=--ptxas-options=-v \
  -DNAJA_EIGEN_ROOT="$B/../naja-test-jack/deps/eigen-3.4.0" \
  -DNAJA_GUROBI_ROOT="$B/gurobi1302/linux64"
"$B/scripts/enter.sh" cmake --build "$B/build-h100-coordinate" -j 8
