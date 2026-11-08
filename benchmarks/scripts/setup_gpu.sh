#!/usr/bin/env bash
set -euo pipefail
BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$BENCH_ROOT"/{home,tmp,cache,logs,results}
export HOME="$BENCH_ROOT/home" TMPDIR="$BENCH_ROOT/tmp" XDG_CACHE_HOME="$BENCH_ROOT/cache" MAMBA_ROOT_PREFIX="$BENCH_ROOT/mamba"
if [[ ! -x "$BENCH_ROOT/env-gpu/bin/nvcc" ]]; then
  if [[ -f "$BENCH_ROOT/results/gpu-environment.explicit.txt" ]]; then
    micromamba create --no-rc -y -p "$BENCH_ROOT/env-gpu" --file "$BENCH_ROOT/results/gpu-environment.explicit.txt"
  else
    micromamba create --no-rc -y -p "$BENCH_ROOT/env-gpu" -c conda-forge \
      'cuda-version=12.9' cuda-nvcc cuda-cudart-dev libcurand-dev libcublas-dev libcusolver-dev cmake ninja python numpy
  fi
fi
if [[ ! -f "$BENCH_ROOT/gurobi1302/linux64/lib/libgurobi130.so" ]]; then
  curl -fsSL https://packages.gurobi.com/13.0/gurobi13.0.2_linux64.tar.gz -o "$BENCH_ROOT/cache/gurobi13.0.2_linux64.tar.gz"
  tar -xzf "$BENCH_ROOT/cache/gurobi13.0.2_linux64.tar.gz" -C "$BENCH_ROOT"
fi
micromamba list --no-rc -p "$BENCH_ROOT/env-gpu" --explicit > "$BENCH_ROOT/results/gpu-environment.explicit.txt"
BUILD_DIR="${NAJA_BENCH_BUILD_DIR:-$BENCH_ROOT/build-h100}"
if [[ -x "$BUILD_DIR/naja" && -z "${NAJA_BENCH_BUILD_DIR:-}" ]]; then
  BUILD_DIR="$BENCH_ROOT/build-h100-reproduction"
fi
"$BENCH_ROOT/scripts/enter.sh" cmake -S "$BENCH_ROOT/../naja-test-jack/naja" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 -DNAJA_ENABLE_GPU_TESTS=ON \
  -DNAJA_EIGEN_ROOT="$BENCH_ROOT/../naja-test-jack/deps/eigen-3.4.0" \
  -DNAJA_GUROBI_ROOT="$BENCH_ROOT/gurobi1302/linux64"
"$BENCH_ROOT/scripts/enter.sh" cmake --build "$BUILD_DIR" -j 12
