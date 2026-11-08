#!/usr/bin/env bash
set -euo pipefail
BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HOME="$BENCH_ROOT/home"
export TMPDIR="$BENCH_ROOT/tmp"
export XDG_CACHE_HOME="$BENCH_ROOT/cache"
export MAMBA_ROOT_PREFIX="$BENCH_ROOT/mamba"
export UV_CACHE_DIR="$BENCH_ROOT/cache/uv"
export PYTHONDONTWRITEBYTECODE=1
export PATH="$BENCH_ROOT/env-gpu/bin:$PATH"
export CUDACXX="$BENCH_ROOT/env-gpu/bin/nvcc"
export LD_LIBRARY_PATH="$BENCH_ROOT/env-gpu/lib:$BENCH_ROOT/env-gpu/targets/x86_64-linux/lib:$BENCH_ROOT/gurobi1302/linux64/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${BENCH_BLAS_THREADS:-1}"
export MKL_NUM_THREADS="${BENCH_BLAS_THREADS:-1}"
ulimit -c 0
exec "$@"
