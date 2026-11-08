#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=bench-closeout-reference-build
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --deadline=2026-11-08T22:00:00
#SBATCH --output=benchmarks/logs/closeout-reference-build-%j.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
export HOME="$ROOT/home" TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDACXX="$ROOT/env-gpu/bin/nvcc"
export PATH="$ROOT/env-gpu/bin:$PATH"
srun --cpu-bind=cores "$ROOT/scripts/enter.sh" "$ROOT/env-gpu/bin/cmake" -S "$ROOT/closeout-baseline-seed101" -B "$ROOT/build-closeout-reference" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 -DEIGEN_DIR="$SLURM_SUBMIT_DIR/naja-test-jack/deps/eigen-3.4.0" -DNAJA_GUROBI_ROOT="$ROOT/gurobi1302/linux64" -DNAJA_ENABLE_GPU_TESTS=OFF
srun --cpu-bind=cores "$ROOT/scripts/enter.sh" "$ROOT/env-gpu/bin/cmake" --build "$ROOT/build-closeout-reference" --target naja -j8
sha256sum "$ROOT/build-closeout-reference/naja" > "$ROOT/results/closeout/task1/reference-binary-sha256.txt"
