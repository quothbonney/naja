#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=bench-closeout-task1-tests
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --deadline=2026-11-08T22:00:00
#SBATCH --output=benchmarks/logs/closeout-task1-tests-%j.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
OUT="$ROOT/results/closeout/task1"
mkdir -p "$OUT"
export HOME="$ROOT/home" TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
date -u +%FT%TZ > "$OUT/tests-started.txt"
git -C "$ROOT/closeout-release" rev-parse HEAD > "$OUT/tests-commit.txt"
sha256sum "$ROOT/build-closeout/naja" > "$OUT/tests-binary-sha256.txt"
srun --cpu-bind=cores "$ROOT/scripts/enter.sh" "$ROOT/env-gpu/bin/ctest" --test-dir "$ROOT/build-closeout" --output-on-failure -R '^(test_gpu_(sampling_feasibility|sampling_cube_moments|sampling_simplex_moments|backmap_correctness|thin_polytope|index_width)|test_large_sizes_and_npy)$' | tee "$OUT/gpu-correctness-tests.log"
date -u +%FT%TZ > "$OUT/tests-passed.txt"
