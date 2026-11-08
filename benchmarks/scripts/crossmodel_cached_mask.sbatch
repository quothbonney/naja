#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=bench-crossmodel-cached-mask
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --deadline=2026-11-08T22:00:00
#SBATCH --output=benchmarks/logs/crossmodel-cached-mask-%j.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
export HOME="$ROOT/home" TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
MODEL="$ROOT/results/crossmodel_extension/cached_models/$1"
srun --cpu-bind=cores "$ROOT/env-cpu/bin/python" "$ROOT/scripts/cached_target_mask_parallel.py" "$MODEL" --output "$MODEL" --workers 4 --lp-time-ms 1000
