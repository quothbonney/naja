#!/usr/bin/env bash
#SBATCH --partition=all
#SBATCH --job-name=metrics-512cached
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --deadline=2026-11-08T22:00:00
#SBATCH --output=benchmarks/logs/512cached-metrics-%j.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
export HOME="$ROOT/home" TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --cpu-bind=cores "$ROOT/env-cpu/bin/python" "$ROOT/metrics.py" \
  --samples "$1" --model-dir "$2" --output "$3" --prefixes "$4" --workers 4
