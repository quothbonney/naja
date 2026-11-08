#!/usr/bin/env bash
#SBATCH --job-name=metrics-mask-v2
#SBATCH --partition=hpc-mid
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
B="$PWD"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
srun --cpu-bind=cores "$B/env-cpu/bin/python" "$B/scripts/recompute_metrics.py" \
  --execute "$1" --index "$SLURM_ARRAY_TASK_ID"
