#!/usr/bin/env bash
#SBATCH --partition=all
#SBATCH --job-name=metrics-rescue-controls
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-12%4
#SBATCH --time=00:30:00
#SBATCH --deadline=2026-11-08T22:00:00
#SBATCH --output=benchmarks/logs/rescue-metrics-%A_%a.log
set -euo pipefail
B="$SLURM_SUBMIT_DIR/benchmarks"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
srun --cpu-bind=cores "$B/env-cpu/bin/python" "$B/scripts/conditioned_rescue_controls.py" --stage metrics --index "$SLURM_ARRAY_TASK_ID"
