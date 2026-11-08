#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=bench-worker-math16
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --deadline=2026-11-08T11:00:00
#SBATCH --output=benchmarks/logs/worker-math16-%A_%a.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --cpu-bind=cores "$ROOT/env-cpu-native/bin/python" "$ROOT/scripts/math_worker_case.py" --task "$SLURM_ARRAY_TASK_ID"
