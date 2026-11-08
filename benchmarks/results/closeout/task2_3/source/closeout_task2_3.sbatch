#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=bench-closeout-task2-3
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=benchmarks/logs/closeout-task2-3-%j.log
set -euo pipefail
B="$SLURM_SUBMIT_DIR/benchmarks"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache" MPLCONFIGDIR="$B/cache/matplotlib" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --cpu-bind=cores "$B/env-cpu/bin/python" "$B/scripts/closeout_task2_3.py" "$@"
