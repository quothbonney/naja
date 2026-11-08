#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=bench-conditional-ablation
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=benchmarks/logs/conditional-ablation-%A_%a.log
set -euo pipefail
B="$SLURM_SUBMIT_DIR/benchmarks"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache" MPLCONFIGDIR="$B/cache/matplotlib" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --cpu-bind=cores "$B/env-cpu/bin/python" "$B/scripts/conditioned_gpu_ablation.py" --manifest "$1" --index "$SLURM_ARRAY_TASK_ID"
