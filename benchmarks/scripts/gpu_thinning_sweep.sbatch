#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=gpu-thinning-sweep
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/logs/gpu-thinning-sweep-%A_%a.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
srun --cpu-bind=cores "$ROOT/env-cpu/bin/python" "$ROOT/scripts/gpu_thinning_sweep.py" --task "$SLURM_ARRAY_TASK_ID" "$@"
