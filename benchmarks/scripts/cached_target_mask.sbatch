#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=bench-cached-mask
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/logs/bench-cached-mask-%j.log
set -euo pipefail
B="$SLURM_SUBMIT_DIR/benchmarks"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --cpu-bind=cores "$B/env-cpu/bin/python" "$B/scripts/cached_target_mask.py" "$1" --output "$2"
