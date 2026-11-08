#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=bench-cached-mask4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/logs/bench-cached-mask4-%j.log
set -euo pipefail
B="$SLURM_SUBMIT_DIR/benchmarks"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --cpu-bind=cores "$B/env-cpu/bin/python" "$B/scripts/cached_target_mask_parallel.py" "$1" --output "$2" --workers 4 --lp-time-ms 1000
