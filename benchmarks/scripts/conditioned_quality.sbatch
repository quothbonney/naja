#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=quality-conditioned
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/logs/quality-conditioned-%j.log
set -euo pipefail
B="$SLURM_SUBMIT_DIR/benchmarks"
export TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --cpu-bind=cores "$B/env-cpu/bin/python" "$B/scripts/conditioned_quality.py" "$@"
