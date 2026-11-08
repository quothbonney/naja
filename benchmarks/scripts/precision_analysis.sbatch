#!/usr/bin/env bash
#SBATCH --job-name=naja-precision-analysis
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -euo pipefail
B="$PWD"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
srun --cpu-bind=cores /usr/bin/env "$B/env-cpu/bin/python" "$B/scripts/compare_precision.py"
