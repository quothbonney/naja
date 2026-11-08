#!/usr/bin/env bash
#SBATCH --job-name=uniformity-analysis
#SBATCH --partition=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -euo pipefail
B="$PWD"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
srun --cpu-bind=cores /usr/bin/env "$B/env-cpu/bin/python" "$B/scripts/compare_uniformity.py"
