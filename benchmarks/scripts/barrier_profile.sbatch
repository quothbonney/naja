#!/usr/bin/env bash
#SBATCH --job-name=bench-barrier-profile
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --deadline=2026-11-08T11:00:00
set -euo pipefail
B="$PWD"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --cpu-bind=cores "$B/scripts/enter.sh" "$B/env-cpu/bin/python" "$B/scripts/barrier_profile_runs.py"
