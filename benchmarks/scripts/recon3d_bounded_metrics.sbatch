#!/usr/bin/env bash
#SBATCH --partition=all
#SBATCH --job-name=metrics-closeout-Recon3D
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=00:40:00
#SBATCH --deadline=2026-11-08T22:00:00
#SBATCH --output=benchmarks/logs/closeout-Recon3D-metrics-%j.log
set -euo pipefail
B="$SLURM_SUBMIT_DIR/benchmarks"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
srun --cpu-bind=cores "$B/env-cpu/bin/python" "$B/scripts/recon3d_bounded.py" --stage metrics
