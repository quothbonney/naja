#!/usr/bin/env bash
#SBATCH --partition=hpc-high
#SBATCH --job-name=bench-closeout-Recon3D
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=192G
#SBATCH --time=00:59:00
#SBATCH --deadline=2026-11-08T22:00:00
#SBATCH --output=benchmarks/logs/closeout-Recon3D-capture-%j.log
set -euo pipefail
B="$SLURM_SUBMIT_DIR/benchmarks"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
srun --cpu-bind=cores "$B/env-cpu/bin/python" "$B/scripts/recon3d_bounded.py" --stage capture
