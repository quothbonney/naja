#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=Recon3D-validation-publish
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/logs/recon3d-validation-publish-%j.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
srun --cpu-bind=cores "$ROOT/env-cpu/bin/python" "$ROOT/scripts/recon3d_completion_worker.py" "$1"
