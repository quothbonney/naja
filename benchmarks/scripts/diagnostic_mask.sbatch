#!/usr/bin/env bash
#SBATCH --job-name=diagnostic-mask
#SBATCH --partition=hpc-mid
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -euo pipefail
B="$PWD"
MODELS=(e_coli_core iAB_RBC_283 iJR904 iAF1260 iJO1366 iML1515 iMM904 RECON1 Recon3D)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
srun --cpu-bind=cores "$B/env-cpu/bin/python" "$B/diagnostic_mask.py" "$B/data/models/$MODEL"
