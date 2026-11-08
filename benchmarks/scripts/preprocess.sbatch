#!/usr/bin/env bash
#SBATCH --job-name=naja-polyround
#SBATCH --partition=hpc-mid
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=09:00:00
#SBATCH --array=0-6%3
set -euo pipefail
B="$PWD"
MODELS=(iJR904 iJO1366 iMM904 iAF1260 iML1515 RECON1 Recon3D)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$B/data/models/$MODEL"
printf '%s model=%s job=%s host=%s\n' "$(date -u +%FT%TZ)" "$MODEL" "$SLURM_JOB_ID" "$(hostname)"
srun --cpu-bind=cores "$B/env-cpu/bin/python" "$B/cpu_baseline.py" prep \
  --model "$B/data/raw/$MODEL.json" --out "$B/data/models/$MODEL"
