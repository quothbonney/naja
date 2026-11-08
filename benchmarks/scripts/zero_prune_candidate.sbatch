#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=zero-prune-prep
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=benchmarks/logs/zero-prune-%j.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
MODE="$1";MODEL="${2:-Recon3D}";VARIANT="${3:-default}"
export HOME="$ROOT/home" TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
CACHE="$ROOT/data/preprocessing/${MODEL}_zero_fva"
PYTHON="$ROOT/env-cpu/bin/python"
if [[ "$MODE" == fva ]]; then
  export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  exec "$PYTHON" "$ROOT/scripts/zero_prune_fva.py" --model "$ROOT/data/raw/$MODEL.json" \
    --out "$CACHE" --workers "$SLURM_CPUS_PER_TASK"
fi
FLAGS=()
if [[ "$VARIANT" == presolve ]]; then FLAGS=(--presolve); fi
if [[ "$VARIANT" == faces ]]; then FLAGS=(--fva-inactive-faces); fi
if [[ "$VARIANT" == faces_fast ]]; then FLAGS=(--fva-inactive-faces --fast-sparse-assembly); fi
exec "$PYTHON" "$ROOT/cpu_baseline.py" prep --model "$ROOT/data/raw/$MODEL.json" \
  --out "$ROOT/data/models/${MODEL}_zero_fva_${VARIANT}" --zero-prune-fva "$CACHE" \
  --verbose "${FLAGS[@]}"
