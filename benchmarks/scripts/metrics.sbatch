#!/usr/bin/env bash
# CPU-only diagnostics, so GPU allocations finish immediately after sampling.
set -euo pipefail
B="$PWD"
MODEL="$1"
SEEDS=(101 202 303)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
OUT="$B/results/naja/$MODEL/seed$SEED"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --cpu-bind=cores "$B/env-cpu/bin/python" "$B/metrics.py" \
  --samples "$OUT/samples.npy" --model-dir "$B/data/models/$MODEL" --output "$OUT"
