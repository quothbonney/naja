#!/usr/bin/env bash
set -euo pipefail
B="$PWD"
MODEL="$1"; VARIANT="$2"; SEED="$3"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
if [[ "$MODEL" == e_coli_core ]]; then
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
fi
srun --cpu-bind=cores /usr/bin/env "$B/env-cpu/bin/python" "$B/scripts/quality.py" --model "$MODEL" --variant "$VARIANT" --seed "$SEED" "${@:4}"
