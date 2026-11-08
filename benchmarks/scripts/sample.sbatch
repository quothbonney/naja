#!/usr/bin/env bash
# Resource limits and array concurrency are supplied by campaign.py.
set -euo pipefail
B="$PWD"
METHOD="$1"
MODEL="$2"
SEEDS=(101 202 303)
TASK="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${SEEDS[$((TASK % 3))]}"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$METHOD" == naja ]]; then
  OUT="$B/results/naja/$MODEL/seed$SEED"
  srun --cpu-bind=cores "$B/scripts/enter.sh" python "$B/naja_runner.py" \
    --exe "$B/build-h100/naja" --model "$B/data/models/$MODEL" --out "$OUT" \
    --seed "$SEED" --timeout 3600 --resume
else
  LABEL=default
  if (( TASK >= 3 )); then LABEL=jadebeck; fi
  LABEL="${3:-$LABEL}"
  OUT="$B/results/hopsy/$MODEL/$LABEL/seed$SEED"
  srun --cpu-bind=cores "$B/env-cpu/bin/python" "$B/cpu_baseline.py" sample \
    --model "$B/data/models/$MODEL" --out "$OUT" --seed "$SEED" \
    --thinning "$LABEL" --workers 16 --chunk 11000
fi
"$B/env-cpu/bin/python" - "$OUT" "$SLURM_JOB_ID" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1])/'slurm.json'
p.write_text(json.dumps({k:v for k,v in os.environ.items() if k.startswith('SLURM_')},indent=2)+'\n')
PY
if [[ "$METHOD" == naja && "${BENCH_SEPARATE_METRICS:-0}" == 1 ]]; then
  exit 0
fi
srun --cpu-bind=cores "$B/env-cpu/bin/python" "$B/metrics.py" \
  --samples "$OUT/samples.npy" --model-dir "$B/data/models/$MODEL" --output "$OUT"
