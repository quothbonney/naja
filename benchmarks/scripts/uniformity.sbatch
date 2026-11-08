#!/usr/bin/env bash
set -euo pipefail
B="$PWD"
METHOD="$1"
mkdir -p "$B/results/uniformity"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$METHOD" == hopsy ]]; then
  for seed in 1201 2401; do
    OUT="$B/results/uniformity/hopsy$seed"
    if [[ ! -f "$OUT/run.json" ]] || ! "$B/env-cpu/bin/python" -c 'import json,sys;sys.exit(json.load(open(sys.argv[1])).get("status")!="complete")' "$OUT/run.json"; then
      srun --cpu-bind=cores "$B/env-cpu/bin/python" "$B/cpu_baseline.py" sample \
        --model "$B/data/models/e_coli_core" --out "$OUT" --seed "$seed" \
        --chains 16 --draws 62500 --warmup 1000 --thinning jadebeck --workers "${UNIFORMITY_WORKERS:-1}" --chunk 63500
    fi
  done
else
  for mode in rotate whiten; do
    PROB=0
    if [[ "$mode" == whiten ]]; then PROB=0.5; fi
    srun --cpu-bind=cores /usr/bin/env bash "$B/scripts/enter.sh" python "$B/naja_runner.py" \
      --exe "$B/build-h100/naja" --model "$B/data/models/e_coli_core" \
      --out "$B/results/uniformity/naja_$mode" --seed 3601 \
      --chains 16 --draws 62500 --warmup 1000 --barrier "$mode" --pair-prob "$PROB" \
      --timeout 900 --resume
  done
fi
