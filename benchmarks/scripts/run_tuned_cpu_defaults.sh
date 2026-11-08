#!/usr/bin/env bash
# Run inside a Slurm allocation; 16 independent chains executed by one worker.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HOME="$ROOT/home" TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON="$ROOT/env-cpu/bin/python"
if [[ $# -eq 0 ]]; then
  set -- iAB_RBC_283 iJR904 iMM904 iJO1366 iAF1260 iML1515
fi
for model in "$@"; do
  if [[ ! -f "$ROOT/data/models/$model/rounding/polytope.npz" ]]; then
    printf 'SKIP missing preprocessed model: %s\n' "$model"
    continue
  fi
  for seed in 101 202 303; do
    output="$ROOT/results/hopsy/$model/default_workers1/seed$seed"
    complete=0
    if [[ -f "$output/run.json" ]]; then
      complete="$($PYTHON -c 'import json,sys; d=json.load(open(sys.argv[1])); print(int(d.get("status")=="complete" and "process_wall_seconds" in d))' "$output/run.json")"
    fi
    if [[ "$complete" != 1 ]]; then
      "$PYTHON" "$ROOT/cpu_baseline.py" sample \
        --model "$ROOT/data/models/$model" --out "$output" \
        --seed "$seed" --chains 16 --draws 10000 --warmup 1000 \
        --thinning default --workers 1 --chunk 11000 --force
    fi
    if [[ ! -f "$output/metrics.json" ]]; then
      "$PYTHON" "$ROOT/metrics.py" --samples "$output/samples.npy" \
        --model-dir "$ROOT/data/models/$model" --output "$output"
    fi
  done
done
