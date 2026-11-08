#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HOME="$ROOT/home" TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKERS="${1:-16}"
SUFFIX=""
if [[ "$WORKERS" != 16 ]]; then SUFFIX="_workers$WORKERS"; fi
for thinning in default jadebeck; do
  for seed in 101 202 303; do
    output="$ROOT/results/hopsy/e_coli_core/$thinning$SUFFIX/seed$seed"
    "$ROOT/env-cpu/bin/python" "$ROOT/cpu_baseline.py" sample \
      --model "$ROOT/data/models/e_coli_core" --out "$output" \
      --seed "$seed" --chains 16 --draws 10000 --warmup 1000 \
      --thinning "$thinning" --workers "$WORKERS" --chunk 11000
    "$ROOT/env-cpu/bin/python" "$ROOT/metrics.py" \
      --samples "$output/samples.npy" --model-dir "$ROOT/data/models/e_coli_core" \
      --output "$output"
  done
done
