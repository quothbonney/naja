#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HOME="$ROOT/home" TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for workers in 16 1; do
  suffix=""
  if [[ "$workers" != 16 ]]; then suffix="_workers$workers"; fi
  for thinning in default jadebeck; do
    for seed in 101 202 303; do
      output="$ROOT/results/hopsy/e_coli_core/$thinning$suffix/seed$seed"
      "$ROOT/env-cpu/bin/python" "$ROOT/cpu_baseline.py" sample \
        --model "$ROOT/data/models/e_coli_core" --out "$output" \
        --seed "$seed" --chains 16 --draws 10000 --warmup 1000 \
        --thinning "$thinning" --workers "$workers" --chunk 11000 --force
      "$ROOT/env-cpu/bin/python" "$ROOT/metrics.py" \
        --samples "$output/samples.npy" --model-dir "$ROOT/data/models/e_coli_core" \
        --output "$output"
    done
  done
done
