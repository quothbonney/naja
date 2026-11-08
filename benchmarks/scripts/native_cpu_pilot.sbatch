#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=hopsy-native-pilot
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/logs/hopsy-native-pilot-%j.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
VARIANT="${1:-native}"
export HOME="$ROOT/home" TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$VARIANT" == native ]]; then
  PYTHON="$ROOT/env-cpu-native/bin/python"
  LABEL=2000_w16
  export HOPSY_CXXFLAGS='-O3 -march=native -mtune=native'
  export HOPSY_BUILD_MANIFEST="$ROOT/audit/hopsy_native_build.json"
  "$PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="passed"' \
    "$ROOT/audit/hopsy_native_cube.json"
elif [[ "$VARIANT" == source_generic ]]; then
  PYTHON="$ROOT/env-cpu-sdist/bin/python"
  LABEL=2000_w16_sourcegeneric
  export HOPSY_CXXFLAGS='-O3'
  export HOPSY_BUILD_MANIFEST="$ROOT/audit/hopsy_source_generic_build.json"
else
  PYTHON="$ROOT/env-cpu/bin/python"
  LABEL=2000_w16_wheelcontrol
  export HOPSY_CXXFLAGS='wheel defaults'
fi
OUTPUT="$ROOT/results/native_cpu/iJO1366/$LABEL/seed101"
"$PYTHON" "$ROOT/cpu_baseline.py" sample --model "$ROOT/data/models/iJO1366" \
  --out "$OUTPUT" --seed 101 --chains 16 --draws 10000 --warmup 1000 \
  --thinning 2000 --workers 16 --chunk 11000 --build-variant "$VARIANT"
"$PYTHON" "$ROOT/metrics.py" --samples "$OUTPUT/samples.npy" \
  --model-dir "$ROOT/data/models/iJO1366" --output "$OUTPUT"
