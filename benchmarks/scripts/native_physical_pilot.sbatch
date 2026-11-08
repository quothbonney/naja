#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=hopsy-native-physical16
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/logs/hopsy-native-physical16-%j.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HOPSY_CXXFLAGS='-O3 -march=native -mtune=native'
export HOPSY_BUILD_MANIFEST="$ROOT/audit/hopsy_native_build.json"
"$ROOT/env-cpu-native/bin/python" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="passed"' "$ROOT/audit/hopsy_native_cube.json"
srun --cpu-bind=cores "$ROOT/env-cpu-native/bin/python" "$ROOT/scripts/native_physical_pilot.py" --seed "${SLURM_ARRAY_TASK_ID:-101}"
