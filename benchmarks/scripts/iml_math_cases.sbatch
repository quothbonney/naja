#!/usr/bin/env bash
#SBATCH --partition=all
#SBATCH --job-name=bench-iml-worker-math
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --deadline=2026-11-08T11:00:00
#SBATCH --nice=1000
#SBATCH --output=benchmarks/logs/iml-worker-math-%A_%a.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HOPSY_CXXFLAGS='-O3 -march=native -mtune=native'
export HOPSY_BUILD_MANIFEST="$ROOT/audit/hopsy_native_build.json"
srun --cpu-bind=cores "$ROOT/env-cpu-native/bin/python" "$ROOT/scripts/iml_math_case.py" --task "$SLURM_ARRAY_TASK_ID"
