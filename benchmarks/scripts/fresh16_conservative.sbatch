#!/usr/bin/env bash
#SBATCH --partition=all
#SBATCH --job-name=bench-fresh16-conservative
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%1
#SBATCH --time=00:30:00
#SBATCH --deadline=2026-11-08T22:00:00
#SBATCH --output=benchmarks/logs/fresh16-conservative-%A_%a.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
export HOME="$ROOT/home" TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HOPSY_CXXFLAGS='-O3 -march=native -mtune=native'
export HOPSY_BUILD_MANIFEST="$ROOT/audit/hopsy_native_build.json"
srun --cpu-bind=cores "$ROOT/env-cpu-native/bin/python" "$ROOT/scripts/fresh16_conservative.py" --task "$SLURM_ARRAY_TASK_ID"
