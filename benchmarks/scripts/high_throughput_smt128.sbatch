#!/usr/bin/env bash
#SBATCH --partition=hpc-high
#SBATCH --job-name=bench-cpu512-logical128
#SBATCH --cpus-per-task=128
#SBATCH --mem=768G
#SBATCH --time=00:15:00
#SBATCH --deadline=2026-11-08T22:00:00
#SBATCH --output=benchmarks/logs/cpu512-logical128-%j.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
export HOME="$ROOT/home" TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HOPSY_CXXFLAGS='-O3 -march=native -mtune=native'
export HOPSY_BUILD_MANIFEST="$ROOT/audit/hopsy_native_build.json"
srun --cpu-bind=cores "$ROOT/env-cpu-native/bin/python" "$ROOT/scripts/high_throughput_smt128.py" --seed "${1:-101001}"
