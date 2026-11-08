#!/usr/bin/env bash
#SBATCH --partition=hpc-high
#SBATCH --job-name=bench-cpu512-physical64
#SBATCH --cpus-per-task=128
#SBATCH --mem=384G
#SBATCH --time=00:30:00
#SBATCH --deadline=2026-11-08T11:00:00
#SBATCH --output=benchmarks/logs/cpu512-physical64-%j.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ARGS=(--method cpu --seed "${1:-101001}" --draws "${2:-10000}")
if [[ -n "${3:-}" ]]; then ARGS+=(--out "$3"); fi
srun --cpu-bind=cores "$ROOT/env-cpu-native/bin/python" "$ROOT/scripts/high_throughput_sample.py" "${ARGS[@]}"
