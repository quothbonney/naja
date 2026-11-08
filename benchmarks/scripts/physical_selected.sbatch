#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=hopsy-physical-selected
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/logs/hopsy-physical-selected-%A_%a.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --cpu-bind=cores "$ROOT/env-cpu-native/bin/python" "$ROOT/scripts/native_physical_pilot.py" --selected-task "$SLURM_ARRAY_TASK_ID"
