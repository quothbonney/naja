#!/usr/bin/env bash
#SBATCH --job-name=larger-model-quality
#SBATCH --partition=all
#SBATCH --mem=64G
#SBATCH --time=00:30:00
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
export HOME="$ROOT/home" TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --cpu-bind=cores "$ROOT/env-cpu-native/bin/python" "$ROOT/scripts/larger_model_quality.py" --method "$1" --task "$SLURM_ARRAY_TASK_ID"
