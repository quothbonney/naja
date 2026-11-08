#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=bench-crossmodel-tpb-pair
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --deadline=2026-11-08T22:00:00
#SBATCH --output=benchmarks/logs/crossmodel-tpb-pair-%j.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
export HOME="$ROOT/home" TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --cpu-bind=cores "$ROOT/scripts/enter.sh" "$ROOT/env-cpu/bin/python" "$ROOT/scripts/crossmodel_tpb_pair.py"
