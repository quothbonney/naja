#!/usr/bin/env bash
#SBATCH --partition=all
#SBATCH --job-name=metrics-512chain
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --deadline=2026-11-08T11:00:00
#SBATCH --output=benchmarks/logs/512chain-common-metrics-%j.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
srun --cpu-bind=cores "$ROOT/env-cpu/bin/python" "$ROOT/scripts/high_throughput_metrics.py" --run "$1"
