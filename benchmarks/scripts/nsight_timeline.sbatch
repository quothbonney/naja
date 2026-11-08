#!/usr/bin/env bash
#SBATCH --job-name=bench-nsight-timeline
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:20:00
set -euo pipefail
B="$PWD"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$B/results/nsight_timeline"
env | sort | rg '^SLURM|^CUDA_VISIBLE' > "$B/results/nsight_timeline/slurm-environment.txt"
srun --cpu-bind=cores "$B/scripts/enter.sh" "$B/env-cpu/bin/python" "$B/scripts/nsight_timeline_runs.py"
