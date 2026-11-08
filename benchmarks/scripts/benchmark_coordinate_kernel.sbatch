#!/usr/bin/env bash
#SBATCH --job-name=bench-coordinate-paired
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:25:00
#SBATCH --deadline=2026-11-08T11:00:00
#SBATCH --output=benchmarks/logs/coordinate-paired-%j.log
set -euo pipefail
B="$SLURM_SUBMIT_DIR/benchmarks"
bash "$B/scripts/enter.sh" python "$B/scripts/benchmark_coordinate_kernel.py"
