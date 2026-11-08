#!/usr/bin/env bash
#SBATCH --job-name=bench-coordinate-verify
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --deadline=2026-11-08T11:00:00
#SBATCH --output=benchmarks/logs/coordinate-verify-%j.log
set -euo pipefail
B="$SLURM_SUBMIT_DIR/benchmarks"
bash "$B/scripts/enter.sh" python "$B/scripts/verify_coordinate_kernel.py"
