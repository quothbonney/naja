#!/usr/bin/env bash
#SBATCH --job-name=bench-coordinate-clock-audit
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --nice=2000
#SBATCH --time=00:12:00
#SBATCH --deadline=2026-11-08T11:00:00
#SBATCH --output=benchmarks/logs/coordinate-clock-%j.log
set -euo pipefail
B="$SLURM_SUBMIT_DIR/benchmarks"
bash "$B/scripts/enter.sh" python "$B/scripts/coordinate_clock_audit.py"
