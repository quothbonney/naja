#!/usr/bin/env bash
#SBATCH --job-name=bench-deadline-cleanup
#SBATCH --partition=hpc-high
#SBATCH --begin=2026-11-08T11:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:02:00
#SBATCH --output=benchmarks/logs/deadline-cleanup-%j.log
set -euo pipefail
B="$SLURM_SUBMIT_DIR/benchmarks"
exec "$B/env-cpu/bin/python" "$B/scripts/slurm_deadline_cleanup.py"
