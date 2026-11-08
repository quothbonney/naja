#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=hopsy-tuned-default
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=benchmarks/logs/hopsy-tuned-default-%j.log
set -euo pipefail
# Submit from the project root; optionally pass model IDs as script arguments.
exec /usr/bin/bash "$SLURM_SUBMIT_DIR/benchmarks/scripts/run_tuned_cpu_defaults.sh" "$@"
