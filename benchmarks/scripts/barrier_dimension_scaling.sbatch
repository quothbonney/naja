#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=bench-barrier-dimension
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --deadline=2026-11-08T22:00:00
#SBATCH --output=benchmarks/logs/barrier-dimension-%j.log
set -euo pipefail
B="$SLURM_SUBMIT_DIR/benchmarks"
srun --cpu-bind=cores "$B/scripts/enter.sh" "$B/env-cpu/bin/python" -u "$B/scripts/barrier_dimension_scaling.py"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache" MPLCONFIGDIR="$B/cache/matplotlib"
"$B/env-plot/bin/python" "$B/scripts/barrier_dimension_scaling.py" --plot
