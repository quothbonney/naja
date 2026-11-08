#!/usr/bin/env bash
#SBATCH --job-name=bench-Recon3D-nullspace
#SBATCH --partition=hpc-mid
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --deadline=2026-11-08T10:30:00
#SBATCH --output=benchmarks/logs/recon3d-nullspace-%j.log
set -euo pipefail
B="$SLURM_SUBMIT_DIR/benchmarks"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
"$B/env-cpu/bin/python" "$B/scripts/recon3d_nullspace_probe.py"
"$B/env-cpu/bin/python" "$B/scripts/validate_preprocessed.py" \
  --model "$B/data/models/Recon3D_nullspace_only" --raw "$B/data/raw/Recon3D.json"
