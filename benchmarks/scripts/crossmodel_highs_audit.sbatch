#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=bench-crossmodel-highs-audit
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --deadline=2026-11-08T22:00:00
#SBATCH --output=benchmarks/logs/crossmodel-highs-audit-%j.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
export HOME="$ROOT/home" TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
MODEL="$ROOT/results/crossmodel_extension/cached_models/iAF1260"
srun --cpu-bind=cores "$ROOT/env-cpu/bin/python" "$ROOT/results/crossmodel_extension/source_snapshots/cached_mask_highs_audit.py" --model "$MODEL" --parent "$MODEL/diagnostic_mask.json" --output "$ROOT/results/crossmodel_extension/highs_audit_iAF1260" --limit 20
