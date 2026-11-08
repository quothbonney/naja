#!/usr/bin/env bash
#SBATCH --job-name=bench-Recon3D-barrier-probe
#SBATCH --partition=h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --deadline=2026-11-08T11:00:00
#SBATCH --output=benchmarks/logs/recon3d-barrier-probe-%j.log
set -euo pipefail
B="$SLURM_SUBMIT_DIR/benchmarks"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
bash "$B/scripts/enter.sh" python "$B/naja_runner.py" \
  --exe "$B/build-h100-f64/naja" --model "$B/data/models/Recon3D_nullspace_only" \
  --out "$B/results/exploratory_recon3d_barrier/seed101" --seed 101 \
  --chains 16 --draws 200 --warmup 100 --tpb 128 --thinning 0 \
  --barrier whiten --pair-prob 0 --sample-dtype float64 --timeout 1200
"$B/env-cpu/bin/python" "$B/scripts/recon3d_barrier_check.py"
