#!/usr/bin/env bash
#SBATCH --partition=hpc-mid
#SBATCH --job-name=bench-closeout-wide-output
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=00:02:00
#SBATCH --deadline=2026-11-08T22:00:00
#SBATCH --output=benchmarks/logs/closeout-wide-output-%j.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
OUT="$ROOT/results/closeout/task1"
export HOME="$ROOT/home" TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --cpu-bind=cores "$ROOT/scripts/enter.sh" "$ROOT/env-gpu/bin/nvcc" -std=c++17 -arch=sm_90 --expt-relaxed-constexpr -Xcudafe --diag_suppress=20012 -I"$ROOT/closeout-release/src" -I"$ROOT/closeout-release/src/gpu" -I"$SLURM_SUBMIT_DIR/naja-test-jack/deps/eigen-3.4.0" "$OUT/wide_output_smoke.cu" "$ROOT/build-closeout-release/CMakeFiles/naja.dir/src/gpu/dmatrix.cu.o" "$ROOT/build-closeout-release/CMakeFiles/naja.dir/src/gpu/dvector.cu.o" "$ROOT/build-closeout-release/CMakeFiles/naja.dir/src/gpu/backmap.cu.o" -L"$ROOT/env-gpu/lib" -L"$ROOT/env-gpu/targets/x86_64-linux/lib" -lcublas -o "$OUT/wide_output_smoke"
sha256sum "$OUT/wide_output_smoke.cu" "$ROOT/build-closeout-release/CMakeFiles/naja.dir/src/gpu/dmatrix.cu.o" "$OUT/wide_output_smoke" > "$OUT/wide-output-sources.sha256"
set +e
srun --cpu-bind=cores "$ROOT/scripts/enter.sh" "$OUT/wide_output_smoke" "$OUT/wide_output_result.json"
CODE=$?
set -e
if [[ "$CODE" != 0 && "$CODE" != 77 ]]; then exit "$CODE"; fi
