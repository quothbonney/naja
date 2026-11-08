#!/usr/bin/env bash
#SBATCH --job-name=bench-nsight-l2-recheck
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
set -euo pipefail
B="$PWD"
O="$B/results/nsight_timeline/ncu_l2_recheck"
mkdir -p "$O"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
NCU="$B/tools/nsight/extracted/opt/nvidia/nsight-compute/2025.2.1/ncu"
exec > "$O/job.log" 2>&1
for C in 16 128 512; do
  CMD=("$NCU" --metrics lts__t_sector_hit_rate.pct,lts__t_sectors_lookup_hit.sum,lts__t_sectors_lookup_miss.sum --clock-control none -k regex:.*chrKernel.* -c 1 -o "$O/chains$C" "$B/build-h100-safe/naja" sample run --model-dir "$B/data/models/iJO1366" --out-root "$O/native$C" --gpu 0 --n-chains "$C" --n-samples 1200 --seed 101 --tpb 128 --thinning 0 --constraint-eps 0 --extra-constraint-eps 0 --pair-prob 0 --bounds-policy ignore --write-npy --barrier-rotate)
  printf '%q ' "${CMD[@]}" >> "$O/commands.sh"
  printf '\n' >> "$O/commands.sh"
  "$B/scripts/enter.sh" "${CMD[@]}" > "$O/chains$C.log" 2>&1
  "$NCU" --import "$O/chains$C.ncu-rep" --page raw --csv > "$O/chains$C.csv"
done
