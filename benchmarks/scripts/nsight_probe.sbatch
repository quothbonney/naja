#!/usr/bin/env bash
#SBATCH --job-name=bench-nsight-probe
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:15:00
set -uo pipefail
B="$PWD"
O="$B/results/nsight_timeline/probe"
mkdir -p "$O"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
NSYS="$B/tools/nsight/extracted/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys"
NCU="$B/tools/nsight/extracted/opt/nvidia/nsight-compute/2025.2.1/ncu"
exec >"$O/probe.log" 2>&1
date -u
env | sort | rg '^SLURM|^CUDA_VISIBLE' > "$O/slurm-environment.txt"
cat /proc/driver/nvidia/params > "$O/nvidia-driver-params.txt"
nvidia-smi -q > "$O/gpu-hardware.txt"
"$NSYS" --version
"$NCU" --version
"$NSYS" status --environment > "$O/nsys-environment.txt" 2>&1
"$NSYS" profile --gpu-metrics-devices=0 --gpu-metrics-set=help > "$O/gpu-metric-sets.txt" 2>&1
printf '%s\n' "$?" > "$O/gpu-metric-sets.exit"
"$NSYS" profile --help > "$O/nsys-profile-help.txt" 2>&1
"$NCU" --help > "$O/ncu-help.txt" 2>&1
CMD=("$B/build-h100-safe/naja" sample run --model-dir "$B/data/models/iJO1366" --out-root "$O/nsys-native" --gpu 0 --n-chains 16 --n-samples 220 --seed 101 --tpb 128 --thinning 0 --constraint-eps 0 --extra-constraint-eps 0 --pair-prob 0 --bounds-policy ignore --write-npy --barrier-rotate)
printf '%q ' "$NSYS" profile -t cuda,nvtx,osrt --gpu-metrics-devices=0 --gpu-metrics-frequency=10000 -o "$O/counter-probe" "${CMD[@]}" > "$O/nsys-command.sh"
printf '\n' >> "$O/nsys-command.sh"
"$B/scripts/enter.sh" "$NSYS" profile -t cuda,nvtx,osrt --gpu-metrics-devices=0 --gpu-metrics-frequency=10000 -o "$O/counter-probe" "${CMD[@]}" > "$O/nsys-counter-probe.log" 2>&1
printf '%s\n' "$?" > "$O/nsys-counter-probe.exit"
CMD[6]="$O/ncu-native"
printf '%q ' "$NCU" --set full -k 'regex:.*chrKernel.*' -c 1 -o "$O/ncu-counter-probe" "${CMD[@]}" > "$O/ncu-command.sh"
printf '\n' >> "$O/ncu-command.sh"
"$B/scripts/enter.sh" "$NCU" --set full -k 'regex:.*chrKernel.*' -c 1 -o "$O/ncu-counter-probe" "${CMD[@]}" > "$O/ncu-counter-probe.log" 2>&1
printf '%s\n' "$?" > "$O/ncu-counter-probe.exit"
date -u
