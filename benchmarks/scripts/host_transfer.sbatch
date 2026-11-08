#!/usr/bin/env bash
#SBATCH --partition=h100
#SBATCH --job-name=host-transfer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=benchmarks/logs/host-transfer-%j.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR/benchmarks"
OUT="$ROOT/results/host_transfer/job$SLURM_JOB_ID"
mkdir -p "$OUT"
date -u +%FT%TZ
hostname
nvidia-smi --query-gpu=name,uuid,pci.bus_id,memory.total,driver_version --format=csv > "$OUT/gpu.csv"
nvidia-smi topo -m > "$OUT/topology.txt"
lscpu > "$OUT/cpu.txt"
ulimit -l > "$OUT/memlock-limit.txt"
sha256sum "$ROOT/tools/host_transfer.cu" "$ROOT/bin/host_transfer" > "$OUT/program.sha256"
"$ROOT/scripts/enter.sh" "$ROOT/bin/host_transfer" > "$OUT/transfers.csv"
date -u +%FT%TZ
