#!/usr/bin/env bash
#SBATCH --job-name=naja-correctness
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -euo pipefail
B="$PWD"
mkdir -p "$B/results/setup"
{
  date -u +%FT%TZ
  hostname
  nvidia-smi
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,power.limit --format=csv
  lscpu
  free -h
  "$B/env-gpu/bin/nvcc" --version
  "$B/env-gpu/bin/cmake" --version
  git -C "$B/../naja-test-jack/naja" rev-parse HEAD
  git -C "$B/../naja-test-jack/naja" diff --stat
  "$B/scripts/enter.sh" "$B/build-h100/naja" --help || true
  command -v matlab || true
} > "$B/results/setup/h100-environment.txt" 2>&1
git -C "$B/../naja-test-jack/naja" diff > "$B/results/setup/source.patch"
sha256sum "$B/build-h100/naja" > "$B/results/setup/binary.sha256"
"$B/scripts/enter.sh" ctest --test-dir "$B/build-h100" -V -R '^test_gpu_' | tee "$B/results/setup/gpu-correctness.log"
