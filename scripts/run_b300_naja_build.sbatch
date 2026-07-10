#!/bin/bash
#SBATCH --job-name=naja-build
#SBATCH --partition=b300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/home/jack/naja/logs/slurm-%j.out
#SBATCH --error=/mnt/home/jack/naja/logs/slurm-%j.err
#
# Build naja inside the on-disk NGC PyTorch container (CUDA 13) on a B300 node.
# `native` arch detection needs a visible GPU at configure time -> gpu:1.
#   sbatch scripts/run_b300_naja_build.sbatch
set -euo pipefail

LAB=/mnt/home/jack/dist-lab
IMAGE="$LAB/images/nvidia+pytorch+26.04-py3.sqsh"
NAJA=/mnt/home/jack/naja
mkdir -p "$NAJA/logs"

export ENROOT_DATA_PATH=/tmp/enroot-$USER-$SLURM_JOB_ID/data
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER-$SLURM_JOB_ID/runtime
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH"

srun \
  --container-image="$IMAGE" \
  --container-mounts="$NAJA:$NAJA" \
  --container-workdir="$NAJA" \
  --no-container-mount-home --container-remap-root --no-container-entrypoint \
  bash -c '
    set -euo pipefail
    export NAJA_CUDA_ARCH="${NAJA_CUDA_ARCH:-native}"
    echo "=== host GPU ==="; nvidia-smi -L || true
    ./scripts/build_container.sh clean
  '
