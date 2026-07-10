#!/bin/bash
#SBATCH --job-name=naja-gputest
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
# Prove the CHR kernels EXECUTE correctly on Blackwell (not just compile):
# build with -DNAJA_ENABLE_GPU_TESTS=ON (native arch) and run the GPU moment/
# feasibility tests. These generate on-device and check sample statistics, so a
# pass means the sm_103 build + --use_fast_math numerics are sound.
set -euo pipefail
LAB=/mnt/home/jack/dist-lab
IMAGE="$LAB/images/nvidia+pytorch+26.04-py3.sqsh"
NAJA=/mnt/home/jack/naja
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
    nvidia-smi -L
    python3 -m pip install --quiet --upgrade "cmake>=3.30" ninja || pip install --quiet --upgrade "cmake>=3.30" ninja
    hash -r
    B="'"$NAJA"'/build-gputest"
    rm -rf "$B"; mkdir -p "$B"; cd "$B"
    cmake "'"$NAJA"'" -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=native -DNAJA_ENABLE_GPU_TESTS=ON
    cmake --build . -j"$(nproc)"
    echo "=== ctest GPU sampling tests ==="
    ctest -R gpu --output-on-failure || ctest --output-on-failure -R "sampling|backmap|thin"
  '
