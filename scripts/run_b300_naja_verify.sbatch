#!/bin/bash
#SBATCH --job-name=naja-verify
#SBATCH --partition=b300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/home/jack/naja/logs/slurm-%j.out
#SBATCH --error=/mnt/home/jack/naja/logs/slurm-%j.err
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
    set -uo pipefail
    echo "=== ldd build/naja (shared-lib resolution) ==="
    ldd ./build/naja | grep -iE "cuda|gurobi|not found" || true
    echo
    echo "=== naja --help ==="
    ./build/naja --help; echo "help rc=$?"
    echo
    echo "=== CLI dry-run test (no GPU sampling, no gurobi license) ==="
    bash tests/test_sample_cli_dry_run.sh && echo "DRYRUN_PASS" || echo "DRYRUN_FAIL rc=$?"
  '
