#!/bin/bash
#SBATCH --job-name=naja-wfix
#SBATCH --partition=hpc-mid
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/home/jack/naja/logs/slurm-%j.out
#SBATCH --error=/mnt/home/jack/naja/logs/slurm-%j.err
# Build into a SIDE dir (build_fix) so a concurrent job using build/naja is not
# disturbed, then verify: --flat-output implies WRITE_DATA=true (no --write-npy).
set -euo pipefail
LAB=/mnt/home/jack/dist-lab; IMAGE="$LAB/images/nvidia+pytorch+26.04-py3.sqsh"
NAJA=/mnt/home/jack/naja
export ENROOT_DATA_PATH=/tmp/enroot-$USER-$SLURM_JOB_ID/data
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER-$SLURM_JOB_ID/runtime
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH"
srun --container-image="$IMAGE" --container-mounts="$NAJA:$NAJA" --container-workdir="$NAJA" \
  --no-container-mount-home --container-remap-root --no-container-entrypoint bash -c '
    set -euo pipefail
    python3 -m pip install --quiet --upgrade "cmake>=3.30" ninja || pip install --quiet "cmake>=3.30" ninja
    hash -r
    B='"$NAJA"'/build_fix; rm -rf "$B"; mkdir -p "$B"; cd "$B"
    cmake '"$NAJA"' -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native >/tmp/cfg.log 2>&1
    cmake --build . -j"$(nproc)" >/tmp/bld.log 2>&1 && echo "BUILD_OK" || { echo "BUILD_FAIL"; tail -20 /tmp/bld.log; exit 1; }
    export LD_LIBRARY_PATH='"$NAJA"'/extern/gurobi/linux64/lib:${LD_LIBRARY_PATH:-}
    # tiny valid model
    T=$(mktemp -d); M="$T/models/MODEL_X"; mkdir -p "$M/rounding"
    printf "1,0\n0,1\n" > "$M/rounding/MODEL_X_rounding_A.csv"
    printf "0\n0\n"      > "$M/rounding/MODEL_X_rounding_b.csv"
    printf "0\n0\n"      > "$M/rounding/MODEL_X_rounding_start.csv"
    printf "MODEL_X\n" > "$T/list.txt"
    echo "=== WITH --flat-output, NO --write-npy (expect WRITE_DATA=true) ==="
    "$B/naja" sample bulk --models-root "$T/models" --model-list "$T/list.txt" \
      --out-root "$T/o1" --name t --gpus 0 --n-chains 1 --n-samples 1 \
      --flat-output "$T/o1/npy" --print-config --dry-run 2>&1 | grep -iE "WRITE_DATA|write_data" || echo "(WRITE_DATA not in config dump)"
    echo "=== NO --flat-output, NO --write-npy (expect WRITE_DATA=false, regression) ==="
    "$B/naja" sample bulk --models-root "$T/models" --model-list "$T/list.txt" \
      --out-root "$T/o2" --name t --gpus 0 --n-chains 1 --n-samples 1 \
      --print-config --dry-run 2>&1 | grep -iE "WRITE_DATA|write_data" || echo "(WRITE_DATA not in config dump)"
  '
