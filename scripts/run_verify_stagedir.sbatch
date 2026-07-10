#!/bin/bash
#SBATCH --job-name=naja-vstage
#SBATCH --partition=hpc-mid
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/home/jack/naja/logs/slurm-%j.out
#SBATCH --error=/mnt/home/jack/naja/logs/slurm-%j.err
set -euo pipefail
LAB=/mnt/home/jack/dist-lab; IMAGE="$LAB/images/nvidia+pytorch+26.04-py3.sqsh"
NAJA=/mnt/home/jack/naja
export ENROOT_DATA_PATH=/tmp/enroot-$USER-$SLURM_JOB_ID/data
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER-$SLURM_JOB_ID/runtime
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH"
srun --container-image="$IMAGE" --container-mounts="$NAJA:$NAJA" --container-workdir="$NAJA" \
  --no-container-mount-home --container-remap-root --no-container-entrypoint bash -c '
    set -uo pipefail
    export LD_LIBRARY_PATH="'"$NAJA"'/extern/gurobi/linux64/lib:${LD_LIBRARY_PATH:-}"
    echo "=== --stage-dir advertised / accepted? ==="
    ./build/naja sample bulk --help 2>&1 | grep -i "stage-dir" || echo "(not in help text — checking parse)"
    echo "=== error when --stage-dir without --flat-output ==="
    ./build/naja sample bulk --models-root /tmp/x --model-list /tmp/x.txt --name t --gpus 0 \
      --n-chains 1 --n-samples 1 --stage-dir /tmp/stg 2>&1 | grep -i "requires --flat-output" \
      && echo "GUARD_OK" || echo "GUARD_MISSING"
    echo "=== staged dry-run (build a tiny model, sample bulk --dry-run --stage-dir) ==="
    T=$(mktemp -d); M="$T/models/MODEL_X"; mkdir -p "$M/rounding"
    printf "1,0\n0,1\n" > "$M/rounding/MODEL_X_rounding_A.csv"
    printf "0\n0\n"      > "$M/rounding/MODEL_X_rounding_b.csv"
    printf "0\n0\n"      > "$M/rounding/MODEL_X_rounding_start.csv"
    printf "MODEL_X\n" > "$T/list.txt"; mkdir -p "$T/out" "$T/stage"
    ./build/naja sample bulk --models-root "$T/models" --model-list "$T/list.txt" \
      --out-root "$T/out" --name smoke --gpus 0 --n-chains 1 --n-samples 1 \
      --flat-output "$T/out/npy" --stage-dir "$T/stage" --dry-run 2>&1 | tail -5 \
      && echo "DRYRUN_STAGE_OK" || echo "DRYRUN_STAGE_FAIL"
  '
