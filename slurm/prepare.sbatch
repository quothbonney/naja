#!/bin/bash
#SBATCH --job-name=naja-prepare
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --time=12:00:00
# CPU-only Phase-B array (conditioning-inheritance + feasible-start LPs). Runs on
# a CPU partition so it never occupies GPU nodes. --partition/--array/--output
# and --cpus-per-task supplied by naja-slurm. Outputs are small (extra_A/extra_b,
# bounds, start) -> no NVMe staging needed. Required env:
#   MODELS_ROOT SHARD_DIR IMAGE NAJA_REPO NAJA_BIN ; optional PREPARE_FLAGS GRB_LICENSE_FILE
set -euo pipefail
: "${MODELS_ROOT:?}"; : "${SHARD_DIR:?}"; : "${IMAGE:?}"; : "${NAJA_REPO:?}"; : "${NAJA_BIN:?}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
SHARD="${SHARD_DIR}/shard_${TASK}.txt"
[[ -f "$SHARD" ]] || { echo "shard not found: $SHARD" >&2; exit 1; }
export ENROOT_DATA_PATH="/tmp/enroot-${USER}-${SLURM_JOB_ID}/data"
export ENROOT_RUNTIME_PATH="/tmp/enroot-${USER}-${SLURM_JOB_ID}/runtime"
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH"
MOUNTS="${NAJA_REPO}:${NAJA_REPO},${MODELS_ROOT}:${MODELS_ROOT}"
[[ -n "${GRB_LICENSE_FILE:-}" ]] && MOUNTS="${MOUNTS},$(dirname "$GRB_LICENSE_FILE"):$(dirname "$GRB_LICENSE_FILE")"

srun --container-image="$IMAGE" --container-mounts="$MOUNTS" \
     --container-workdir="$NAJA_REPO" \
     --no-container-mount-home --container-remap-root --no-container-entrypoint \
     bash -c '
       set -euo pipefail
       export LD_LIBRARY_PATH="'"$NAJA_REPO"'/extern/gurobi/linux64/lib:${LD_LIBRARY_PATH:-}"
       [[ -n "'"${GRB_LICENSE_FILE:-}"'" ]] && export GRB_LICENSE_FILE="'"${GRB_LICENSE_FILE:-}"'"
       echo "[prepare '"$TASK"'] $(wc -l < "'"$SHARD"'") models"
       "'"$NAJA_BIN"'" sample prepare \
         --models-root "'"$MODELS_ROOT"'" \
         --model-list "'"$SHARD"'" \
         '"${PREPARE_FLAGS:-}"'
     '
