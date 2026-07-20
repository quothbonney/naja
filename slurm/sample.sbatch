#!/bin/bash
#SBATCH --job-name=naja-sample
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --time=24:00:00
# --partition, --gres, --array, --output are supplied by naja-slurm at submit
# time (or on the sbatch command line). Driven by exported env vars so it stays
# cluster-agnostic and hand-runnable. Required env:
#   MODELS_ROOT SHARD_DIR OUT METADIR IMAGE NAJA_REPO NAJA_BIN
# optional: NAJA_GPUS_CSV NAJA_STAGE_ROOT NAJA_SYNC_INTERVAL SAMPLE_FLAGS GRB_LICENSE_FILE
#
# One array task = one whole B300 node: `sample bulk` work-steals across all 8
# GPUs, writing ~117 MB .npy to node-local NVMe; a background rsync drains them
# to NFS so sampling never contends on NFS. Per-node granularity chosen so the
# in-binary work-stealing keeps all 8 GPUs busy despite ~10 s/model host-prep.
set -euo pipefail

: "${MODELS_ROOT:?}"; : "${SHARD_DIR:?}"; : "${OUT:?}"; : "${METADIR:?}"
: "${IMAGE:?}"; : "${NAJA_REPO:?}"; : "${NAJA_BIN:?}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
SHARD="${SHARD_DIR}/shard_${TASK}.txt"
[[ -f "$SHARD" ]] || { echo "shard not found: $SHARD" >&2; exit 1; }
mkdir -p "$OUT" "$METADIR"

export ENROOT_DATA_PATH="/tmp/enroot-${USER}-${SLURM_JOB_ID}/data"
export ENROOT_RUNTIME_PATH="/tmp/enroot-${USER}-${SLURM_JOB_ID}/runtime"
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH"

# Mount repo (binary + vendored gurobi rpath), models, both NFS out dirs, and
# the gurobi license dir if set.
MOUNTS="${NAJA_REPO}:${NAJA_REPO},${MODELS_ROOT}:${MODELS_ROOT},${OUT}:${OUT},${METADIR}:${METADIR}"
if [[ -n "${GRB_LICENSE_FILE:-}" ]]; then
  MOUNTS="${MOUNTS},$(dirname "$GRB_LICENSE_FILE"):$(dirname "$GRB_LICENSE_FILE")"
fi
# Extra bind mounts (comma-separated src:dst). Needed when models inherit rounding
# via symlinks to a shared base dir OUTSIDE models-root (else the symlinks dangle
# in-container). Set NAJA_SAMPLE_MOUNTS=/path/to/base:/path/to/base.
if [[ -n "${NAJA_SAMPLE_MOUNTS:-}" ]]; then
  MOUNTS="${MOUNTS},${NAJA_SAMPLE_MOUNTS}"
fi

# Pass everything as ARGS to the task script (robust vs container env-propagation).
srun --container-image="$IMAGE" \
     --container-mounts="$MOUNTS" \
     --container-workdir="$NAJA_REPO" \
     --no-container-mount-home --container-remap-root --no-container-entrypoint \
     bash "${NAJA_REPO}/slurm/_task_sample.sh" \
       "$MODELS_ROOT" "$SHARD" "$OUT" "$METADIR" "$NAJA_BIN" \
       "${NAJA_GPUS_CSV:-0,1,2,3,4,5,6,7}" "${NAJA_STAGE_ROOT:-/tmp}" \
       "${NAJA_SYNC_INTERVAL:-45}" "$TASK" "$NAJA_REPO" "${GRB_LICENSE_FILE:-}" \
       -- ${SAMPLE_FLAGS:-}
