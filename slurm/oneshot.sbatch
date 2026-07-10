#!/bin/bash
#SBATCH --job-name=naja-oneshot
#SBATCH --nodes=1
#SBATCH --time=24:00:00
# Generic single-job runner in the naja container, for the non-array phases:
#   Phase A reround (once, CPU, hours) and Phase D migrate/assemble (single proc).
# Runs one command (ONESHOT_CMD) in the container. Choose CPU or GPU partition +
# resources at submit time via naja-slurm (reround/migrate -> CPU partition).
# Required env: IMAGE NAJA_REPO ONESHOT_CMD ; optional MOUNTS_EXTRA GRB_LICENSE_FILE
#
# Examples (set ONESHOT_CMD):
#   reround:  python /mnt/home/jack/rs10k_build/reround_ijo1366.py
#   migrate:  python /mnt/home/jack/rs10k_build/migrate_to_RS_V3.py --src ... --dst ...
set -euo pipefail
: "${IMAGE:?}"; : "${NAJA_REPO:?}"; : "${ONESHOT_CMD:?set ONESHOT_CMD to the command to run}"
export ENROOT_DATA_PATH="/tmp/enroot-${USER}-${SLURM_JOB_ID}/data"
export ENROOT_RUNTIME_PATH="/tmp/enroot-${USER}-${SLURM_JOB_ID}/runtime"
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH"
MOUNTS="${NAJA_REPO}:${NAJA_REPO}${MOUNTS_EXTRA:+,${MOUNTS_EXTRA}}"
[[ -n "${GRB_LICENSE_FILE:-}" ]] && MOUNTS="${MOUNTS},$(dirname "$GRB_LICENSE_FILE"):$(dirname "$GRB_LICENSE_FILE")"

srun --container-image="$IMAGE" --container-mounts="$MOUNTS" \
     --container-workdir="$NAJA_REPO" \
     --no-container-mount-home --container-remap-root --no-container-entrypoint \
     bash -c '
       set -euo pipefail
       export LD_LIBRARY_PATH="'"$NAJA_REPO"'/extern/gurobi/linux64/lib:${LD_LIBRARY_PATH:-}"
       [[ -n "'"${GRB_LICENSE_FILE:-}"'" ]] && export GRB_LICENSE_FILE="'"${GRB_LICENSE_FILE:-}"'"
       echo "[oneshot] $ONESHOT_CMD"
       '"$ONESHOT_CMD"'
     '
