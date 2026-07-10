#!/usr/bin/env bash
# naja/slurm/config.sh — cluster defaults for the OPTIONAL Slurm layer.
# Everything here is overridable via environment. naja itself does not need any
# of this; the binary runs standalone (`naja sample bulk ...`). This file only
# configures how the slurm/ helpers submit jobs on a Slurm+Pyxis cluster.
#
# Tuned for the CoreWeave SUNK cluster (Slurm 25.05.3, Pyxis/enroot, 5x B300).

# --- partitions -------------------------------------------------------------
: "${NAJA_GPU_PARTITION:=hpc-mid}"        # GPU phases (sample). hpc-{low,mid,high,prod} priority tiers.
: "${NAJA_CPU_PARTITION:=cd-gp-i64-erapids}"  # CPU phases (prepare/reround/migrate); no GPUs there.

# --- GPU resources (per-node work-stealing: one array task = one whole node) --
: "${NAJA_GRES:=gpu:b300:8}"              # 8 B300 GPUs per node; the gres TYPE string is 'b300'.
: "${NAJA_GPUS_CSV:=0,1,2,3,4,5,6,7}"     # passed to `sample bulk --gpus`; must match NAJA_GRES count.

# --- container (Pyxis/enroot) ----------------------------------------------
: "${NAJA_IMAGE:=/mnt/home/jack/dist-lab/images/nvidia+pytorch+26.04-py3.sqsh}"
: "${NAJA_REPO:=/mnt/home/jack/naja}"
: "${NAJA_BIN:=${NAJA_REPO}/build/naja}"  # built by scripts/build_container.sh

# --- staging & scheduler caps ----------------------------------------------
: "${NAJA_STAGE_ROOT:=/tmp}"              # node-local NVMe (28 TB on B300). Not a schedulable TRES.
: "${NAJA_SYNC_INTERVAL:=45}"             # seconds between background stage->NFS rsyncs
: "${NAJA_MAX_ARRAY:=1000}"               # MaxArraySize-1 on this cluster (MaxArraySize=1001)

# --- gurobi license (needed only to RUN prepare/sample, not to build) -------
: "${GRB_LICENSE_FILE:=}"                 # export a valid Gurobi 13 license before running

# enroot per-job scratch (mirrors scripts/run_b300_naja_*.sbatch)
naja_enroot_env() {
  export ENROOT_DATA_PATH="/tmp/enroot-${USER}-${SLURM_JOB_ID}/data"
  export ENROOT_RUNTIME_PATH="/tmp/enroot-${USER}-${SLURM_JOB_ID}/runtime"
  mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH"
}
