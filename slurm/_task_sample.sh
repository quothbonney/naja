#!/usr/bin/env bash
# Runs INSIDE the container for one sample.sbatch array task (one node).
# Args (positional, then `--` then pass-through sample flags):
#   1 MODELS_ROOT  2 SHARD  3 OUT  4 METADIR  5 NAJA_BIN  6 GPUS_CSV
#   7 STAGE_ROOT   8 SYNC_INTERVAL  9 TASK  10 REPO  11 GRB_LICENSE_FILE  -- <sample flags...>
set -euo pipefail

MODELS_ROOT=$1 SHARD=$2 OUT=$3 METADIR=$4 NAJA_BIN=$5 GPUS=$6 \
  STAGE_ROOT=$7 SYNC_INTERVAL=$8 TASK=$9 REPO=${10} GRB_LIC=${11}
shift 11
[[ "${1:-}" == "--" ]] && shift
SAMPLE_FLAGS=("$@")

STAGE="${STAGE_ROOT}/naja_stage_${SLURM_JOB_ID:-0}_${TASK}"
mkdir -p "$STAGE" "$OUT" "$METADIR"
export LD_LIBRARY_PATH="${REPO}/extern/gurobi/linux64/lib:${LD_LIBRARY_PATH:-}"
[[ -n "$GRB_LIC" ]] && export GRB_LICENSE_FILE="$GRB_LIC"

# Background rate-limited drain: node-local NVMe -> NFS. Uses cp + atomic mv
# (rsync is NOT present in the CUDA container). The glob matches only COMPLETE
# .npy (naja writes in-flight files as *.npy.tmp.<pid>.<dev>). We copy to a dest
# .part then rename (atomic on the OUT filesystem) so readers never see a partial
# file, then free the NVMe copy.
drain() {
  shopt -s nullglob
  local f b
  for f in "$STAGE"/*.npy; do
    b=$(basename "$f")
    if cp -f "$f" "$OUT/.$b.part" 2>/dev/null && mv -f "$OUT/.$b.part" "$OUT/$b" 2>/dev/null; then
      rm -f "$f" 2>/dev/null || true
    fi
  done
  shopt -u nullglob
}
( while true; do sleep "$SYNC_INTERVAL"; drain; done ) &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null || true; drain' EXIT

echo "[task $TASK] $(wc -l < "$SHARD") models | stage=$STAGE -> out=$OUT"
"$NAJA_BIN" sample bulk \
  --models-root "$MODELS_ROOT" \
  --model-list "$SHARD" \
  --out-root "$METADIR" \
  --name "shard_${TASK}" \
  --gpus "$GPUS" \
  --skip-existing \
  --flat-output "$OUT" \
  --stage-dir "$STAGE" \
  "${SAMPLE_FLAGS[@]}"

kill "$SYNC_PID" 2>/dev/null || true
drain   # final flush
cp -f "$METADIR"/shard_${TASK}*/bulk_summary.csv "$OUT"/_logs_shard_${TASK}.csv 2>/dev/null || true
echo "[task $TASK] done"
