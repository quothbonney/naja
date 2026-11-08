#!/usr/bin/env bash
#SBATCH --job-name=naja-scaling
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
set -euo pipefail
B="$PWD"
OUT="$B/results/scaling/iJO1366"
mkdir -p "$OUT"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache" TZ=UTC
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$B/env-cpu/bin/python" - "$OUT/job_start.json" <<'PY'
from datetime import datetime,timezone
from pathlib import Path
import json,os,sys
Path(sys.argv[1]).write_text(json.dumps({'started_utc':datetime.now(timezone.utc).isoformat(),
 'slurm':{k:v for k,v in os.environ.items() if k.startswith('SLURM_')},
 'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES')},indent=2)+'\n')
PY
# Resolve CUDA logical device zero to a PCI ID, so telemetry selects our GPU.
GPU_ID="$("$B/scripts/enter.sh" python - <<'PY'
import ctypes
lib=ctypes.CDLL('libcudart.so.12')
buf=ctypes.create_string_buffer(64)
code=lib.cudaDeviceGetPCIBusId(buf,len(buf),0)
if code: raise RuntimeError(f'cudaDeviceGetPCIBusId failed: {code}')
print(buf.value.decode())
PY
)"
printf '%s\n' "$GPU_ID" > "$OUT/telemetry_gpu_pci_id.txt"
nvidia-smi -i "$GPU_ID" > "$OUT/nvidia-smi.txt"
nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,nounits --loop=1 > "$OUT/telemetry.csv" &
TELEMETRY_PID=$!
cleanup() {
  kill "$TELEMETRY_PID" 2>/dev/null || true
  wait "$TELEMETRY_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
srun --cpu-bind=cores /usr/bin/env bash "$B/scripts/enter.sh" python "$B/scripts/scaling.py" \
  --model "$B/data/models/iJO1366" --out "$OUT" --axis both --draws 10000 --warmup 1000 --timeout 900
cleanup
trap - EXIT INT TERM
"$B/env-cpu/bin/python" - "$OUT/job_end.json" <<'PY'
from datetime import datetime,timezone
from pathlib import Path
import json,sys
Path(sys.argv[1]).write_text(json.dumps({'finished_utc':datetime.now(timezone.utc).isoformat()})+'\n')
PY
# Summary and deterministic subset feasibility use CPU only after telemetry stops.
srun --cpu-bind=cores /usr/bin/env "$B/env-cpu/bin/python" "$B/scripts/summarize_scaling.py" --root "$OUT" --model "$B/data/models/iJO1366"
