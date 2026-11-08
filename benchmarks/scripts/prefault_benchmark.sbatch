#!/usr/bin/env bash
#SBATCH --job-name=naja-host-prefault
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:15:00
set -euo pipefail
B="$PWD"
OUT="$B/results/host_prefault/iJO1366"
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
FREE_MIB="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits)"
if (( FREE_MIB < 60000 )); then echo "Insufficient free device memory for exploratory512-chain run: $FREE_MIB MiB" >&2; exit 1; fi
printf '%s\n' "$GPU_ID" > "$OUT/telemetry_gpu_pci_id.txt"
nvidia-smi -i "$GPU_ID" > "$OUT/nvidia-smi.txt"
nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,nounits --loop=1 > "$OUT/telemetry.csv" &
TELEMETRY_PID=$!
cleanup() {
  kill "$TELEMETRY_PID" 2>/dev/null || true
  wait "$TELEMETRY_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
srun --cpu-bind=cores /usr/bin/env bash "$B/scripts/enter.sh" python "$B/scripts/prefault_benchmark.py" --out "$OUT"
cleanup
trap - EXIT INT TERM
"$B/env-cpu/bin/python" - "$OUT/job_end.json" <<'PY'
from datetime import datetime,timezone
from pathlib import Path
import json,sys
Path(sys.argv[1]).write_text(json.dumps({'finished_utc':datetime.now(timezone.utc).isoformat()})+'\n')
PY
