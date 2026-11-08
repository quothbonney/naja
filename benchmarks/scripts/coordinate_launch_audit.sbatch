#!/usr/bin/env bash
#SBATCH --job-name=bench-coordinate-launch-audit
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --nice=2000
#SBATCH --time=00:03:00
#SBATCH --deadline=2026-11-08T11:00:00
#SBATCH --output=benchmarks/logs/coordinate-launch-%j.log
set -euo pipefail
B="$SLURM_SUBMIT_DIR/benchmarks"
O="$B/results/coordinate_kernel/launch_audit"
mkdir -p "$O"
NSYS="$B/tools/nsight/extracted/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys"
nvidia-smi -q -x > "$O/hardware.xml"
bash "$B/scripts/enter.sh" "$NSYS" profile -t cuda --sample=none --cpuctxsw=none \
  -o "$O/trace" "$B/build-h100-coordinate/naja" sample run \
  --model-dir "$B/data/models/iJO1366" --out-root "$O/native" \
  --n-chains 512 --n-samples 8 --seed 101001 --tpb 128 --thinning 500 \
  --constraint-eps 0 --extra-constraint-eps 0 --pair-prob 0 \
  --bounds-policy ignore --write-npy --barrier-rotate --host-prefault-threads 8
bash "$B/scripts/enter.sh" "$NSYS" export --type sqlite --output "$O/trace.sqlite" "$O/trace.nsys-rep"
"$B/env-cpu/bin/python" - <<'PY'
import json, os, sqlite3
from pathlib import Path
B=Path(os.environ['SLURM_SUBMIT_DIR'])/'benchmarks'
O=B/'results/coordinate_kernel/launch_audit'
db=sqlite3.connect(O/'trace.sqlite')
names=[r[0] for r in db.execute("SELECT value FROM StringIds WHERE value LIKE '%chrKernel%'")]
(O/'kernel_names.json').write_text(json.dumps(names,indent=2)+'\n')
print(json.dumps(names,indent=2))
PY
