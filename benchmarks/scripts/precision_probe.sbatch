#!/usr/bin/env bash
#SBATCH --job-name=naja-precision-probe
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
set -euo pipefail
B="$PWD"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache" PYTHONDONTWRITEBYTECODE=1
"$B/env-cpu/bin/python" - "$B" <<'PY'
from pathlib import Path
import json,hashlib,sys
b=Path(sys.argv[1]);p=json.loads((b/'results/f64-build-verification/verification.json').read_text())
assert p['default_f32_old_new_bitwise_equal'] and p['f64_cast_to_f32_matches_original']
with (b/'build-h100-f64/naja').open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==p['binary_sha256']
PY
srun --cpu-bind=cores /usr/bin/env bash "$B/scripts/enter.sh" python "$B/naja_runner.py" \
  --exe "$B/build-h100-f64/naja" --model "$B/data/models/RECON1" \
  --out "$B/results/precision/RECON1/seed303/f64" --seed 303 --chains 16 \
  --draws 10000 --warmup 1000 --tpb 128 --sample-dtype float64 --timeout 600 --resume
