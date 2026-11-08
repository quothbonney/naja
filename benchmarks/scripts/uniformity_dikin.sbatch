#!/usr/bin/env bash
#SBATCH --job-name=uniformity-dikin
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
set -euo pipefail
B="$PWD"
export HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache" PYTHONDONTWRITEBYTECODE=1
srun --cpu-bind=cores /usr/bin/env bash "$B/scripts/enter.sh" python "$B/naja_runner.py" \
  --exe "$B/build-h100/naja" --model "$B/data/models/e_coli_core" \
  --out "$B/results/uniformity/naja_whiten_dikin_pair" --seed 3601 \
  --chains 16 --draws 62500 --warmup 1000 --barrier whiten --pair-prob .5 \
  --pair-schedule-method barrier --verbose --timeout 240 --resume
"$B/env-cpu/bin/python" - "$B" <<'PY'
from pathlib import Path
import json,re,sys,hashlib
b=Path(sys.argv[1]);out=b/'results/uniformity/naja_whiten_dikin_pair'
r=json.loads((out/'run.json').read_text());log=(out/'stdout.log').read_text()
match=re.search(r'dikin_directions\s+(\d+) escape directions \(full Hessian\)',log)
assert match and int(match[1])>0,'No confirmed positive Dikin direction count'
assert not r['profile']['config']['back_transform'],'Dikin disabled by backmap path'
assert '--backmap' not in r['command']
assert r['pair_schedule_method']=='barrier' and r['pair_probability']==.5
source=(b/'../naja-test-jack/naja/src/engine/job_sampling.cu').resolve()
s=source.read_text();assert 'dikin_prob = 0.15;' in s
proof={'confirmed_direction_count':int(match[1]),'back_transform':False,
 'dikin_probability':.15,'pair_probability':.5,'coordinate_probability':.35,
 'dikin_probability_evidence':'Hardcoded .15 assignment when direction count>0 in job_sampling.cu; reduced-output branch forwards dikin_prob to CoordinateHitAndRun; no backmap path used.',
 'runtime_evidence':match.group(0),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'binary_sha256':r['exe_sha256'],'command':r['command']}
(out/'mixture_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
PY
