"""Small counter lookup/check on the merged release; no convergence campaign."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
import numpy as np

B = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--exe', type=Path, default=B/'build-closeout-release/naja')
parser.add_argument('--output', type=Path, default=B/'results/closeout/task4/final_release')
args = parser.parse_args()
O = args.output.resolve()
O.mkdir(exist_ok=True)
EXE = args.exe.resolve()
NCU = B/'tools/nsight/extracted/opt/nvidia/nsight-compute/2025.2.1/ncu'
def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
commit = subprocess.check_output(['git','-C',str(B/'closeout-release'),'rev-parse','HEAD'], text=True).strip()
cases = [('base', B/'data/models/iJO1366'),
         ('exact_PGI_KO', B/'results/knockout_panel/conditions/b4025/models/shared')]
rows = []
for label, model in cases:
    out = O/label
    out.mkdir(exist_ok=True)
    with np.load(model/'rounding/polytope.npz') as z:
        m, d = z['A'].shape
        reactions = z['T'].shape[0]
    cmd = [str(NCU), '--metrics', 'lts__t_bytes.sum,dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum',
        '--clock-control','none','--launch-count','1','-k','regex:.*chrKernel.*','--export',str(out/'report'),
        str(EXE),'sample','run','--model-dir',str(model),'--out-root',str(out/'native'),
        '--gpu','0','--n-chains','512','--n-samples','1200','--seed','101','--tpb','128','--thinning','96',
        '--constraint-eps','0','--extra-constraint-eps','0','--pair-prob','0','--bounds-policy','ignore',
        '--write-npy','--sample-dtype','float32','--host-prefault-threads','0','--barrier-rotate',
        '--barrier-rotate-backend','gpu']
    (out/'command.sh').write_text(shlex.join(cmd)+'\n')
    started = time.perf_counter()
    with (out/'stdout.log').open('w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=600)
    record = dict(case=label, commit=commit, binary_sha256=sha(EXE), geometry_sha256=sha(model/'rounding/polytope.npz'),
        m=m, d=d, reactions=reactions, command=cmd, process_seconds=time.perf_counter()-started,
        returncode=result.returncode, chains=512, native_draws=1200, thinning=96,
        total_chain_coordinate_steps=512*1200*96,
        output_reduced_float32_bytes_per_sample=4*d,
        output_reaction_float32_bytes_per_sample=4*reactions,
        scope='Sampling kernel only; counters include all1200 native draws. File payload excludes NPY header. No convergence claim.')
    if result.returncode == 0:
        with (out/'metrics.csv').open('w') as f:
            subprocess.run([str(NCU),'--import',str(out/'report.ncu-rep'),'--page','raw','--csv'], stdout=f,
                           stderr=(out/'export.stderr').open('w'), check=True, timeout=120)
        raw = list(csv.reader((out/'metrics.csv').open()))
        assert len(raw) == 3
        metrics = {n: dict(unit=u, value=v) for n,u,v in zip(*raw)}
        (out/'parsed_metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
        def count(name):
            metric = metrics[name]
            scale = {'byte':1,'Kbyte':1e3,'Mbyte':1e6,'Gbyte':1e9}[metric['unit']]
            return float(metric['value'].replace(',',''))*scale
        record['L2_bytes_per_chain_step'] = count('lts__t_bytes.sum')/record['total_chain_coordinate_steps']
        record['DRAM_bytes_per_chain_step'] = count('dram__bytes.sum')/record['total_chain_coordinate_steps']
        record['total_L2_bytes'] = count('lts__t_bytes.sum')
        record['total_DRAM_bytes'] = count('dram__bytes.sum')
    (out/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    rows.append(record)
    (O/'summary.json').write_text(json.dumps(dict(rows=rows,
        requested_1500_row_KO_available=False,
        limitation='No m~1500 knockout model exists in the supplied local artifacts. Actual exact PGI KO row count is reported; no duplicated constraints or synthetic condition is substituted.'), indent=2)+'\n')
    print(label, record.get('L2_bytes_per_chain_step'), record.get('DRAM_bytes_per_chain_step'), flush=True)
