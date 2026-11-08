"""Collect the three requested 10kHz GH100 timelines and paired wall times."""
from pathlib import Path
import hashlib
import json
import os
import shlex
import subprocess
import time

B = Path(__file__).resolve().parents[1]
O = B/'results/nsight_timeline'
NSYS = B/'tools/nsight/extracted/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys'
PYTHON = B/'env-cpu/bin/python'
cases = [('chains16',16,'build-h100-safe',0),
         ('chains512',512,'build-h100-safe',0),
         ('chains512_prefault8',512,'build-h100-prefault',8)]
all_commands=(O/'commands.sh').read_text().splitlines() if (O/'commands.sh').exists() else []
for label,chains,build,prefault in cases:
    out=O/label
    out.mkdir(parents=True,exist_ok=True)
    manifest=dict(label=label,chains=chains,seed=101,barrier='rotate',tpb=128,
        requested_thinning=0,warmup=1000,retained_draws=10000,
        host_prefault_threads=prefault,binary=str(B/build/'naja'),
        binary_sha256=hashlib.file_digest((B/build/'naja').open('rb'),'sha256').hexdigest(),
        slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},
        cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),runs=[])
    if (out/'manifest.json').exists():manifest=json.loads((out/'manifest.json').read_text())
    for phase in ['unprofiled_before','profiled','unprofiled_after']:
        target=out/phase
        target.mkdir(exist_ok=True)
        if (target/'target-wall.json').exists():
            continue
        cmd=[str(B/build/'naja'),'sample','run','--model-dir',str(B/'data/models/iJO1366'),
            '--out-root',str(target/'native'),'--gpu','0','--n-chains',str(chains),
            '--n-samples','11000','--seed','101','--tpb','128','--thinning','0',
            '--constraint-eps','0','--extra-constraint-eps','0','--pair-prob','0',
            '--bounds-policy','ignore','--write-npy','--barrier-rotate']
        if prefault:cmd+=['--host-prefault-threads',str(prefault)]
        timed=[str(PYTHON),str(B/'scripts/nsight_timed_exec.py'),str(target/'target-wall.json'),*cmd]
        launch=([str(NSYS),'profile','-t','cuda,nvtx,osrt',
            '--gpu-metrics-devices=0','--gpu-metrics-frequency=10000','--gpu-metrics-set=gh100',
            '-o',str(out/'timeline'),*timed] if phase=='profiled' else timed)
        all_commands.append(shlex.join(launch))
        (O/'commands.sh').write_text('# Commands executed inside the localized CUDA environment.\n'+'\n'.join(all_commands)+'\n')
        start=time.perf_counter()
        with (target/'stdout.log').open('w') as stdout,(target/'stderr.log').open('w') as stderr:
            result=subprocess.run(launch,stdout=stdout,stderr=stderr,timeout=600)
        record=dict(phase=phase,command=launch,outer_wall_seconds=time.perf_counter()-start,returncode=result.returncode)
        if result.returncode:raise RuntimeError(record)
        record.update(json.loads((target/'target-wall.json').read_text()))
        profiles=list((target/'native').glob('*/profile.json'))
        assert len(profiles)==1,profiles
        record['naja_profile']=json.loads(profiles[0].read_text())
        assert record['naja_profile']['config']['thinning']==96
        manifest['runs'].append(record)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(label,phase,record['target_wall_seconds'],flush=True)
    export=[str(NSYS),'export','--type','sqlite','--output',str(out/'timeline.sqlite'),str(out/'timeline.nsys-rep')]
    all_commands.append(shlex.join(export))
    if not (out/'timeline.sqlite').exists():
        with (out/'export.log').open('w') as log:subprocess.run(export,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    (O/'commands.sh').write_text('# Commands executed inside the localized CUDA environment.\n'+'\n'.join(all_commands)+'\n')
