#!/usr/bin/env python3
"""Launch bounded exact-budget retiming as verified source diagnostics appear."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from quality import B,VARIANTS,source_run

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--hours',type=float,default=10)
p.add_argument('--once',action='store_true')
a=p.parse_args()
state_path=B/'results/quality/state.json';state_path.parent.mkdir(parents=True,exist_ok=True)
state=json.loads(state_path.read_text()) if state_path.exists() else {'jobs':[],'source_nonqualifiers':[]}
models=['e_coli_core','iAB_RBC_283','iJR904','iJO1366']
tasks=[]
for model in models:
    variants=['naja128','cpu_default16','cpu_jade16']
    if model=='e_coli_core': variants+=['cpu_default1','cpu_jade1']
    if model=='iJO1366': variants+=['naja512','cpu_fixed100','cpu_fixed500']
    tasks.extend((model,variant,seed) for variant in variants for seed in [101,202,303])
end=time.monotonic()+a.hours*3600
while time.monotonic()<end:
    live=set(subprocess.check_output(['/usr/bin/squeue','-h','-u',str(os.getuid()),'-o','%F'],text=True).split())
    launched={(j['model'],j['variant'],j['seed']) for j in state['jobs']}
    skipped={(j['model'],j['variant'],j['seed']) for j in state['source_nonqualifiers']}
    active={method:sum(VARIANTS[j['variant']]['method']==method and j['job_id'] in live for j in state['jobs'])
            for method in ['naja','hopsy']}
    for model,variant,seed in tasks:
        if (model,variant,seed) in launched|skipped: continue
        v=VARIANTS[variant];method=v['method']
        source=source_run(model,variant,seed)
        if not (B/'data/models'/model/'diagnostic_mask.json').exists(): continue
        if source is not None:
            try: m=json.loads((source/'metrics.json').read_text())
            except (OSError,json.JSONDecodeError): continue
            if m.get('schema_version')!=2 or m.get('diagnostic_mask_status')!='model_lp_verified': continue
            if m.get('first_quality_prefix_draws') is None and method=='hopsy' and not(model=='e_coli_core' and variant=='cpu_default1'):
                out=B/'results/quality'/model/variant/f'seed{seed}'
                out.mkdir(parents=True,exist_ok=True)
                result={'model':model,'variant':variant,'seed':seed,'status':'no_quality_within_source_budget',
                        'source_run':str(source),'source_draws_per_chain':m['shape'][1],
                        'source_split_rhat_median':m['split_rhat_median'],'source_bulk_ess_median':m['bulk_ess_median'],
                        'note':'No passing evaluated prefix; no extrapolation or measured qualifying time claimed.'}
                (out/'quality.json').write_text(json.dumps(result,indent=2)+'\n')
                state['source_nonqualifiers'].append(result);continue
        if active[method]>=(1 if method=='naja' else 2): continue
        cpus=8 if method=='naja' or v.get('workers')==1 else 16
        command=['/usr/bin/sbatch','--parsable',f'--chdir={B}',f'--job-name=quality-{model}-{variant}-{seed}',
                 f'--output={B}/logs/quality-{model}-{variant}-{seed}-%j.log',
                 '--partition=h100' if method=='naja' else '--partition=all',f'--cpus-per-task={cpus}',
                 '--mem=64G' if method=='naja' else '--mem=48G','--time=00:30:00']
        if method=='naja': command+=['--gres=gpu:1']
        command += [str(B/'scripts/quality.sbatch'),model,variant,str(seed)]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        state['jobs'].append({'model':model,'variant':variant,'seed':seed,'job_id':job,'command':command,
                              'submitted_utc':datetime.now(timezone.utc).isoformat()})
        active[method]+=1
        print(f'Submitted {model} {variant} seed{seed}: {job}',flush=True)
    state['updated_utc']=datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state,indent=2)+'\n')
    if a.once or len(state['jobs'])+len(state['source_nonqualifiers'])==len(tasks) and not any(j['job_id'] in live for j in state['jobs']): break
    time.sleep(15)
