#!/usr/bin/env python3
"""Launch bounded exact-budget retiming as verified source diagnostics appear."""
import argparse
import fcntl
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
lock=(state_path.parent/'controller.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(state_path.parent/'controller.pid').write_text(str(os.getpid())+'\n')
state=json.loads(state_path.read_text()) if state_path.exists() else {'jobs':[],'source_nonqualifiers':[]}
models=['e_coli_core','iAB_RBC_283','iJR904','iJO1366']
tasks=[]
for model in models:
    variants=['naja128','cpu_default16','cpu_jade16']
    if model=='e_coli_core': variants+=['cpu_default1','cpu_jade1','cpu_nativejade1']
    if model in ['iAB_RBC_283','iJR904']: variants+=['cpu_nativejade16']
    if model=='iJO1366': variants=['naja128','naja512','cpu_native2000','cpu_native5000','cpu_fixed500','cpu_jade16','cpu_fixed2000','cpu_fixed5000','cpu_fixed100','cpu_default16']
    tasks.extend((model,variant,seed) for variant in variants for seed in [101,202,303])
end=time.monotonic()+a.hours*3600
while time.monotonic()<end:
    live=set(subprocess.check_output(['/usr/bin/squeue','-h','-u',str(os.getuid()),'-o','%F'],text=True).split())
    launched={(j['model'],j['variant'],j['seed']) for j in state['jobs']}
    skipped={(j['model'],j['variant'],j['seed']) for j in state['source_nonqualifiers']}
    active={method:sum(VARIANTS[j['variant']]['method']==method and j['job_id'] in live for j in state['jobs'])
            for method in ['naja','hopsy']}
    for model,variant,seed in tasks:
        key=(model,variant,seed)
        remeasure=False
        resume=False
        if key in launched:
            prior=[j for j in state['jobs'] if (j['model'],j['variant'],j['seed'])==key]
            if any(j['job_id'] in live for j in prior): continue
            qp=B/'results/quality'/model/variant/f'seed{seed}'/'quality.json'
            try: q=json.loads(qp.read_text())
            except (OSError,json.JSONDecodeError): continue
            measured_path=q.get('measured',{}).get('run_dir')
            remeasure=bool(q.get('status')=='complete' and measured_path and any(attempt.get('role') in ['extension_budget','budget_finding_pilot'] and attempt.get('run_dir')==measured_path for attempt in q.get('attempts',[])))
            resume=q.get('status')=='running' and len(prior)<2
            if not remeasure and not resume: continue
        if key in skipped: continue
        v=VARIANTS[variant];method=v['method']
        source=source_run(model,variant,seed)
        if variant in ['cpu_nativejade1','cpu_nativejade16']:
            native_iJO=[B/'results/quality/iJO1366/cpu_native2000'/f'seed{s}'/'quality.json' for s in [101,202,303]]
            if not all(path.exists() and json.loads(path.read_text()).get('status') not in ['running'] for path in native_iJO): continue
        defer_reason=None
        if v.get('defer'): defer_reason='Further fixed100 extensions deferred in favor of native CPU quality tuning.'
        if variant=='cpu_native5000':
            native_source=source_run(model,'cpu_native2000',seed)
            pilot_metrics=(native_source/'metrics.json') if native_source else B/'results/quality'/model/'cpu_native2000'/f'seed{seed}'/'pilot-budget10000'/'metrics.json'
            if not pilot_metrics.exists(): continue
            nm=json.loads(pilot_metrics.read_text())
            if nm.get('first_quality_prefix_draws') is not None:
                defer_reason='Native2000 meets target within10k; native5000 conditional expansion not needed.'
        if defer_reason:
            out=B/'results/quality'/model/variant/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
            result={'model':model,'variant':variant,'seed':seed,'status':'deferred_by_declared_priority',
                    'note':defer_reason,'source_run':str(source) if source else None}
            qp=out/'quality.json'
            if qp.exists():
                previous=json.loads(qp.read_text());result['prior_record']=previous
            qp.write_text(json.dumps(result,indent=2)+'\n')
            state['source_nonqualifiers'].append(result)
            continue

        if not (B/'data/models'/model/'diagnostic_mask.json').exists(): continue
        if source is not None:
            try: m=json.loads((source/'metrics.json').read_text())
            except (OSError,json.JSONDecodeError): continue
            if m.get('schema_version')!=2 or m.get('diagnostic_mask_status')!='model_lp_verified': continue
            if m.get('first_quality_prefix_draws') is None and method=='hopsy' and not(v.get('extend') or model=='e_coli_core' and variant=='cpu_default1'):
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
        if remeasure: command+=['--remeasure-final']
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        state['jobs'].append({'model':model,'variant':variant,'seed':seed,'job_id':job,'command':command,
                              'submitted_utc':datetime.now(timezone.utc).isoformat()})
        active[method]+=1
        print(f'Submitted {model} {variant} seed{seed}: {job}',flush=True)
    state['updated_utc']=datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state,indent=2)+'\n')
    with (state_path.parent/'progress-summary.log').open('w') as log:
        subprocess.run(['/usr/bin/python3',str(B/'scripts/summarize_quality.py')],stdout=log,stderr=subprocess.STDOUT,timeout=30)
    covered={(j['model'],j['variant'],j['seed']) for j in state['jobs']+state['source_nonqualifiers']}
    if a.once or len(covered)>=len(tasks) and not sum(active.values()): break
    time.sleep(15)
