#!/usr/bin/env python3
"""Submit bounded concurrent Slurm arrays as shared model preprocessing finishes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

B = Path(__file__).resolve().parents[1]
MODELS = ['e_coli_core','iAB_RBC_283','iJR904','iJO1366','iMM904','iAF1260','iML1515','RECON1','Recon3D']


def main():
    (B/'results/campaign.pid').write_text(str(os.getpid())+'\n')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hours',type=float,default=11)
    p.add_argument('--once',action='store_true')
    p.add_argument('--deadline-utc',default='2026-11-08T11:05:00+00:00')
    args=p.parse_args()
    deadline=datetime.fromisoformat(args.deadline_utc)
    path=B/'results/campaign-state.json'
    state=json.loads(path.read_text()) if path.exists() else {'started_utc':datetime.now(timezone.utc).isoformat(),'jobs':[]}
    end=time.monotonic()+args.hours*3600
    while time.monotonic()<end:
        remaining_minutes=int((deadline-datetime.now(timezone.utc)).total_seconds()/60)-10
        if remaining_minutes<10: break
        live=set(subprocess.check_output(['/usr/bin/squeue','-h','-u',str(os.getuid()),'-o','%F'],text=True).split())
        for method,limit in [('naja',4),('hopsy',2)]:
            active=sum(j['method']==method and str(j['job_id']) in live for j in state['jobs'])
            submitted={j['model'] for j in state['jobs'] if j['method']==method}
            for model in MODELS:
                if active>=limit: break
                if model in submitted or (method=='hopsy' and model=='e_coli_core'): continue
                if not (B/'data/models'/model/'meta.json').exists(): continue
                if model in ('RECON1','Recon3D'):
                    validation=B/'data/models'/model/'validation.json'
                    if not validation.exists() or json.loads(validation.read_text()).get('status')!='passed': continue
                gpu=method=='naja'
                cmd=['/usr/bin/sbatch','--parsable',f'--chdir={B}',f'--job-name=bench-{method}-{model}',
                     f'--output={B}/logs/{method}-{model}-%A_%a.log',
                     '--partition=h100' if gpu else '--partition=all',
                     '--array=0-2%1' if gpu else '--array=0-5%2',
                     '--cpus-per-task=4' if gpu else '--cpus-per-task=16',
                     '--mem=48G' if gpu else '--mem=64G',
                     f'--time={min(80 if gpu else 480,remaining_minutes)}',
                     '--deadline=2026-11-08T11:05:00']
                if gpu: cmd += ['--gres=gpu:1','--export=ALL,BENCH_SEPARATE_METRICS=1']
                cmd += [str(B/'scripts/sample.sbatch'),method,model]
                job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
                item={'method':method,'model':model,'job_id':job,'command':cmd,'submitted_utc':datetime.now(timezone.utc).isoformat()}
                if gpu:
                    diag=['/usr/bin/sbatch','--parsable',f'--chdir={B}',f'--job-name=metrics-{model}',
                          f'--output={B}/logs/metrics-{model}-%A_%a.log',f'--dependency=afterok:{job}',
                          '--partition=all','--cpus-per-task=4','--mem=32G','--time=01:00:00','--array=0-2%2',
                          str(B/'scripts/metrics.sbatch'),model]
                    item['metrics_job_id']=subprocess.check_output(diag,text=True).strip().split(';')[0]
                    item['metrics_command']=diag
                state['jobs'].append(item)
                path.write_text(json.dumps(state,indent=2)+'\n')
                print(f'Submitted {method} {model}: {job}',flush=True)
                active+=1
        if args.once: break
        completed=len(state['jobs'])==17 and not any(str(j['job_id']) in live for j in state['jobs'])
        if completed: break
        time.sleep(15)
    state['controller_finished_utc']=datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__': main()
