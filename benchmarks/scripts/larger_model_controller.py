"""Resume-safe bounded dispatch/report for additional matched-quality models."""
import csv
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
B=Path(__file__).resolve().parents[1]
ROOT=B.parent
OUT=B/'results/larger_quality';OUT.mkdir(exist_ok=True)
lock=(OUT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(OUT/'controller.pid').write_text(str(os.getpid())+'\n')
STATE=OUT/'controller-state.json'
state=json.loads(STATE.read_text()) if STATE.exists() else {'jobs':{'gpu_initial':'442','cpu2000_initial':'443','cpujade_initial':'444'},'expansions':{},'last_cpu_job':'444'}
MODELS=['iAF1260','iML1515','iMM904'];METHODS=['gpu','cpu2000','cpujade'];SEEDS=[101,202,303]
END=datetime(2026,11,8,11,0,tzinfo=timezone.utc)
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,json.JSONDecodeError):return {}
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def snapshot():
 rows=[];groups=[]
 for model in MODELS:
  for method in METHODS:
   rs=[]
   for seed in SEEDS:
    q=read(OUT/model/method/f'seed{seed}'/'quality.json')
    row={'model':model,'method':method,'seed':seed,'status':q.get('status','not_started'),**q.get('measured',{})};rows.append(row)
    if q.get('status')=='complete' and q.get('measured',{}).get('quality_met'):rs.append(row)
   g={'model':model,'method':method,'complete_passing_seeds':len(rs),'rows':rs}
   if len(rs)==3:g['wall_seconds_median']=statistics.median(r['wall_seconds'] for r in rs)
   groups.append(g)
 comparisons=[]
 for model in MODELS:
  gg=[g for g in groups if g['model']==model and g.get('complete_passing_seeds')==3]
  gpu=next((g for g in gg if g['method']=='gpu'),None);cpu=[g for g in gg if g['method']!='gpu']
  if gpu and cpu:
   best=min(cpu,key=lambda g:g['wall_seconds_median'])
   ratios=[min(g['rows'][i]['wall_seconds'] for g in cpu)/gpu['rows'][i]['wall_seconds'] for i in range(3)]
   comparisons.append(dict(model=model,gpu_median_seconds=gpu['wall_seconds_median'],cpu_best_complete_group=best['method'],cpu_group_median_seconds=best['wall_seconds_median'],ratio_of_group_medians=best['wall_seconds_median']/gpu['wall_seconds_median'],median_paired_fastest_tested_cpu_ratio=statistics.median(ratios)))
 write(OUT/'summary.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),groups=groups,comparisons=comparisons,notes='Fresh actual selected-budget processes,16chains,1000retainedwarmup; same mask/estimators. NativeCPU16physicalcores32logicalreserved. Pilot/search/diagnostics excluded, bounds separately recorded. Finite tested configurations; no global optimality.'))
 keys=['model','method','seed','status','draws_per_chain','wall_seconds','split_rhat_median','bulk_ess_median','quality_met','sample_bound_violation_fraction','max_absolute_bound_violation','bounds_tolerance','run_dir']
 tmp=OUT/'table.csv.tmp'
 with tmp.open('w') as f:
  w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
 tmp.replace(OUT/'table.csv')
 return rows
while datetime.now(timezone.utc)<END:
 for mi,model in enumerate(MODELS):
  for method in METHODS:
   key=f'{model}/{method}'
   if key in state['expansions']:continue
   q=read(OUT/model/method/'seed101/quality.json')
   if q.get('status') in [None,'running']:continue
   if q.get('status')!='complete':state['expansions'][key]={'status':'seed101_screen_not_passing','seed101_status':q.get('status')};continue
   tasks=f'{mi*3+1},{mi*3+2}%'+('2' if method=='gpu' else '1')
   cmd=['/usr/bin/sbatch','--parsable',f'--chdir={ROOT}',f'--job-name=quality-larger-{method}-{model}',f'--array={tasks}','--deadline=2026-11-08T11:00:00',f'--output={B}/logs/larger-{method}-%A_%a.log']
   if method=='gpu':cmd+=['--partition=h100','--cpus-per-task=8','--gres=gpu:1']
   else:cmd+=['--partition=all','--cpus-per-task=32',f'--dependency=afterany:{state["last_cpu_job"]}']
   cmd += [str(B/'scripts/larger_model_quality.sbatch'),method]
   job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
   state['expansions'][key]={'status':'submitted','job_id':job,'command':cmd}
   if method!='gpu':state['last_cpu_job']=job
   write(STATE,state);print(f'Submitted {key}: {job}',flush=True)
 rows=snapshot();state['updated_utc']=datetime.now(timezone.utc).isoformat();write(STATE,state)
 live=set(subprocess.check_output(['/usr/bin/squeue','-h','-u',str(os.getuid()),'-o','%F'],text=True).split())
 owned=set(state['jobs'].values())|{x['job_id'] for x in state['expansions'].values() if 'job_id'in x}
 if len(state['expansions'])==9 and not (owned&live):break
 time.sleep(20)
snapshot();state['finished_utc']=datetime.now(timezone.utc).isoformat();write(STATE,state)
