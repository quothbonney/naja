#!/usr/bin/env python3
"""Persistent bounded consumer: capacity milestone, then verified exact KO panel."""
import argparse,fcntl,json,os,subprocess,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
B=Path(__file__).resolve().parents[1]
R=B/'results/conditioned_prototype'
p=argparse.ArgumentParser();p.add_argument('--deadline',default='2026-11-08T11:00:00+00:00');p.add_argument('--once',action='store_true');p.add_argument('--max-cpu',type=int,default=2);p.add_argument('--max-gpu',type=int,default=2);p.add_argument('--max-prep',type=int,default=2);a=p.parse_args()
R.mkdir(exist_ok=True);lock=(R/'quality_consumer.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(R/'quality_consumer.pid').write_text(str(os.getpid())+'\n')
statepath=R/'quality_consumer_state.json'
def read(path,default=None):
 try:return json.loads(Path(path).read_text())
 except (OSError,json.JSONDecodeError):return {} if default is None else default

def save(path,value):
 path=Path(path);tmp=path.with_name(path.name+f'.tmp.{os.getpid()}');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
state=read(statepath,dict(jobs=[],plots={},started_utc=datetime.now(timezone.utc).isoformat()))
if not state['jobs']:
 for row in read(R/'curve_jobs.json',[]):
  state['jobs'].append(dict(root=str(R/'iJO1366'),kind='gpu' if row['variant']=='naja_rotate' else 'cpu',variant=row['variant'],seed=row['seed'],job=row['job'],adopted=True))
end=datetime.fromisoformat(a.deadline)

def submit(root,kind,variant=None,seed=101,known_affinity_retry=False):
 command=['/usr/bin/sbatch','--parsable',f'--chdir={B.parent}','--time=00:45:00',f'--deadline={end.strftime("%Y-%m-%dT%H:%M:%S")}']
 if kind=='prep':command += ['--cpus-per-task=4','--mem=64G','--job-name=bench-condition-prepare',str(B/'scripts/conditional_prepare.sbatch'),str(root)]
 else:
  cpus=32 if kind=='cpu' else 4
  command += [f'--cpus-per-task={cpus}','--mem=128G' if kind=='cpu' else '--mem=64G',f'--job-name=quality-condition-{root.name}-{variant}-{seed}']
  if kind=='gpu':command+=['--gpus=1']
  command += [str(B/'scripts/conditioned_quality.sbatch'),'--condition-root',str(root),'--variant',variant,'--seed',str(seed)]
  if root==R/'iJO1366' and variant in ('stale_cpu','naja_rotate'):
   budget=50000 if variant=='stale_cpu' else 10000
   source=root/'runs'/variant/f'seed{seed}'/f'budget{budget}'
   if (source/'metrics.json').exists():command+=['--pilot-source',str(source)]
 job=subprocess.check_output(command,text=True).strip().split(';')[0]
 record=dict(root=str(root),kind=kind,variant=variant,seed=seed,job=job,known_affinity_retry=known_affinity_retry,command=command,submitted_utc=datetime.now(timezone.utc).isoformat())
 state['jobs'].append(record);save(statepath,state);print('submitted',json.dumps(record),flush=True);return record

while datetime.now(timezone.utc)<end:
 try:
  now=datetime.now(timezone.utc)
  live=set(subprocess.check_output(['/usr/bin/squeue','-h','-u',str(os.getuid()),'-o','%i'],text=True).split())
  active={kind:sum(j['kind']==kind and j['job'] in live for j in state['jobs']) for kind in ('cpu','gpu','prep')}
  capacity=R/'iJO1366';conditions=[capacity]
  # This is the sole geometry-release oracle. Never import the unvetted catalog.
  manifest=read(B/'results/knockout_panel/ready_manifest.json')
  preferred={'b4025':0,'b0154':1,'b0181':2,'b0343':3,'b0508':4,'b0002':5}
  for entry in sorted(manifest.get('conditions',[]),key=lambda e:(preferred.get(e.get('gene'),6),e.get('gene',''))):
   if entry.get('status')!='ready':continue
   directory=Path(entry.get('geometry_directory',''))
   if directory.name=='shared':root=directory.parent.parent
   else:root=B/'results/knockout_panel/conditions'/entry['gene']
   info=read(root/'condition.json')
   if info.get('status')=='ready' and info.get('no_bound_epsilon_relaxation'):conditions.append(root)
  # Stop launching with45minutes left; every subprocess itself honors11:00.
  launch=now<end-timedelta(minutes=45)
  for root in conditions:
   root=Path(root)
   if root!=capacity:
    prepared=read(root/'quality/preparation_status.json')
    prior=[j for j in state['jobs'] if j['root']==str(root) and j['kind']=='prep']
    if prepared.get('status')!='complete':
     if not prior and launch and active['prep']<a.max_prep:
      submit(root,'prep');active['prep']+=1
     if not prepared.get('shared_mask_ready'):continue
   for seed in ((101,202,303) if root==capacity else (101,)):
    for variant in ('global_cpu','stale_cpu','naja_rotate'):
     if root!=capacity and variant=='global_cpu' and read(root/'quality/preparation_status.json').get('status')!='complete':continue
     quality=read(root/'quality'/variant/f'seed{seed}'/'quality.json')
     prior=[j for j in state['jobs'] if j['root']==str(root) and j.get('variant')==variant and j.get('seed')==seed]
     retry=(quality.get('status')=='failed' and 'Sibling reservation incomplete' in quality.get('error','') and not any(j.get('known_affinity_retry') for j in prior))
     if quality.get('status') in ('complete','no_quality_within_cap','fresh_selected_budget_failed','failed') and not retry:continue
     if prior and (not retry or any(j['job'] in live for j in prior)):continue
     kind='gpu' if variant=='naja_rotate' else 'cpu'
     if launch and active[kind]<getattr(a,'max_'+kind):
      if retry:
       backup=root/'quality'/variant/f'seed{seed}'/'quality.failed_affinity.json'
       if not backup.exists():save(backup,quality)
      submit(root,kind,variant,seed,known_affinity_retry=retry);active[kind]+=1
  # Regenerate only changed per-condition artifacts, with no sampling or estimator mutation.
  for root in conditions:
   files=list((root/'quality').glob('*/seed*/quality.json'))+list((root/'quality').glob('*/seed*/curve.json'))
   fingerprint=max((x.stat().st_mtime_ns for x in files),default=0)
   if fingerprint and fingerprint!=state['plots'].get(str(root)):
    with (root/'quality/plot_refresh.log').open('w') as log:
     result=subprocess.run([str(B/'env-plot/bin/python'),str(B/'scripts/summarize_conditioned_quality.py'),'--condition-root',str(root)],stdout=log,stderr=subprocess.STDOUT,timeout=120)
    if result.returncode==0:state['plots'][str(root)]=fingerprint
  summaries=[]
  for root in conditions:
   summary=read(root/'quality/summary.json')
   if summary:summaries.append(dict(condition_root=str(root),summary=summary))
  save(B/'results/knockout_panel/quality_summary.json',dict(generated_utc=now.isoformat(),condition_count=len(conditions)-1,conditions=[s for s in summaries if s['condition_root']!=str(capacity)],note='Only stable ready_manifest targets consumed; one seed per panel target, three for capacity. Failed/capped/incomplete tasks are explicit.'))
  state.update(updated_utc=now.isoformat(),deadline_utc=a.deadline,stop_launch_utc=(end-timedelta(minutes=45)).isoformat(),slurm_deadline_argument=True,active=active,ready_conditions=len(conditions),pid=os.getpid(),ppid=os.getppid())
  save(statepath,state)
  if a.once:break
  time.sleep(20)
 except Exception as error:
  state['last_controller_error']=repr(error);state['updated_utc']=datetime.now(timezone.utc).isoformat();save(statepath,state)
  print('controller error',repr(error),flush=True)
  if a.once:raise
  time.sleep(20)

if not a.once:
 live=set(subprocess.check_output(['/usr/bin/squeue','-h','-u',str(os.getuid()),'-o','%i'],text=True).split())
 owned=[j['job'] for j in state['jobs'] if j['job'] in live]
 if owned:subprocess.run(['/usr/bin/scancel',*owned],check=False)
 state['deadline_action']=dict(utc=datetime.now(timezone.utc).isoformat(),cancelled_own_jobs=owned)
 save(statepath,state)
