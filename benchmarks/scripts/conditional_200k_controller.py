#!/usr/bin/env python3
"""Durable bounded split200k pipeline, retaining original captures and all attempts."""
import argparse,fcntl,json,os,subprocess,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
B=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--max-gpu',type=int,default=2);p.add_argument('--max-cpu',type=int,default=4);a=p.parse_args();m=json.loads(a.manifest.read_text());R=Path(m['output_directory']);lock=(R/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(R/'pipeline.pid').write_text(str(os.getpid())+'\n');statepath=R/'pipeline_controller.json';state=json.loads(statepath.read_text()) if statepath.exists() else dict(jobs=[]);deadline=datetime.fromisoformat(m['deadline_utc'])
def read(path):
 try:return json.loads(Path(path).read_text())
 except (OSError,json.JSONDecodeError):return {}
def save():
 state.update(pid=os.getpid(),ppid=os.getppid(),updated_utc=datetime.now(timezone.utc).isoformat());temp=statepath.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(statepath)
while datetime.now(timezone.utc)<deadline:
 live=set(subprocess.check_output(['squeue','-h','-u',str(os.getuid()),'-o','%i'],text=True).split());active={kind:sum(j['kind']==kind and j['job'] in live for j in state['jobs']) for kind in ('gpu','cpu')}
 # Capture every predeclared pilot before spending GPU slots on retimes.
 ordered=sorted(enumerate(m['cases']),key=lambda z:(bool(read(R/z[1]['condition_id']/'pipeline_state.json').get('pilot_captured')),z[0]))
 for i,c in ordered:
  s=read(R/c['condition_id']/'pipeline_state.json')
  if s.get('terminal_status'):continue
  phase='capture-pilot' if not s.get('pilot_captured') else 'analyze-pilot' if not s.get('pilot_analyzed') else 'capture-selected' if not s.get('selected_captured') else 'analyze-selected'
  previous=[j for j in state['jobs'] if j['index']==i and j['phase']==phase]
  if previous:continue
  kind='gpu' if phase.startswith('capture') else 'cpu';minutes=20 if kind=='gpu' else 45
  if datetime.now(timezone.utc)>=deadline-timedelta(minutes=minutes) or active[kind]>=getattr(a,'max_'+kind):continue
  command=['sbatch','--parsable',f'--chdir={B.parent}','--partition=hpc-mid',f'--job-name=bench-cond200k-{phase}',f'--time=00:{minutes:02d}:00',f'--deadline={deadline.strftime("%Y-%m-%dT%H:%M:%S")}','--mem=128G',f'--cpus-per-task={4 if kind=="gpu" else 8}',f'--output={B}/logs/conditional-pipeline-%j.log']
  if kind=='gpu':command+=['--gpus=1']
  command+=[str(B/'scripts/conditional_pipeline.sbatch'),str(a.manifest.resolve()),str(i),phase]
  job=subprocess.check_output(command,text=True).strip().split(';')[0];state['jobs'].append(dict(job=job,index=i,condition_id=c['condition_id'],phase=phase,kind=kind,command=command));active[kind]+=1;save()
 state.update(active=active,deadline_utc=m['deadline_utc']);save()
 if all(read(R/c['condition_id']/'pipeline_state.json').get('terminal_status') for c in m['cases']):break
 time.sleep(20)
live=set(subprocess.check_output(['squeue','-h','-u',str(os.getuid()),'-o','%i'],text=True).split());owned=[j['job'] for j in state['jobs'] if j['job'] in live]
if owned:subprocess.run(['scancel',*owned],check=False)
state.update(stopped_utc=datetime.now(timezone.utc).isoformat(),cancelled_own_jobs=owned);save()
