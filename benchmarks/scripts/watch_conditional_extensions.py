#!/usr/bin/env python3
"""Bounded read-only reporting/Slurm accounting for the three extension studies."""
import json,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
B=Path(__file__).resolve().parents[1];R=B/'results/conditioned_extensions';(R/'reports.pid').write_text(str(os.getpid())+'\n');end=datetime.fromisoformat('2026-11-08T22:02:00+00:00')
studies=['off_rotate_whiten_20261108','rotate_200k_20261108','whiten_pair_mixture_20261108']
while True:
 records=[]
 for name in studies:
  root=R/name
  with (root/'report_refresh.log').open('w') as log:
   result=subprocess.run([str(B/'env-plot/bin/python'),str(B/'scripts/summarize_conditioned_ablation.py'),'--study',str(root)],stdout=log,stderr=subprocess.STDOUT,timeout=90)
  ids=[]
  for path in [root/'slurm_job.json',root/'pipeline_controller.json']:
   if path.exists():
    state=json.loads(path.read_text());ids+=([state['job']] if state.get('job') else [r['job'] for r in state.get('jobs',[])])
  if ids:
   ac=subprocess.run(['sacct','-n','-X','-j',','.join(sorted(set(ids))),'-o','JobID,State,ExitCode,Elapsed,AllocCPUS,ReqMem','-P'],capture_output=True,text=True)
   if ac.returncode==0:(root/'slurm-accounting-latest.psv').write_text(ac.stdout)
  records.append(dict(study=name,summary_returncode=result.returncode))
 now=datetime.now(timezone.utc);(R/'reports_status.json').write_text(json.dumps(dict(updated_utc=now.isoformat(),pid=os.getpid(),ppid=os.getppid(),until_utc=end.isoformat(),studies=records),indent=2)+'\n')
 if now>=end:break
 time.sleep(120)
