#!/usr/bin/env python3
"""Refresh the prototype and enforce this campaign's explicit 12-hour limit."""
import argparse
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import subprocess
import time

B=Path(__file__).resolve().parents[1]
(B/'results/refresh.pid').write_text(str(os.getpid())+'\n')
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--until',default='2026-11-08T11:10:00+00:00')
p.add_argument('--once',action='store_true')
a=p.parse_args()
until=datetime.fromisoformat(a.until)
prefixes=('bench-','quality-','metrics-','recon1-','naja-polyround','diagnostic-mask','hopsy-tuned-default')
cancelled=False
while True:
    now=datetime.now(timezone.utc)
    if now>=until-timedelta(minutes=5) and not cancelled and not a.once:
        lines=subprocess.check_output(['/usr/bin/squeue','-h','-u',str(os.getuid()),'-o','%i|%j|%Z'],text=True).splitlines()
        owned=[]
        for line in lines:
            job,name,workdir=line.split('|',2)
            if name.startswith(prefixes) and Path(workdir) in (B,B.parent):
                base=job.split('_')[0].split('.')[0]
                if base.isdigit(): owned.append(base)
        owned=sorted(set(owned))
        if owned: subprocess.run(['/usr/bin/scancel',*owned],check=False)
        (B/'results/deadline-action.json').write_text(json.dumps({'utc':now.isoformat(),'cancelled_campaign_jobs':owned,'reason':'User-specified 12-hour maximum; preserve five minutes for final reporting.'},indent=2)+'\n')
        time.sleep(10)
        for path in (B/'results').rglob('run.json'):
            try: record=json.loads(path.read_text())
            except (OSError,json.JSONDecodeError): continue
            if record.get('status')=='running' and record.get('method') in ('naja','hopsy_chrr'):
                record.update(status='time_cap',error='Campaign deadline; unfinished samples are not used for diagnostics.')
                tmp=path.with_suffix('.deadline.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)
        cancelled=True
    # User requested a separate report for results since the utilization request.
    # Keep the earlier Markdown report and its figures as the historical snapshot.
    for script in ['summarize.py','scripts/summarize_quality.py','scripts/new_results_report.py']:
        if (B/script).exists():
            interpreter=B/('env-plot/bin/python' if script=='plots.py' else 'env-cpu/bin/python')
            subprocess.run([str(interpreter),str(B/script)],check=False)
    accounting=subprocess.run(['/usr/bin/sacct','-S','2026-11-07T23:15:00','-u',str(os.getuid()),
                               '--format=JobIDRaw,JobName%80,State,ExitCode,Elapsed,TotalCPU,AllocTRES%200','-P'],capture_output=True,text=True)
    if accounting.returncode==0:
        (B/'results/setup/slurm-accounting-latest.psv').write_text(accounting.stdout)
    (B/'results/refresh-status.json').write_text(json.dumps({'last_refresh_utc':now.isoformat(),'until_utc':until.isoformat(),'deadline_enforced':cancelled},indent=2)+'\n')
    if a.once or now>=until: break
    time.sleep(60)
