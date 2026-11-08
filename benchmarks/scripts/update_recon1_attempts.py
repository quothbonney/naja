#!/usr/bin/env python3
"""Refresh the persistent RECON1 preprocessing audit from Slurm accounting."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[1]
specs=[
 ('default_glpk','150_5','glpk',False,True,'prep-150_5.log','RECON1',False),
 ('glpk_presolve','224','glpk',True,True,'recon1-presolve.log','RECON1',False),
 ('hybrid_initial_allocation','258','hybrid',True,True,'recon1-hybrid-attempt30min.log','RECON1_hybrid',False),
 ('hybrid_two_hour_retry','277','hybrid',True,True,'recon1-hybrid.log','RECON1_hybrid',False),
 ('glpk_conservative','292','glpk',False,False,'recon1-conservative.log','RECON1_conservative',False),
 ('hybrid_failure_capture','296','hybrid',True,True,'recon1-lp-capture.log','RECON1_capture',True),
 ('glpk_failure_capture','323','glpk',False,True,'recon1-glpk-capture.log','RECON1_glpk_capture',True),
]
text=subprocess.check_output(['/usr/bin/sacct','-j',','.join(r[1] for r in specs),
    '--format=JobID,State,Elapsed,Start,End,ExitCode','-P','-n'],text=True)
accounting={r[0]:r[1:] for r in (line.split('|') for line in text.splitlines()) if len(r)==6}
attempts=[]
for name,job,backend,presolve,reduce,log,output,diagnostic in specs:
    state,elapsed,start,end,exit_code=accounting.get(job,['UNKNOWN']*5)
    content=(root/'logs'/log).read_text(errors='replace') if (root/'logs'/log).exists() else ''
    failures=[line.strip() for line in content.splitlines() if
              line.startswith(('ValueError:','RuntimeError:'))]
    entry=dict(attempt=name,slurm_job_id=job,backend=backend,
               settings=dict(presolve=presolve,reduce=reduce,feasibility_tolerance=1e-9),
               status=state.lower().split()[0],elapsed=elapsed,start_utc=start,end_utc=end,
               exit_code=exit_code,log='benchmarks/logs/'+log,
               model_dir='benchmarks/data/models/'+output,diagnostic_only=diagnostic,
               error=failures[-1] if failures else None)
    if job=='258':
        entry['note']='Restarted after Slurm denied increasing a running 30-minute time limit; next allocation requested two hours.'
    attempts.append(entry)
report=dict(model='RECON1',updated_utc=datetime.now(timezone.utc).isoformat(),
            source='benchmarks/data/raw/RECON1.json',
            source_sha256=hashlib.sha256((root/'data/raw/RECON1.json').read_bytes()).hexdigest(),
            original_model_feasible=True,
            feasibility_evidence='benchmarks/audit/recon1_feasibility.json',
            biological_bounds_modified=False,attempts=attempts)
report['failed_lp_evidence']={}
for backend,name in [('hybrid','recon1_failed_lp_diagnosis.json'),
                     ('glpk','recon1_failed_lp_glpk_diagnosis.json')]:
    path=root/'audit'/name
    if path.exists(): report['failed_lp_evidence'][backend]=json.loads(path.read_text())
validation=root/'data/models/RECON1/validation.json'
if validation.exists(): report['published_validation']=json.loads(validation.read_text())
out=root/'results/setup/RECON1_preprocessing_attempts.json'
out.parent.mkdir(parents=True,exist_ok=True)
tmp=out.with_suffix('.json.tmp')
tmp.write_text(json.dumps(report,indent=2)+'\n')
tmp.replace(out)
print(out)
