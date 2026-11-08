#!/usr/bin/env python3
"""Persist Recon3D preprocessing attempts without modifying any geometry."""
from datetime import datetime,timezone
import json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
specs=[('original','150_6','prep-150_6.log','Recon3D'),
       ('full_parallel_FVA','363','zero-prune-363.log',None),
       ('zero_pruned_default_noop','367','zero-prune-367.log','Recon3D_zero_fva_default'),
       ('zero_pruned_presolve_noop','368','zero-prune-368.log','Recon3D_zero_fva_presolve'),
       ('FVA_faces_upstream_assembly','386','zero-prune-386.log','Recon3D_zero_fva_faces'),
       ('FVA_faces_bulk_assembly','401','zero-prune-401.log','Recon3D_zero_fva_faces_fast')]
data=subprocess.check_output(['/usr/bin/sacct','-j',','.join(r[1] for r in specs),
     '--format=JobID,State,Elapsed,Start,End,ExitCode','-P','-n'],text=True)
states={r[0]:r[1:] for r in (s.split('|') for s in data.splitlines()) if len(r)==6}
rows=[]
for name,job,log,model in specs:
    state,elapsed,start,end,exit_code=states.get(job,['UNKNOWN']*5)
    row=dict(attempt=name,slurm_job_id=job,status=state,elapsed=elapsed,start=start,end=end,
             exit_code=exit_code,log='benchmarks/logs/'+log,model_dir='benchmarks/data/models/'+model if model else None)
    if job in ('367','368'):row['note']='Cancelled: full FVA certified no zero reactions, so these duplicate candidates offered no variable reduction.'
    if job=='386':row['note']='Replaced by opt-in exact-value bulk CSR assembly after coefficient equivalence and full core validation.'
    rows.append(row)
result=dict(model='Recon3D',updated_utc=datetime.now(timezone.utc).isoformat(),attempts=rows,
            original_bounds_unchanged=True,FVA_precision=1e-9,inactive_face_margin=1e-3,
            proof_note='benchmarks/audit/zero_pruning.md',bulk_assembly_evidence='benchmarks/audit/fast_sparse_assembly.json')
fva=root/'data/preprocessing/Recon3D_zero_fva/fva.json'
if fva.exists():result['FVA']=json.loads(fva.read_text())
validation=root/'data/models/Recon3D/validation.json'
if validation.exists():result['published_validation']=json.loads(validation.read_text())
out=root/'results/setup/Recon3D_preprocessing_attempts.json';tmp=out.with_suffix('.tmp')
tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(out)
print(out)
