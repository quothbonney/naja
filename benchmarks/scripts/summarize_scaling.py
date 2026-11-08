#!/usr/bin/env python3
"""Summarize fixed-protocol scaling; telemetry peaks are sampled, not exact."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--model',type=Path,required=True)
a=p.parse_args()

def epoch(s):
    return datetime.fromisoformat(s).timestamp()

telemetry=[]
with (a.root/'telemetry.csv').open() as f:
    for row in csv.DictReader(f,skipinitialspace=True):
        row={k.strip():v.strip() for k,v in row.items()}
        try:
            ts=datetime.strptime(row['timestamp'],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
            telemetry.append((ts,float(row['memory.used [MiB]']),float(row['utilization.gpu [%]'])))
        except (KeyError,ValueError):
            continue
with np.load(a.model/'rounding/polytope.npz') as bundle:
    A,b=bundle['A'],bundle['b']
rows=[]
for path in sorted(a.root.glob('*/run.json')):
    r=json.loads(path.read_text())
    if r.get('status')!='complete': continue
    profile=r['profile']; cfg=profile['config']; timing=profile['timing']
    start,end=epoch(r['started_utc']),epoch(r['finished_utc'])
    observed=[t for t in telemetry if start<=t[0]<=end]
    # Deterministically spread 1024 retained samples across chains and draws.
    samples=np.load(path.parent/'samples.npy',mmap_mode='r')
    flat=samples.reshape(-1,samples.shape[-1])
    indices=np.linspace(0,len(flat)-1,min(1024,len(flat)),dtype=np.int64)
    violation=(np.asarray(flat[indices],dtype=np.float64)@A.T-b).max(axis=1)
    wall=r.get('wall_including_normalization_seconds',r['wall_seconds']+r.get('normalization_seconds',0))
    retained=r['chains']*r['draws']
    rows.append(dict(case=path.parent.name,axis='chains' if path.parent.name.startswith('chains') else 'tpb',
        chains=r['chains'],tpb=cfg['tpb_ss'],draws_per_chain=r['draws'],warmup=r['warmup'],
        seed=r['seed'],thinning=cfg['thinning'],barrier=r['barrier'],pair_probability=r['pair_probability'],
        sampling_seconds=timing['gpu_sampling_s'],native_process_wall_seconds=r['wall_seconds'],
        normalization_seconds=r.get('normalization_seconds',0),end_to_end_seconds=wall,
        kernel_throughput_including_warmup=profile['performance']['throughput_samples_per_sec'],
        retained_samples_per_sampling_second=retained/timing['gpu_sampling_s'],
        retained_samples_per_end_to_end_second=retained/wall,
        sampled_peak_gpu_memory_mib=max((t[1] for t in observed),default=None),
        sampled_mean_gpu_utilization_pct=float(np.mean([t[2] for t in observed])) if observed else None,
        telemetry_samples=len(observed),feasibility_subset_count=len(indices),
        feasibility_subset_invalid_fraction_eps_1e6=float(np.mean(violation>1e-6)),
        feasibility_subset_max_violation=float(max(0,np.max(violation)))))
if not rows: raise RuntimeError('No completed scaling runs')
with (a.root/'summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
start=json.loads((a.root/'job_start.json').read_text())
end=json.loads((a.root/'job_end.json').read_text())
seconds=epoch(end['finished_utc'])-epoch(start['started_utc'])
best_kernel=max((r for r in rows if r['axis']=='chains'),key=lambda r:r['retained_samples_per_sampling_second'])
best_wall=max((r for r in rows if r['axis']=='chains'),key=lambda r:r['retained_samples_per_end_to_end_second'])
result={'model':a.model.name,'rows':rows,'sampling_job_wall_seconds':seconds,
        'sampling_job_allocated_gpu_hours':seconds/3600,'allocated_gpus':1,
        'telemetry_interval_seconds':1,'gpu_memory_peak_caveat':'1-second sampling may miss transient peaks; sampled device-wide usage on assigned GPU',
        'feasibility_caveat':'Only 1024 deterministic retained reduced-space samples per case; full-output feasibility not checked',
        'canonical_table_unchanged':True,'protocol':'One-factor chain/TPB sweeps; rotate, adaptive thinning, seed101; 10k retained +1k warmup per chain',
        'highest_measured_chain_kernel_throughput_case':best_kernel['case'],
        'highest_measured_chain_end_to_end_throughput_case':best_wall['case'],
        'job':start,'finished':end}
(a.root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2),flush=True)
