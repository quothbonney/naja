"""Fresh F64 quality runs plus paired export-quantization controls."""
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import numpy as np

B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B))
import metrics
from compare_precision import Stats

def read(path):return json.loads(path.read_text())
def optional_read(path):
    try:return read(path)
    except (OSError,json.JSONDecodeError):return {}
def write(path,data):path.write_text(json.dumps(metrics.json_safe(data),indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.file_digest(path.open('rb'),'sha256').hexdigest()

def run(model,seed):
    root=B/'results/precision_quality'/model;out=root/f'seed{seed}'
    if (out/'quality.json').exists():return read(out/'quality.json')
    oldq=read(B/'results'/('quality/iJR904/naja128' if model=='iJR904' else 'larger_quality/iML1515/gpu')/f'seed{seed}/quality.json')
    old=Path(oldq['measured']['run_dir']);oldrun=read(old/'run.json')
    draws=20000 if model=='iJR904' else 2000
    thinning,tpb=(0,128) if model=='iJR904' else (1000,512)
    proof=read(B/'results/f64-build-verification/verification.json')
    exe=B/'build-h100-f64/naja'
    assert proof['gpu_correctness_tests_passed']==5 and sha(exe)==proof['binary_sha256']
    command=[str(B/'env-cpu/bin/python'),str(B/'naja_runner.py'),'--exe',str(exe),
        '--model',str(B/'data/models'/model),'--out',str(out),'--seed',str(seed),
        '--chains','16','--draws',str(draws),'--warmup','1000','--tpb',str(tpb),
        '--thinning',str(thinning),'--barrier','rotate','--pair-prob','0',
        '--sample-dtype','float64','--timeout','600']
    root.mkdir(parents=True,exist_ok=True)
    write(root.parent/'progress.json',dict(model=model,seed=seed,stage='sampling',job_id=os.getenv('SLURM_JOB_ID')))
    if optional_read(out/'run.json').get('status')!='complete':
        if out.exists():out.rename(root/f'seed{seed}-interrupted-{time.time_ns()}')
        with (root/f'seed{seed}-sampling.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=660)
    run=read(out/'run.json');assert run['status']=='complete'
    assert run['exe_sha256']==proof['binary_sha256'] and run['profile']['config']['seed']==seed
    assert run['profile']['config']['thinning']==oldrun['profile']['config']['thinning']
    assert run['profile']['config']['tpb_ss']==tpb
    modeldir=B/'data/models'/model
    write(root.parent/'progress.json',dict(model=model,seed=seed,stage='fresh_f64_metrics',job_id=os.getenv('SLURM_JOB_ID')))
    # Every quality/bounds result is recomputed on the actual F64 samples.
    if optional_read(out/'metrics.json').get('schema_version')!=2:
        metrics.evaluate(out/'samples.npy',modeldir,out,prefixes=[draws])
    m=read(out/'metrics.json')
    new=np.load(out/'samples.npy',mmap_mode='r');reference=np.load(old/'samples.npy',mmap_mode='r')
    assert new.dtype==np.float64 and reference.dtype==np.float32 and new.shape==reference.shape
    with np.load(modeldir/'rounding/polytope.npz') as p:A,b,T,shift=p['A'],p['b'].reshape(-1),p['T'],p['shift'].reshape(-1)
    lower=np.loadtxt(modeldir/'gem/l_bounds.csv',delimiter=',').reshape(-1)
    upper=np.loadtxt(modeldir/'gem/u_bounds.csv',delimiter=',').reshape(-1)
    ids=(modeldir/'gem/reaction_ids.txt').read_text().splitlines();active=np.load(modeldir/'active_reactions.npy')
    mask=read(modeldir/'diagnostic_mask.json');assert mask['geometry_sha256']==sha(modeldir/'rounding/polytope.npz')
    cases={name:Stats(len(ids),active) for name in ['float64_export','paired_float32_quantization']}
    write(root.parent/'progress.json',dict(model=model,seed=seed,stage='paired_export_analysis',job_id=os.getenv('SLURM_JOB_ID')))
    mismatches=0;max_delta=0.;begun=time.perf_counter()
    for lo in range(0,draws,256):
        hi=min(lo+256,draws);z=np.asarray(new[:,lo:hi]).reshape(-1,new.shape[-1]);q=z.astype(np.float32)
        old32=np.asarray(reference[:,lo:hi]).reshape(q.shape)
        mismatches+=int(np.count_nonzero(q.view(np.uint32)!=old32.view(np.uint32)))
        max_delta=max(max_delta,float(np.max(np.abs(q.astype(np.float64)-old32.astype(np.float64)))))
        for name,points in [('float64_export',z),('paired_float32_quantization',q.astype(np.float64))]:
            cases[name].update(points@T.T+shift,points,A,b,lower,upper)
    comparison=dict(model=model,seed=seed,shape=list(new.shape),historical_run=str(old),
        historical_binary_sha256=oldrun['exe_sha256'],new_binary_sha256=run['exe_sha256'],
        normalized_cast_to_float32_bit_mismatches=mismatches,normalized_cast_max_difference=max_delta,
        historical_float32_reproduced_bitwise=mismatches==0,geometry_sha256=mask['geometry_sha256'],
        no_filtering_or_clipping=True,representations={name:s.result(ids) for name,s in cases.items()},
        analysis_wall_seconds=time.perf_counter()-begun,
        causal_protocol='Same GPU F64 points, T@Z64+shift versus T@float64(float32(Z64))+shift; compare cast coordinates to historical frozen F32 output.')
    write(out/'precision_comparison.json',comparison)
    quality=bool(m.get('schema_version')==2 and m.get('diagnostic_mask_status')=='model_lp_verified'
        and m['split_rhat_median']<=1.01 and m['bulk_ess_median']>=100)
    strict=bool(m['sample_bound_violation_fraction']==0 and m['bounds_tolerance']==1e-7)
    result=dict(model=model,seed=seed,variant='verified_f64_export',status='complete',
        quality_met=quality,strict_bounds_met=strict,historical_float32_reproduced_bitwise=mismatches==0,
        run_dir=str(out),binary_sha256=run['exe_sha256'],sample_dtype='float64',
        draws_per_chain=draws,chains=16,warmup=1000,requested_thinning=thinning,tpb=tpb,
        wall_seconds=run['wall_including_normalization_seconds'],native_process_wall_seconds=run['wall_seconds'],
        sampling_phase_seconds=run['profile']['timing']['gpu_sampling_s'],
        profile=run['profile'],split_rhat_median=m['split_rhat_median'],bulk_ess_median=m['bulk_ess_median'],
        sample_bound_violation_fraction=m['sample_bound_violation_fraction'],max_absolute_bound_violation=m['max_absolute_bound_violation'],
        bounds_tolerance=m['bounds_tolerance'],precision_comparison=str(out/'precision_comparison.json'))
    write(out/'quality.json',result);print(json.dumps(result),flush=True)
    return result

def summarize(model):
    out=B/'results/precision_quality'/model
    rows=[read(p) for p in sorted(out.glob('seed*/quality.json'))]
    if not rows:return
    complete=len(rows)==3 and all(r['quality_met'] and r['strict_bounds_met'] for r in rows)
    summary=dict(model=model,variant='verified_f64_export',rows=rows,complete_three_seed_group=complete,
        median_wall_seconds=float(np.median([r['wall_seconds'] for r in rows])),
        median_sampling_phase_seconds=float(np.median([r['sampling_phase_seconds'] for r in rows])),
        note='Separate precision-corrected fresh runs; canonical F32 evidence is unchanged. F64 versus historical F32 timing is not a controlled export-only speed comparison.')
    write(out/'summary.json',summary)
    fields=[k for k in rows[0] if k!='profile']
    with (out/'table.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
    for model,first,remaining in [('iJR904',101,[202,303]),('iML1515',303,[101,202])]:
        result=run(model,first);summarize(model)
        if result['strict_bounds_met'] and result['quality_met']:
            for seed in remaining:run(model,seed);summarize(model)
    write(B/'results/precision_quality/progress.json',dict(stage='finished',job_id=os.getenv('SLURM_JOB_ID')))

if __name__=='__main__':main()
