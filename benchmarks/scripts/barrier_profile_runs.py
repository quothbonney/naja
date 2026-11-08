"""Paired barrier-stage traces with a single frozen experimental executable."""
from pathlib import Path
from functools import lru_cache
import csv
import hashlib
import json
import os
import re
import shlex
import sqlite3
import subprocess
import time
import numpy as np

B=Path(__file__).resolve().parents[1]
O=Path(os.environ.get('NAJA_BARRIER_PROFILE_OUT',str(B/'results/barrier_profile'))).resolve()
assert O.is_relative_to(B),'Profiling output must remain inside benchmarks/'
EXE=B/'build-h100-barrier-profile/naja'
EXPECTED_SHA='8a9387cb0339b83132191a43b9e04d86a28d9381287cdc22c8dac1bb2214b54f'
NSYS=B/'tools/nsight/extracted/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys'
PYTHON=B/'env-cpu/bin/python'
MODELS=[('iJO1366','rotate',1e-7),('RECON1','rotate',1e-6),('Recon3D_nullspace_only','whiten',1e-7)]

def read(p):return json.loads(p.read_text())
def dump(p,x):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(temp,p)
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def union_seconds(intervals):
    result=0.;end=-float('inf')
    for a,b in sorted(intervals):
        result+=max(0.,b-max(a,end));end=max(end,b)
    return result/1e9

@lru_cache(maxsize=3)
def geometry(model):
    p=B/'data/models'/model
    with np.load(p/'rounding/polytope.npz') as z:A,b,T,shift=(z[k] for k in ['A','b','T','shift'])
    return A,b.reshape(-1),T,shift.reshape(-1),np.loadtxt(p/'gem/l_bounds.csv',delimiter=',').reshape(-1),np.loadtxt(p/'gem/u_bounds.csv',delimiter=',').reshape(-1)

def feasibility(model,raw,tolerance):
    A,b,T,shift,lo,hi=geometry(model);z=np.load(raw,mmap_mode='r')
    assert z.dtype==np.float64 and z.shape==(128,A.shape[1]) and np.isfinite(z).all(),(z.dtype,z.shape)
    maxbound=0.;maxreduced=0.;counts={str(t):0 for t in [0.,1e-7,1e-6]}
    for start in range(0,len(z),32):
        block=np.asarray(z[start:start+32]);flux=block@T.T+shift
        assert np.isfinite(flux).all(),'Nonfinite original-reaction backmap'
        v=np.maximum(np.maximum(lo-flux,flux-hi),0).max(axis=1)
        maxbound=max(maxbound,float(v.max()));maxreduced=max(maxreduced,float(np.maximum(block@A.T-b,0).max()))
        for t in counts:counts[t]+=int(np.count_nonzero(v>float(t)))
    return dict(checked_samples=len(z),all_samples=True,max_original_bound_violation=maxbound,
        max_reduced_Az_minus_b_violation=maxreduced,bound_violation_fraction={t:n/len(z) for t,n in counts.items()},
        declared_tolerance=tolerance,declared_bounds_passed=maxbound<=tolerance,
        no_filtering_or_clipping=True,scope='Short feasibility test, no convergence inference.')

def analyze_trace(sqlite_path):
    c=sqlite3.connect(sqlite_path)
    ranges=c.execute('select n.start,n.end,coalesce(n.text,s.value) from NVTX_EVENTS n left join StringIds s on s.id=n.textId where n.end is not null').fetchall()
    spans=[(a,b,n) for a,b,n in ranges if n and n.startswith(('barrier.','rotation.'))]
    total=[(a,b) for a,b,n in spans if n=='barrier.total'];assert len(total)==1,total
    start,end=total[0]
    kernels=c.execute('select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName').fetchall()
    copies=c.execute('select start,end,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY').fetchall()
    def clipped(items,a,b):return [(max(x,a),min(y,b)) for x,y,*_ in items if y>a and x<b]
    active=clipped(kernels,start,end)
    per=[]
    for a,b,n in spans:
        intervals=clipped(kernels,a,b);transfers=clipped(copies,a,b)
        per.append(dict(name=n,start_ns=a,end_ns=b,wall_seconds=(b-a)/1e9,
            gpu_kernel_sum_seconds=sum(y-x for x,y in intervals)/1e9,
            gpu_kernel_union_seconds=union_seconds(intervals),gpu_memcpy_sum_seconds=sum(y-x for x,y in transfers)/1e9))
    return dict(barrier_nvtx_seconds=(end-start)/1e9,gpu_kernel_sum_seconds=sum(b-a for a,b in active)/1e9,
        gpu_kernel_union_seconds=union_seconds(active),ranges=per,
        kernels=[dict(start_ns=a,end_ns=b,name=n,inside_barrier_overlap_ns=max(0,min(b,end)-max(a,start))) for a,b,n in sorted(kernels)],
        gpu_identity=c.execute('select id,name,busLocation,uuid from TARGET_INFO_GPU').fetchall())

def collect(model,barrier,tolerance,mode,role):
    out=O/model/mode/role
    if (out/'result.json').exists():return read(out/'result.json')
    if out.exists() and any(out.iterdir()):out.rename(out.with_name(role+'-interrupted-'+str(time.time_ns())))
    out.mkdir(parents=True,exist_ok=True)
    assert sha(EXE)==EXPECTED_SHA,'Experimental executable changed after freeze'
    env=os.environ.copy();env['NAJA_BARRIER_TRACE']='1'
    for flag in ['NAJA_BARRIER_GPU_ROTATE','NAJA_BARRIER_VALIDATE_ROTATION']:env.pop(flag,None)
    if mode=='gpu':env['NAJA_BARRIER_GPU_ROTATE']='1'
    if role=='validation':env['NAJA_BARRIER_VALIDATE_ROTATION']='1'
    command=[str(EXE),'sample','run','--model-dir',str(B/'data/models'/model),'--out-root',str(out/'native'),
        '--gpu','0','--n-chains','16','--n-samples','8','--seed','101','--tpb','128','--thinning','1',
        '--constraint-eps','0','--extra-constraint-eps','0','--pair-prob','0','--bounds-policy','ignore',
        '--write-npy','--sample-dtype','float64','--host-prefault-threads','8','--barrier-'+barrier]
    timed=[str(PYTHON),str(B/'scripts/nsight_timed_exec.py'),str(out/'target-wall.json'),*command]
    launch=([str(NSYS),'profile','-t','cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
        '-o',str(out/'trace'),*timed] if role=='profiled' else timed)
    printable=['env','-u','NAJA_BARRIER_GPU_ROTATE','-u','NAJA_BARRIER_VALIDATE_ROTATION','NAJA_BARRIER_TRACE=1']
    if mode=='gpu':printable+=['NAJA_BARRIER_GPU_ROTATE=1']
    if role=='validation':printable+=['NAJA_BARRIER_VALIDATE_ROTATION=1']
    with (O/'commands.sh').open('a') as f:f.write(shlex.join(printable+launch)+'\n')
    dump(O/'progress.json',dict(model=model,mode=mode,role=role,stage='target',job_id=os.getenv('SLURM_JOB_ID')))
    start=time.perf_counter()
    with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        result=subprocess.run(launch,env=env,stdout=stdout,stderr=stderr,timeout=600)
    text=(out/'stdout.log').read_text(errors='replace')+'\n'+(out/'stderr.log').read_text(errors='replace')
    backend=re.findall(r'^.*(?:cuda|cublas|cusolver|curand)\s+(?:runtime\s+)?error\b.*$',text,re.I|re.M)
    assert result.returncode==0 and not backend,(result.returncode,backend,str(out))
    stages=[json.loads(s) for s in re.findall(r'NAJA_BARRIER_STAGE (\{[^\n]+\})',text)]
    algebra=[json.loads(s) for s in re.findall(r'NAJA_ROTATION_VALIDATION (\{[^\n]+\})',text)]
    assert any(s['name']=='barrier.total' for s in stages),str(out)
    assert (role=='validation')==any(s['name']=='rotation.validation_not_for_timing' for s in stages),str(out)
    profile_paths=list((out/'native').glob('*/profile.json'));assert len(profile_paths)==1,profile_paths
    profile=read(profile_paths[0]);config=profile['config']
    assert config['seed']==101 and config['n_chains']==16 and config['n_samples']==8 and config['thinning']==1 and config['tpb_ss']==128
    record=dict(model=model,barrier=barrier,mode=mode,role=role,binary_sha256=EXPECTED_SHA,
        command=command,launched_command=launch,target_wall=read(out/'target-wall.json'),
        outer_command_wall_seconds=time.perf_counter()-start,profile=profile,stages=stages,algebra_validation=algebra,
        run_dir=str(out),exploratory_no_mve=model=='Recon3D_nullspace_only',
        slurm_job_id=os.getenv('SLURM_JOB_ID'),timing_eligible=role!='validation')
    record['feasibility']=feasibility(model,profile_paths[0].parent/'samples.npy',tolerance)
    if role=='validation':
        assert len(algebra)==1 and algebra[0]['relative_frobenius']<=1e-10,algebra
        assert record['feasibility']['declared_bounds_passed'],record['feasibility']
    if role=='profiled':
        export=[str(NSYS),'export','--type','sqlite','--output',str(out/'trace.sqlite'),str(out/'trace.nsys-rep')]
        with (O/'commands.sh').open('a') as f:f.write(shlex.join(export)+'\n')
        with (out/'export.log').open('w') as f:subprocess.run(export,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
        trace=analyze_trace(out/'trace.sqlite');record['trace']=trace
        with (out/'kernel_trace.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(trace['kernels'][0]));w.writeheader();w.writerows(trace['kernels'])
        with (out/'nvtx_stages.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(trace['ranges'][0]));w.writeheader();w.writerows(trace['ranges'])
    dump(out/'result.json',record);print(model,mode,role,record['target_wall']['target_wall_seconds'],flush=True)
    return record

def summarize():
    records=[read(p) for p in O.glob('*/*/*/result.json')]
    comparisons=[]
    for model,barrier,tolerance in MODELS:
        rows={(r['mode'],r['role']):r for r in records if r['model']==model}
        if not all((m,p) in rows for m in ['cpu','gpu'] for p in ['unprofiled','profiled']):continue
        cpu,gpu=rows['cpu','unprofiled'],rows['gpu','unprofiled']
        A,_,_,_,_,_=geometry(model)
        row=dict(model=model,barrier=barrier,m=A.shape[0],d=A.shape[1],rotation_algorithmic_flops=2*A.shape[0]*A.shape[1]**2,
            cpu_barrier_seconds=cpu['profile']['timing']['barrier_rotation_s'],gpu_barrier_seconds=gpu['profile']['timing']['barrier_rotation_s'],
            cpu_rotation_seconds=sum(s['seconds'] for s in cpu['stages'] if s['name']=='barrier.cpu_rotate_A'),
            gpu_rotation_including_copies_seconds=sum(s['seconds'] for s in gpu['stages'] if s['name']=='barrier.gpu_rotate_A'),
            all_four_short_runs_bounds_passed=all(rows[m,p]['feasibility']['declared_bounds_passed'] for m in ['cpu','gpu'] for p in ['unprofiled','profiled']),
            declared_tolerance=tolerance,exploratory_no_mve=model=='Recon3D_nullspace_only')
        row['barrier_speedup']=row['cpu_barrier_seconds']/row['gpu_barrier_seconds']
        row['rotation_speedup_including_copies']=row['cpu_rotation_seconds']/row['gpu_rotation_including_copies_seconds']
        validation=rows.get(('gpu','validation'))
        row['gpu_algebra_validation_passed']=bool(validation['algebra_validation'] and validation['algebra_validation'][0]['relative_frobenius']<=1e-10) if validation else None
        for mode in ['cpu','gpu']:
            baseline,traced=rows[mode,'unprofiled'],rows[mode,'profiled']
            row[mode+'_target_profiler_overhead_pct']=100*(traced['target_wall']['target_wall_seconds']/baseline['target_wall']['target_wall_seconds']-1)
            row[mode+'_profiled_gpu_math_sum_seconds']=traced['trace']['gpu_kernel_sum_seconds']
            row[mode+'_profiled_gpu_math_union_seconds']=traced['trace']['gpu_kernel_union_seconds']
        comparisons.append(row)
    dump(O/'summary.json',dict(binary_sha256=EXPECTED_SHA,runs=records,comparisons=comparisons,
        note='Same executable; GPU flag changes A*Q only. Validation runs excluded from timings. Nested stages must not be added together. GPU math sum uses only kernels inside barrier.total NVTX. Short sample runs do not establish convergence.'))
    if comparisons:
        with (O/'comparison.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)

def main():
    O.mkdir(parents=True,exist_ok=True)
    collect('iJO1366','rotate',1e-7,'gpu','validation');summarize()
    # Obtain the expensive-model attribution first; its full reference follows.
    model,barrier,tolerance=MODELS[2]
    for mode,role in [('cpu','unprofiled'),('gpu','unprofiled'),('gpu','profiled'),('cpu','profiled'),('gpu','validation')]:
        collect(model,barrier,tolerance,mode,role);summarize()
    for model,barrier,tolerance in MODELS[:2]:
        for mode in ['cpu','gpu']:
            for role in ['unprofiled','profiled']:collect(model,barrier,tolerance,mode,role);summarize()
    dump(O/'progress.json',dict(stage='finished',job_id=os.getenv('SLURM_JOB_ID')))

if __name__=='__main__':main()
