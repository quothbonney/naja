"""Predeclared TPB128 follow-up at unchanged retained/transition budgets."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B))
import metrics
R=B/'results/crossmodel_extension'

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def main():
    proof=read(B/'results/f64-build-verification/verification.json')
    binary=B/'build-h100-f64/naja'
    assert proof['gpu_correctness_tests_passed']==5 and sha(binary)==proof['binary_sha256']
    rows=[]
    for model,draws in [('iML1515',2000),('iAF1260',5000)]:
        modeldir=R/'cached_models'/model
        for seed in (101,202,303):
            out=R/'gpu_tpb128'/model/f'seed{seed}'
            command=[str(B/'env-cpu/bin/python'),str(B/'naja_runner.py'),'--exe',str(binary),
                '--model',str(modeldir),'--out',str(out),'--seed',str(seed),'--chains','16',
                '--draws',str(draws),'--warmup','1000','--tpb','128','--thinning','1000',
                '--barrier','rotate','--pair-prob','0','--sample-dtype','float64','--timeout','600']
            if not (out/'run.json').exists():
                out.parent.mkdir(parents=True,exist_ok=True)
                with (out.parent/f'seed{seed}.sampling.log').open('w') as log:
                    subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=660)
            run=read(out/'run.json')
            assert run['status']=='complete' and run['exe_sha256']==proof['binary_sha256']
            assert run['profile']['config']['tpb_ss']==128 and run['profile']['config']['thinning']==1000
            if not (out/'metrics.json').exists():
                metrics.evaluate(out/'samples.npy',modeldir,out,prefixes=[draws])
            m=read(out/'metrics.json')
            historical=(B/'results/precision_quality/iML1515'/f'seed{seed}' if model=='iML1515'
                else Path(read(B/'results/larger_quality/iAF1260/gpu'/f'seed{seed}'/'quality.json')['measured']['run_dir']))
            current=np.load(out/'samples.npy',mmap_mode='r');old=np.load(historical/'samples.npy',mmap_mode='r')
            assert current.shape==old.shape
            mismatches=0;maxdiff=0.
            for lo in range(0,draws,256):
                a=np.asarray(current[:,lo:lo+256]).astype(old.dtype);b=np.asarray(old[:,lo:lo+256])
                mismatches+=int(np.count_nonzero(a!=b));maxdiff=max(maxdiff,float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))))
            passed=bool(m['split_rhat_median']<=1.01 and m['bulk_ess_median']>=100
                and m['sample_bound_violation_fraction']==0 and m['bounds_tolerance']==1e-7)
            row=dict(model=model,seed=seed,method='gpu_f64_tpb128',draws=draws,chains=16,warmup=1000,
                thinning=1000,tpb=128,sample_dtype='float64',quality_and_bounds_met=passed,
                split_rhat_median=m['split_rhat_median'],bulk_ess_median=m['bulk_ess_median'],
                sample_bound_violation_fraction=m['sample_bound_violation_fraction'],
                sampler_seconds=run['profile']['timing']['gpu_sampling_s'],
                secondary_wall_seconds=run['wall_including_normalization_seconds'],
                binary_sha256=proof['binary_sha256'],geometry_sha256=sha(modeldir/'rounding/polytope.npz'),
                certificate_sha256=sha(modeldir/'diagnostic_mask.json'),
                historical_tpb512_run=str(historical),historical_output_dtype=str(old.dtype),
                historical_dtype_cast_mismatches=mismatches,historical_dtype_cast_maxdiff=maxdiff,
                run_directory=str(out),command=command,job_id=os.getenv('SLURM_JOB_ID'))
            write(out/'quality.json',row);rows.append(row)
            write(R/'gpu_tpb128/summary.json',dict(rows=rows,
                protocol='Predeclared TPB128 at unchanged sampling budget; float64 verified binary. Common exact-cached-target masks, strict unfiltered bounds. AF historical TPB512 export was float32, so its comparison also changes export precision.'))
            print(json.dumps(row),flush=True)

if __name__=='__main__':main()
