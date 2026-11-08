"""Common-mask budget curves and separate actual selected-budget executions."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import numpy as np

B=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(B),str(B/'scripts')]
import metrics
from conditioned_prototype import physical_affinity
from crossmodel_gpu128 import read,write,sha
R=B/'results/crossmodel_extension'
ALLOCATED=tuple(os.sched_getaffinity(0))
SEEDS=(101,202,303)
GRID=(250,500,1000,2000,5000,10000,12000)

def gpu_tpb(model):
    fixed=read(R/'gpu_tpb128/summary.json')['rows']
    members=[row for row in fixed if row['model']==model]
    assert len(members)==3
    baseline=(read(B/'results/precision_quality/iML1515/summary.json')['rows'] if model=='iML1515'
        else read(B/'results/larger_quality/summary.json')['groups'][0]['rows'])
    old=statistics.median(row.get('sampling_phase_seconds',read(Path(row['run_dir'])/'run.json')['profile']['timing']['gpu_sampling_s']) for row in baseline)
    new=statistics.median(row['sampler_seconds'] for row in members)
    return 128 if all(row['quality_and_bounds_met'] for row in members) and new<old else 512

def source(model,method,seed):
    if method=='cpu':
        count=12000 if model=='iML1515' else 10000
        return Path(read(R/f'{model}_n{count}_seed{seed}.json')['run_directory'])
    return R/'gpu_tpb128'/model/f'seed{seed}'

def select():
    choices=[]
    for model in ('iML1515','iAF1260'):
        modeldir=R/'cached_models'/model
        for method in ('cpu','gpu'):
            curves=[]
            for seed in SEEDS:
                src=source(model,method,seed);run=read(src/'run.json');draws=run['draws']
                out=R/'budget_curves'/model/method/f'seed{seed}'
                out.mkdir(parents=True,exist_ok=True)
                if not (out/'metrics.json').exists():
                    command=[str(B/'env-cpu/bin/python'),str(B/'metrics.py'),'--samples',str(src/'samples.npy'),
                        '--model-dir',str(modeldir),'--output',str(out),'--prefixes',*[str(n) for n in GRID if n<=draws]]
                    with (out/'metrics.log').open('w') as log:
                        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                m=read(out/'metrics.json')
                assert m['sample_bound_violation_fraction']==0 and m['bounds_tolerance']==1e-7
                curves.append(dict(seed=seed,source=str(src),curve=m['prefix_quality'],metrics=str(out/'metrics.json')))
            candidates=sorted(set.intersection(*[{point['draws_per_chain'] for point in curve['curve']} for curve in curves]))
            valid=[n for n in candidates if all(all(point['quality_met'] for point in curve['curve'] if point['draws_per_chain']>=n) for curve in curves)]
            choice=dict(model=model,method=method,status='selected' if valid else 'no_stable_passing_budget',
                selected_draws=min(valid) if valid else None,tpb=gpu_tpb(model) if method=='gpu' else None,
                curves=curves,selection='Earliest evaluated budget passing in all3seeds with no observed later failure; common exact-cached-target mask. Fresh executions supply times, never prefix interpolation.')
            choices.append(choice)
            write(R/'budget_selection.json',dict(choices=choices,generated_utc=datetime.now(timezone.utc).isoformat()))
    return choices

def actual(choice,seed):
    model,method,draws=choice['model'],choice['method'],choice['selected_draws']
    if draws is None:return
    name=f'cpu_selected_cached_n{draws}' if method=='cpu' else f"gpu_f64_tpb{choice['tpb']}_selected_n{draws}"
    out=R/'selected_budget_runs'/model/name/f'seed{seed}'
    if (out/'quality.json').exists():return
    out.parent.mkdir(parents=True,exist_ok=True)
    modeldir=R/'cached_models'/model
    if method=='cpu':
        os.sched_setaffinity(0,ALLOCATED);affinity=physical_affinity()
        os.environ['HOPSY_CXXFLAGS']='-O3 -march=native -mtune=native'
        os.environ['HOPSY_BUILD_MANIFEST']=str(B/'audit/hopsy_native_build.json')
        command=[str(B/'env-cpu-native/bin/python'),str(B/'scripts/timed_cpu_baseline.py'),'sample',
            '--model',str(modeldir),'--out',str(out),'--seed',str(seed),'--chains','16',
            '--draws',str(draws),'--warmup','1000','--thinning','2000','--workers','16',
            '--chunk',str(draws+1000),'--build-variant','native']
    else:
        affinity=None
        command=[str(B/'scripts/enter.sh'),str(B/'env-cpu/bin/python'),str(B/'naja_runner.py'),
            '--exe',str(B/'build-h100-f64/naja'),'--model',str(modeldir),'--out',str(out),
            '--seed',str(seed),'--chains','16','--draws',str(draws),'--warmup','1000',
            '--tpb',str(choice['tpb']),'--thinning','1000','--barrier','rotate',
            '--pair-prob','0','--sample-dtype','float64','--timeout','900']
    if not (out/'run.json').exists():
        with (out.parent/f'seed{seed}.sampling.log').open('w') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
    run=read(out/'run.json');assert run['status']=='complete'
    if affinity:write(out/'physical_affinity.json',affinity)
    if not (out/'metrics.json').exists():
        os.sched_setaffinity(0,ALLOCATED)
        command_metrics=[str(B/'env-cpu/bin/python'),str(B/'metrics.py'),'--samples',str(out/'samples.npy'),
            '--model-dir',str(modeldir),'--output',str(out),'--prefixes',str(draws)]
        with (out/'metrics.log').open('w') as log:subprocess.run(command_metrics,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    m=read(out/'metrics.json')
    if method=='cpu':
        t=read(out/'sampling_math_timing.json');compute=t['worker_draw_critical_path_seconds']
    else:compute=run['profile']['timing']['gpu_sampling_s']
    current=np.load(out/'samples.npy',mmap_mode='r');original=np.load(source(model,method,seed)/'samples.npy',mmap_mode='r')
    parity=all(np.array_equal(current[:,lo:lo+256],original[:,lo:min(lo+256,draws)]) for lo in range(0,draws,256))
    if method=='cpu':assert parity,'Same-seed CPU prefix parity failed'
    row=dict(model=model,method=name,seed=seed,draws=draws,compute_seconds=compute,
        quality_and_bounds_met=bool(m['split_rhat_median']<=1.01 and m['bulk_ess_median']>=100 and m['sample_bound_violation_fraction']==0 and m['bounds_tolerance']==1e-7),
        split_rhat_median=m['split_rhat_median'],bulk_ess_median=m['bulk_ess_median'],
        sample_bound_violation_fraction=m['sample_bound_violation_fraction'],run_directory=str(out),
        source_prefix_bitwise_equal=parity,source_prefix=str(source(model,method,seed)),
        geometry_sha256=sha(modeldir/'rounding/polytope.npz'),certificate_sha256=sha(modeldir/'diagnostic_mask.json'),
        command=command,job_id=os.getenv('SLURM_JOB_ID'),selection=str(R/'budget_selection.json'))
    write(out/'quality.json',row);print(json.dumps(row),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('cpu_select','gpu_retime'),required=True);args=p.parse_args()
    if args.stage=='cpu_select':
        choices=select()
        for choice in choices:
            if choice['method']=='cpu':
                for seed in SEEDS:actual(choice,seed)
        if datetime.now(timezone.utc)<datetime.fromisoformat('2026-11-08T21:15:00+00:00'):
            job=subprocess.check_output(['sbatch','--parsable',str(B/'scripts/crossmodel_budget_gpu.sbatch')],text=True).strip()
            write(R/'selected_gpu_job.json',dict(job=job,submitted_utc=datetime.now(timezone.utc).isoformat()))
    else:
        for choice in read(R/'budget_selection.json')['choices']:
            if choice['method']=='gpu':
                for seed in SEEDS:actual(choice,seed)

if __name__=='__main__':main()
