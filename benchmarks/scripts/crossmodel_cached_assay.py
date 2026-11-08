"""Reassay saved arrays under separate exact-cached-target diagnostic masks."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from reaggregate_cached_metrics import reaggregate

B = Path(__file__).resolve().parents[1]
R = B/'results/crossmodel_extension'
SEEDS = (101,202,303)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def write(path, value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def tasks():
    rows=[]
    for model in ('iML1515','iAF1260'):
        modeldir=R/'cached_models'/model
        certificate=read(modeldir/'diagnostic_mask.json')
        assert certificate['target_geometry_kind']=='exact_cached_polytope'
        assert certificate['geometry_sha256']==sha(modeldir/'rounding/polytope.npz')
        for seed in SEEDS:
            if model=='iML1515':
                gpu=read(B/'results/precision_quality'/model/f'seed{seed}'/'quality.json')
                rows.append(dict(model=model,method='gpu_f64',seed=seed,run=Path(gpu['run_dir']),
                    compute_seconds=gpu['sampling_phase_seconds'],draws=gpu['draws_per_chain']))
                for p in sorted(R.glob(f'{model}_n*_seed{seed}.json')):
                    item=read(p)
                    rows.append(dict(model=model,method=f"cpu_thin2000_n{item['draws']}",seed=seed,
                        run=Path(item['run_directory']),compute_seconds=item['worker_draw_critical_path_seconds'],draws=item['draws']))
                worker=B/'results/native_cpu'/model/'physical16_mathinstrumented_native_jade'/f'seed{seed}'
                timing=read(worker/'sampling_math_timing.json');run=read(worker/'run.json')
                rows.append(dict(model=model,method='cpu_guideline',seed=seed,run=worker,
                    compute_seconds=timing['worker_draw_critical_path_seconds'],draws=run['draws']))
            else:
                gpu=read(B/'results/larger_quality'/model/'gpu'/f'seed{seed}'/'quality.json')['measured']
                run=read(Path(gpu['run_dir'])/'run.json')
                rows.append(dict(model=model,method='gpu',seed=seed,run=Path(gpu['run_dir']),
                    compute_seconds=run['profile']['timing']['gpu_sampling_s'],draws=gpu['draws_per_chain']))
                item=read(R/f'{model}_n10000_seed{seed}.json')
                rows.append(dict(model=model,method='cpu_thin2000_n10000',seed=seed,
                    run=Path(item['run_directory']),compute_seconds=item['worker_draw_critical_path_seconds'],draws=item['draws']))
    tuning=R/'gpu_tpb128/summary.json'
    if tuning.exists():
        for item in read(tuning)['rows']:
            rows.append(dict(model=item['model'],method='gpu_f64_tpb128',seed=item['seed'],
                run=Path(item['run_directory']),compute_seconds=item['sampler_seconds'],draws=item['draws']))
    for path in sorted((R/'selected_budget_runs').glob('*/*/seed*/quality.json')):
        item=read(path)
        rows.append(dict(model=item['model'],method=item['method'],seed=item['seed'],
            run=Path(item['run_directory']),compute_seconds=item['compute_seconds'],draws=item['draws']))
    return rows

def one(task):
    modeldir=R/'cached_models'/task['model'];out=R/'cached_assay'/task['model']/task['method']/f"seed{task['seed']}"
    out.mkdir(parents=True,exist_ok=True)
    command=[str(B/'env-cpu/bin/python'),str(B/'scripts/reaggregate_cached_metrics.py'),
        '--source',str(task['run']),'--model',str(modeldir),'--output',str(out)]
    if not (out/'metrics.json').exists():
        reaggregate(task['run'],modeldir,out)
    m=read(out/'metrics.json');certificate=read(modeldir/'diagnostic_mask.json')
    assert m['schema_version']==2 and m['diagnostic_mask_status']=='model_lp_verified'
    assert m['diagnostic_mask_geometry_sha256']==certificate['geometry_sha256']
    assert m['metrics_code_sha256']==sha(B/'metrics.py')
    passed=bool(m['split_rhat_median']<=1.01 and m['bulk_ess_median']>=100
        and m['sample_bound_violation_fraction']==0 and m['bounds_tolerance']==1e-7)
    row=dict(task,run=str(task['run']),status='complete',quality_and_bounds_met=passed,
        split_rhat_median=m['split_rhat_median'],bulk_ess_median=m['bulk_ess_median'],
        sample_bound_violation_fraction=m['sample_bound_violation_fraction'],
        maximum_bound_violation=m['max_absolute_bound_violation'],
        ess_per_compute_second=m['bulk_ess_median']/task['compute_seconds'],
        geometry_sha256=certificate['geometry_sha256'],certificate_sha256=sha(modeldir/'diagnostic_mask.json'),
        active_reactions=certificate['active_reactions'],excluded_reactions=certificate['excluded_nearconstant_reactions'],
        metrics=str(out/'metrics.json'),command=command)
    write(out/'assay.json',row)
    return row

def summarize(rows):
    groups=[];comparisons=[]
    for model in ('iML1515','iAF1260'):
        for method in sorted({row['method'] for row in rows if row['model']==model}):
            members=sorted([row for row in rows if row['model']==model and row['method']==method],key=lambda row:row['seed'])
            group=dict(model=model,method=method,passing_seeds=sum(row['quality_and_bounds_met'] for row in members),
                tested_seeds=len(members),rows=members,
                complete_three_seed_group=len(members)==3 and all(row['quality_and_bounds_met'] for row in members))
            if group['complete_three_seed_group']:
                group.update(median_compute_seconds=statistics.median(row['compute_seconds'] for row in members),
                    median_ess_per_compute_second=statistics.median(row['ess_per_compute_second'] for row in members))
            groups.append(group)
        eligible=[group for group in groups if group['model']==model and group['complete_three_seed_group']]
        distinct={}
        for group in eligible:
            run=read(Path(group['rows'][0]['run'])/'run.json')
            if group['method'].startswith('gpu'):
                cfg=run['profile']['config']
                identity=('gpu',run['chains'],run['draws'],run['warmup'],cfg['thinning'],cfg['tpb_ss'],cfg.get('sample_dtype','float32'))
            else:
                identity=('cpu',run['chains'],run['draws'],run['warmup'],run['thinning'])
            if identity not in distinct or '_selected_' in group['method']:
                distinct[identity]=group
        candidates=list(distinct.values())
        gpus=[group for group in candidates if group['method'].startswith('gpu')]
        gpu=min(gpus,key=lambda group:group['median_compute_seconds']) if gpus else None
        cpus=[group for group in candidates if group['method'].startswith('cpu')]
        if gpu and cpus:
            cpu=min(cpus,key=lambda group:group['median_compute_seconds'])
            comparisons.append(dict(model=model,cpu_method=cpu['method'],gpu_method=gpu['method'],
                cpu_compute_median_seconds=cpu['median_compute_seconds'],gpu_compute_median_seconds=gpu['median_compute_seconds'],
                cpu_over_gpu_time_ratio=cpu['median_compute_seconds']/gpu['median_compute_seconds'],
                paired_gpu_over_cpu_ess_rate=statistics.median(gr['ess_per_compute_second']/cr['ess_per_compute_second']
                    for cr,gr in zip(cpu['rows'],gpu['rows'])),
                scope='Fastest complete CPU/GPU groups among finite tested configurations, same exact-cached-target mask. A fresh selected-budget retime supersedes an earlier timing at the identical configuration; earlier evidence remains visible.'))
    value=dict(generated_utc=datetime.now(timezone.utc).isoformat(),groups=groups,comparisons=comparisons,
        assay='Exact cached A/b LP range certificates; unresolved remain active; samples and sampler timings unchanged. Every original bound independently checked at1e-7.',
        timer_scope='CPU worker draw critical path includes bindings/statecopy/descheduling; GPU sampler includes allocation/slack/RNG initialization. Both exclude download/write; not identical pure-arithmetic counters.')
    write(R/'cached_assay_summary.json',value)
    lines=['# Cross-model conservative cached-target reassay','',value['assay'],'',value['timer_scope'],'',
        '| Model | Method | Passing / tested seeds | Median sampler compute proxy s |',
        '|---|---|---:|---:|']
    for group in groups:
        timing=f"{group['median_compute_seconds']:.4f}" if 'median_compute_seconds' in group else '—'
        lines.append(f"| {group['model']} | {group['method']} | {group['passing_seeds']}/{group['tested_seeds']} | {timing} |")
    lines+=['','| Model | Selected tested CPU | CPU/GPU time | GPU/CPU ESS/s |','|---|---|---:|---:|']
    for row in comparisons:
        lines.append(f"| {row['model']} | {row['cpu_method']} | {row['cpu_over_gpu_time_ratio']:.4f} | {row['paired_gpu_over_cpu_ess_rate']:.4f} |")
    lines+=['','Finite tested CPU/GPU configurations only; preserved failed groups remain visible. Original raw-model-mask results are separate.','']
    (R/'CACHED_TARGET_RESULTS.md').write_text('\n'.join(lines))
    if rows:
        with (R/'cached_assay_table.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    proof=read(R/'reaggregate_validation/verification.json')
    assert proof['status']=='passed' and len(proof['cases'])==2
    assert proof['reaggregation_code_sha256']==sha(B/'scripts/reaggregate_cached_metrics.py')
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(one,tasks()):
            rows.append(row);summarize(rows)
    summarize(rows)

if __name__=='__main__':
    main()
