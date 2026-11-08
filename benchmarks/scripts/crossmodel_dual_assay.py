"""Report the original-model and exact-cached-target aggregation policies side by side."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import numpy as np
from crossmodel_gpu128 import B,R,read,write,sha
sys.path.insert(0,str(B))
import metrics

def original_assay(source,model,output):
    source,model,output=map(Path,(source,model,output))
    original=read(source/'metrics.json');certificate=read(model/'diagnostic_mask.json')
    frozen='854e524cade5cecc6a9c5541cb15957e0336c3461434c2363d9ef2a2b701947c'
    assert original['metrics_code_sha256']==sha(B/'metrics.py')==frozen
    assert certificate['geometry_sha256']==sha(model/'rounding/polytope.npz')==original['diagnostic_mask_geometry_sha256']
    assert certificate.get('target_geometry_kind')!='exact_cached_polytope'
    source_model=Path(original['model_dir'])
    for name in ('l_bounds.csv','u_bounds.csv','reaction_ids.txt'):
        assert sha(source_model/'gem'/name)==sha(model/'gem'/name)
    with (source/'reaction_metrics.csv').open() as handle:
        reader=csv.DictReader(handle);fields=reader.fieldnames;rows=list(reader)
    ids=(model/'gem/reaction_ids.txt').read_text().splitlines()
    assert ids==[row['reaction'] for row in rows]
    active=np.load(model/'active_reactions.npy')
    extent=np.array([float(row['empirical_range']) for row in rows]);active=active & (extent!=0)
    rhat=np.array([float(row['split_rhat']) for row in rows]);ess=np.array([float(row['bulk_ess']) for row in rows])
    assert not np.any(np.isnan(rhat[extent!=0]))
    summary=metrics.scalar_summary(rhat[active],ess[active])
    passed=bool(summary['split_rhat_median']<=1.01 and summary['bulk_ess_median']>=100
        and original['sample_bound_violation_fraction']==0 and original['bounds_tolerance']==1e-7)
    value=dict(summary,quality_and_bounds_met=passed,
        sample_bound_violation_fraction=original['sample_bound_violation_fraction'],
        max_absolute_bound_violation=original['max_absolute_bound_violation'],
        bounds_tolerance=original['bounds_tolerance'],
        scope='Original COBRA-model LP range mask; no claim that excluded ranges are certified on the finite-precision cached target.',
        source_metrics=str(source/'metrics.json'),source_metrics_sha256=sha(source/'metrics.json'),
        source_reaction_csv_sha256=sha(source/'reaction_metrics.csv'),
        source_mask_sha256=sha(model/'active_reactions.npy'),source_policy_sha256=sha(model/'diagnostic_mask.json'),
        code_sha256=sha(Path(__file__)),geometry_sha256=certificate['geometry_sha256'],
        full_budget_only=True,no_interpolated_times=True,unmasked_bound_checks_unchanged=True)
    if Path(original['model_dir']).resolve()==model.resolve():
        for key in ('split_rhat_median','split_rhat_p95','bulk_ess_median','diagnosed_reactions'):
            assert value[key]==original[key],('historical summary validation',key,value[key],original[key])
        value['reproduces_historical_primary_summary_exactly']=True
    output.mkdir(parents=True,exist_ok=True);write(output/'assay.json',value)
    return value

def distinct(groups):
    output={}
    for group in groups:
        run=read(Path(group['rows'][0]['run'])/'run.json')
        if group['method'].startswith('gpu'):
            cfg=run['profile']['config']
            identity=('gpu',run['chains'],run['draws'],run['warmup'],cfg['thinning'],cfg['tpb_ss'],cfg.get('sample_dtype','float32'))
        else:identity=('cpu',run['chains'],run['draws'],run['warmup'],run['thinning'])
        if identity not in output or '_selected_' in group['method']:output[identity]=group
    return list(output.values())

def main():
    cached=read(R/'cached_assay_summary.json');groups=[]
    validation=[]
    for group in cached['groups']:
        rows=[]
        for item in group['rows']:
            out=R/'original_model_assay'/item['model']/item['method']/f"seed{item['seed']}"
            m=original_assay(item['run'],B/'data/models'/item['model'],out)
            rows.append(dict(item,**{k:m[k] for k in ('quality_and_bounds_met','split_rhat_median','bulk_ess_median','sample_bound_violation_fraction')},
                metrics=str(out/'assay.json'),ess_per_compute_second=m['bulk_ess_median']/item['compute_seconds']))
            if m.get('reproduces_historical_primary_summary_exactly'):validation.append(str(out/'assay.json'))
        complete=len(rows)==3 and all(row['quality_and_bounds_met'] for row in rows)
        value=dict(model=group['model'],method=group['method'],rows=rows,passing_seeds=sum(row['quality_and_bounds_met'] for row in rows),tested_seeds=len(rows),complete_three_seed_group=complete)
        if complete:value.update(median_compute_seconds=statistics.median(row['compute_seconds'] for row in rows),median_ess_per_compute_second=statistics.median(row['ess_per_compute_second'] for row in rows))
        groups.append(value)
    comparisons=[]
    for model in ('iML1515','iAF1260'):
        eligible=distinct([g for g in groups if g['model']==model and g['complete_three_seed_group']])
        cpus=[g for g in eligible if g['method'].startswith('cpu')];gpus=[g for g in eligible if g['method'].startswith('gpu')]
        if cpus and gpus:
            cpu=min(cpus,key=lambda g:g['median_compute_seconds']);gpu=min(gpus,key=lambda g:g['median_compute_seconds'])
            comparisons.append(dict(model=model,cpu_method=cpu['method'],gpu_method=gpu['method'],
                cpu_compute_median_seconds=cpu['median_compute_seconds'],gpu_compute_median_seconds=gpu['median_compute_seconds'],
                cpu_over_gpu_time_ratio=cpu['median_compute_seconds']/gpu['median_compute_seconds'],
                paired_gpu_over_cpu_ess_rate=statistics.median(gr['ess_per_compute_second']/cr['ess_per_compute_second'] for cr,gr in zip(cpu['rows'],gpu['rows']))))
    result=dict(generated_utc=datetime.now(timezone.utc).isoformat(),original_model_assay=dict(groups=groups,comparisons=comparisons),
        cached_target_assay=cached,
        interpretation='Both use the same saved trajectories and unmasked original-bound checks. Conservative LP exclusions on cached A/b do not make a median-Rhat criterion uniformly harder: adding unresolved tiny reactions can lower the median. Budgets and comparisons remain assay-specific.',
        historical_summary_reproduction=dict(status='passed',case_count=len(validation),sources=validation))
    write(R/'dual_assay_summary.json',result)
    lines=['# Cross-model results under both diagnostic aggregation policies','',result['interpretation'],'',
        '| Model | Method | Original-model mask pass/tested | Cached-target mask pass/tested |','|---|---|---:|---:|']
    for raw,conservative in zip(groups,cached['groups']):
        lines.append(f"| {raw['model']} | {raw['method']} | {raw['passing_seeds']}/{raw['tested_seeds']} | {conservative['passing_seeds']}/{conservative['tested_seeds']} |")
    lines+=['','| Assay | Model | Tested CPU | Tested GPU | CPU/GPU time | GPU/CPU ESS/s |','|---|---|---|---|---:|---:|']
    for assay,comp in [('Original-model LP',comparisons),('Cached-target LP',cached['comparisons'])]:
        for row in comp:
            lines.append(f"| {assay} | {row['model']} | {row['cpu_method']} | {row['gpu_method']} | {row['cpu_over_gpu_time_ratio']:.4f} | {row['paired_gpu_over_cpu_ess_rate']:.4f} |")
    lines+=['','These are finite tested configurations, with fresh selected-budget timings preferred over an earlier timing at an identical configuration. '
        'Neither assay changes sample arrays, feasibility tolerance, or sampler math. Original-model LP exclusions and cached-target LP exclusions have different mathematical domains; neither table is silently substituted for the other.','']
    (R/'DUAL_ASSAY_RESULTS.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
