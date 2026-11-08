#!/usr/bin/env python3
"""Summarize measured qualifying runs; never use linearly estimated times."""
import csv
import json
from pathlib import Path
import statistics

B=Path(__file__).resolve().parents[1]
root=B/'results/quality'
rows=[];records=[];tuning=[]
for path in sorted(root.glob('*/*/seed*/quality.json')):
    r=json.loads(path.read_text());records.append(r)
    unique={attempt['run_dir']:attempt for attempt in r.get('attempts',[]) if attempt.get('role') in ['extension_budget','budget_finding_pilot']}
    source_wall=None
    source=r.get('source_run')
    if source and (Path(source)/'run.json').exists():
        source_record=json.loads((Path(source)/'run.json').read_text())
        source_wall=source_record.get('wall_including_normalization_seconds',source_record.get('process_wall_seconds'))
    search_seconds=sum(attempt['wall_seconds'] for attempt in unique.values())
    tuning.append(dict(model=r['model'],variant=r['variant'],seed=r['seed'],status=r['status'],
        newly_executed_search_sampling_wall_seconds=search_seconds,reused_canonical_source_trajectory_seconds=source_wall,
        evaluated_new_search_budgets=[dict(draws_per_chain=attempt['draws_per_chain'],quality_met=attempt['quality_met'],
            split_rhat_median=attempt['split_rhat_median'],bulk_ess_median=attempt['bulk_ess_median'],wall_seconds=attempt['wall_seconds']) for attempt in unique.values()]))
    m=r.get('measured')
    if r.get('status')!='complete' or not m or not m['quality_met']: continue
    final_metrics=json.loads((Path(m['run_dir'])/'metrics.json').read_text())
    final_run=json.loads((Path(m['run_dir'])/'run.json').read_text())
    rows.append(dict(model=r['model'],variant=r['variant'],seed=r['seed'],
        draws_per_chain=m['draws_per_chain'],warmup=1000,chains=16,
        cpu_build_variant=final_run.get('build_variant'),cpu_module_sha256=final_run.get('hopsy_module_sha256'),
        cpu_cxxflags=final_run.get('hopsy_cxxflags'),
        measured_process_wall_seconds=m['wall_seconds'],new_search_sampling_seconds=search_seconds,reused_source_search_seconds=source_wall,split_rhat_median=m['split_rhat_median'],
        bulk_ess_median=m['bulk_ess_median'],split_rhat_p95=final_metrics['split_rhat_p95'],sample_bound_violation_fraction=final_metrics['sample_bound_violation_fraction'],bounds_tolerance=final_metrics['bounds_tolerance'],thinning=m['thinning'],run_dir=m['run_dir']))
if rows:
    with (root/'measured.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
groups=[];speedups=[]
for model in sorted({r['model'] for r in rows}):
    for variant in sorted({r['variant'] for r in rows if r['model']==model}):
        found=[r for r in rows if r['model']==model and r['variant']==variant]
        groups.append(dict(model=model,variant=variant,seeds=[r['seed'] for r in found],
            measured_seed_count=len(found),complete_three_seed_group=len(found)==3,
            median_measured_seconds=statistics.median(r['measured_process_wall_seconds'] for r in found),
            median_budget=statistics.median(r['draws_per_chain'] for r in found)))
    for seed in [101,202,303]:
        cpu=[r for r in rows if r['model']==model and r['seed']==seed and r['variant'].startswith('cpu_')]
        if not cpu: continue
        best=min(cpu,key=lambda r:r['measured_process_wall_seconds'])
        for gpu in [r for r in rows if r['model']==model and r['seed']==seed and r['variant'].startswith('naja')]:
            speedups.append(dict(model=model,seed=seed,gpu_variant=gpu['variant'],
                fastest_measured_cpu_variant=best['variant'],
                cpu_seconds=best['measured_process_wall_seconds'],gpu_seconds=gpu['measured_process_wall_seconds'],
                speedup_cpu_over_gpu=best['measured_process_wall_seconds']/gpu['measured_process_wall_seconds']))
summary={'criterion':'masked median classical split-Rhat<=1.01 and median pooled rank-normalized bulkESS>=100',
    'protocol':'16 chains, discard 1000 retained warmup draws; first evaluated passing prefix chooses exact measured budget',
    'warmup_caveat':'Warmup move count differs across thinning settings; fixed 1000 retained draws per chain.',
    'scope':'Finite predeclared budget grid and tested CPU variants only; not globally optimal time-to-quality.',
    'timing':'Measured sampling process wall including startup/load/warmup/write and Naja normalization; excludes diagnostic and exploratory search cost.',
    'selection_caveat':'Fastest CPU selected per seed from measured passing variants; report included variants and incomplete candidates.',
    'planned_seed_variant_tasks':81,'recorded_seed_variant_tasks':len(records),
    'candidate_search_complete':len(records)==81 and all(r['status'] not in ['running'] for r in records),
    'tuning_candidates':tuning,'measured_rows':rows,'groups':groups,'per_seed_speedups':speedups,
    'all_records_status':[dict(model=r['model'],variant=r['variant'],seed=r['seed'],status=r['status']) for r in records]}
(root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({'measured_rows':len(rows),'groups':groups},indent=2))
