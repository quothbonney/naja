#!/usr/bin/env python3
"""Render saved same-target convergence curves; never interpolate sampling time."""
import argparse,csv,json,statistics
from pathlib import Path
from datetime import datetime,timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--condition-root',type=Path,required=True);a=p.parse_args();root=a.condition_root.resolve();out=root/'quality';out.mkdir(exist_ok=True)
variants=('global_cpu','stale_cpu','naja_rotate');colors={'global_cpu':'#0072B2','stale_cpu':'#D55E00','naja_rotate':'#009E73'}
labels={'global_cpu':'Full global PolyRound + native CPU','stale_cpu':'Inherited rounding CPU','naja_rotate':'Inherited + Naja rotation'}
curves={};states=[]
# Early output: scientifically valid shared-target arrays already exist. The
# independently reduced582-D raw CPU comparator is deliberately never imported.
compilation=root/'quality/mask_compilation.json'
legacy_compatible=not compilation.exists() or json.loads(compilation.read_text()).get('scientific_mask_unchanged',False)
for variant in (('stale_cpu','naja_rotate') if legacy_compatible else ()):
 for seed in (101,202,303):
  source=root/'runs'/variant/f'seed{seed}'
  paths=[x for x in source.glob('budget*/metrics.json') if json.loads(x.read_text()).get('schema_version')==2]
  if not paths:continue
  path=max(paths,key=lambda x:json.loads(x.read_text())['shape'][1]);m=json.loads(path.read_text())
  for x in m['prefix_quality']:
   n=x['draws_per_chain']
   if n not in (250,500,1000,2000,5000,10000,20000,50000,200000):continue
   curves[(variant,seed,n)]=dict(condition_id=root.name,variant=variant,seed=seed,draws_per_chain=n,
    split_rhat_median=x['split_rhat_median'],split_rhat_p95=x['split_rhat_p95'],bulk_ess_median=x['bulk_ess_median'],
    sample_bound_violation_fraction=m['sample_bound_violation_fraction'] if m['sample_bound_violation_fraction']==0 else None,
    diagnostic_source=str(path),actual_execution_at_this_budget=False,measured_sampler_seconds=None,
    note='Frozen diagnostics on a saved shared-target trajectory prefix. No time assigned to prefixes.')
for path in sorted(out.glob('*/seed*/curve.json')):
 for r in json.loads(path.read_text()):curves[(r['variant'],r['seed'],r['draws_per_chain'])]=r
for path in sorted(out.glob('*/seed*/quality.json')):
 row=json.loads(path.read_text());row['state_file']=str(path);states.append(row)
rows=sorted(curves.values(),key=lambda r:(r['variant'],r['seed'],r['draws_per_chain']))
for r in rows:
 thin=r.get('actual_thinning',1000 if r['variant']=='naja_rotate' else 2000)
 r.update(actual_thinning=thin,retained_transitions_per_chain=r['draws_per_chain']*thin,total_transitions_per_chain=(r['draws_per_chain']+1000)*thin)
def csvwrite(name,values):
 keys=sorted(set().union(*(r.keys() for r in values))) if values else ['condition_id','variant','seed','status']
 with (out/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(values)
csvwrite('rhat_curve.csv',rows)
completed=[r for r in states if r.get('status')=='complete'];csvwrite('time_to_target.csv',completed)
fig,ax=plt.subplots(figsize=(8.5,4.6))
for variant in variants:
 first=True
 for seed in (101,202,303):
  data=[r for r in rows if r['variant']==variant and r['seed']==seed and r.get('split_rhat_median') is not None]
  if not data:continue
  ax.plot([r['draws_per_chain'] for r in data],[r['split_rhat_median'] for r in data],'-o',color=colors[variant],alpha=.65,markersize=4,
          label=labels[variant] if first else None);first=False
ax.axhline(1.05,color='#555555',linestyle='--',linewidth=1,label='Declared Rhat threshold1.05')
ax.set_xscale('log');ax.set_xlabel('Actual retained draws per chain');ax.set_ylabel('Masked median classical split-Rhat')
ax.set_title('Same cached conditional target: convergence by chain length');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.text(.1,.015,'Each line is a seed. Prefix diagnostics are not fresh timing runs. ESS≥100 and bounds/mass balance≤1e-7 are separate gates.',fontsize=8)
fig.tight_layout(rect=(0,.045,1,1))
for ext in ('svg','pdf','png'):fig.savefig(out/f'rhat_vs_chain_length.{ext}',dpi=160)
plt.close(fig)
fig,ax=plt.subplots(figsize=(8.5,4.6))
for variant in variants:
 first=True
 for seed in (101,202,303):
  data=[r for r in rows if r['variant']==variant and r['seed']==seed and r.get('split_rhat_median') is not None]
  if not data:continue
  ax.plot([r['total_transitions_per_chain'] for r in data],[r['split_rhat_median'] for r in data],'-o',color=colors[variant],alpha=.65,markersize=4,label=labels[variant] if first else None);first=False
ax.axhline(1.05,color='#555555',linestyle='--',linewidth=1,label='Rhat threshold1.05');ax.set_xscale('log');ax.set_xlabel('Actual transitions per chain, including1000 retained warmup draws');ax.set_ylabel('Masked median classical split-Rhat');ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_title('Mathematical chain work: actual thinning included');fig.tight_layout()
for ext in ('svg','pdf','png'):fig.savefig(out/f'rhat_vs_transitions.{ext}',dpi=160)
plt.close(fig)
groups=[]
for variant in variants:
 records=[r for r in completed if r['variant']==variant]
 if records:groups.append(dict(variant=variant,passing_seed_count=len(records),seeds=[r['seed'] for r in records],median_measured_sampler_seconds=statistics.median(r['measured_sampler_seconds'] for r in records),median_selected_draws=statistics.median(r['selected_draws'] for r in records),median_ess_per_sampler_second=statistics.median(r['median_ess_per_sampler_second'] for r in records)))
if completed:
 fig,ax=plt.subplots(figsize=(8.5,4.6))
 for i,g in enumerate(groups):
  values=[r['measured_sampler_seconds'] for r in completed if r['variant']==g['variant']];med=statistics.median(values)
  ax.errorbar(i,med,yerr=[[med-min(values)],[max(values)-med]],fmt='o',capsize=4,color=colors[g['variant']]);ax.annotate(f"{len(values)}/3",(i,max(values)),xytext=(0,8),textcoords='offset points',ha='center')
 ax.set_xticks(range(len(groups)),[labels[g['variant']] for g in groups],rotation=12,ha='right');ax.set_ylabel('Actual selected-budget sampler seconds');ax.set_title('Fresh selected-budget retiming, output handling excluded')
 ax.grid(axis='y',alpha=.2);fig.text(.1,.01,'CPU: accumulated draw-call critical path (binding/state copy included). GPU: sampler phase includes allocation/slack/RNG init.\nScopes remain explicit; no interpolated time or pure-arithmetic claim.',fontsize=8);fig.tight_layout(rect=(0,.08,1,1))
 for ext in ('svg','pdf','png'):fig.savefig(out/f'time_to_target.{ext}',dpi=160)
 plt.close(fig)
summary=dict(schema_version=1,generated_utc=datetime.now(timezone.utc).isoformat(),condition_id=root.name,criterion=dict(rhat=1.05,bulk_ess=100,bounds_and_mass_balance=1e-7),rows=states,groups=groups,
 mask_compilation=str(compilation),curve_png=str(out/'rhat_vs_chain_length.png'),transition_curve_svg=str(out/'rhat_vs_transitions.svg'),transition_curve_png=str(out/'rhat_vs_transitions.png'),curve_csv=str(out/'rhat_curve.csv'),curve_svg=str(out/'rhat_vs_chain_length.svg'),curve_pdf=str(out/'rhat_vs_chain_length.pdf'),time_to_target_csv=str(out/'time_to_target.csv'),
 measured_time_plot_svg=str(out/'time_to_target.svg') if completed else None,
 setup_proof=str(root/'models/global_reround/global_reround_proof.json'),invalid_reference='Independent raw-model582-D preparation is excluded; all main methods use the same cached projected target.',
 note='Prefix curves use actual chain length, not inferred time. Only fresh selected-budget executions populate timing groups. Timer scopes are explicit; no global optimality or uniformity proof.')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if root.parent.name=='conditioned_prototype':(root.parent/'quality_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'curves':len(rows),'fresh_selected_runs':len(completed),'output':str(out)}))
