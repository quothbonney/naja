#!/usr/bin/env python3
"""Aggregate separate same-GPU conditional ablations without changing assays."""
import argparse,csv,json,math
from pathlib import Path
from datetime import datetime,timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--study',type=Path,default=B/'results/conditioned_extensions/off_rotate_whiten_20261108');a=p.parse_args();root=a.study.resolve();manifest=json.loads((root/'manifest.json').read_text());rows=[];curves=[]
freeze_path=root.parent/'CLOSEOUT_FREEZE.json'
freeze=json.loads(freeze_path.read_text()) if freeze_path.exists() else {}
cancelled_genes=set()
controller=root/'pipeline_controller.json'
if freeze and controller.exists():
 cancelled_ids=set(freeze.get('cancelled_in_progress_200k_captures',[]))
 cancelled_genes={j['condition_id'] for j in json.loads(controller.read_text()).get('jobs',[]) if j['job'] in cancelled_ids}
for case in manifest['cases']:
 for mode in manifest['mode_order']:
  path=root/case['condition_id']/mode/'quality.json'
  r=json.loads(path.read_text()) if path.exists() else dict(condition_id=case['condition_id'],mode=mode,status='pending')
  r.update(state_file=str(path),barrier_kappa_at_saved_start=case.get('barrier_kappa_at_saved_start'))
  observed=path.with_name('quality_200k_observed.json')
  if observed.exists():
   o=json.loads(observed.read_text());r.update(observed_budget=o['observed_budget'],observed_status=o['status'],observed_rhat=o['split_rhat_median'],observed_ess=o['bulk_ess_median'],observed_sampler_seconds=o['measured_sampler_seconds'],observed_source=str(observed))
  if freeze:
   if r['status']=='pending':r.update(raw_status='pending',status='not_run_closeout')
   elif r['status']=='running' and r.get('observed_status'):r.update(raw_status='running',status=r['observed_status']+'_no_selected_retime_closeout')
   elif r['status']=='running' and case['condition_id'] in cancelled_genes:r.update(raw_status='running',status='interrupted_capture_closeout')
  rows.append(r)
  curve=path.with_name('curve.json')
  if curve.exists():curves.extend(json.loads(curve.read_text()))
for name,data in [('summary',rows),('curves',curves)]:
 if not data:continue
 keys=list(dict.fromkeys(k for r in data for k in r))
 with (root/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
report=dict(generated_utc=datetime.now(timezone.utc).isoformat(),manifest=str(root/'manifest.json'),selection=manifest['selection'],protocol=f"16chains,1000warmup,thinning1000,TPB512,float64; sameallocatedGPU within each condition. Full{manifest.get('pilot_draws',50000):,}-draw pilot plus separately executed selectedbudget; no inferred timing. Median splitRhat<=1.05, bulkESS>=100, allbounds/massbalance1e-7.",rows=rows,curves=curves,note='This is an outcome/geometry-selected hard-case study, not the original50-target denominator. Unfinished task states are last-recorded; verifySlurm accounting. Existing original results untouched.')
if manifest.get('decoupled_pipeline'):
 report['protocol']=report['protocol'].replace('sameallocatedGPU within each condition.', 'GPU captures and CPU diagnostics use separate jobs; physical H100 may differ between pilot and fresh retime.')
if freeze:
 report['closeout_freeze']=freeze
 report['note']+=' User closed out old studies; queued cases were canceled and no new selected GPU retimes will be launched. Completed full-budget observations remain distinct from fresh selected-budget validation.'
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
colors={'off':'#777777','rotate':'#2463a0','whiten':'#c87921'}
for metric,label,filename in [('split_rhat_median','Median split Rhat','rhat_vs_chain_length'),('bulk_ess_median','Median bulk ESS','ess_vs_chain_length')]:
 fig,axes=plt.subplots(math.ceil(len(manifest['cases'])/3),3,figsize=(11,2.65*math.ceil(len(manifest['cases'])/3)),squeeze=False)
 for ax,case in zip(axes.ravel(),manifest['cases']):
  for mode in manifest['mode_order']:
   c=sorted((r for r in curves if r['condition_id']==case['condition_id'] and r['mode']==mode),key=lambda x:x['draws_per_chain'])
   if c:ax.plot([r['draws_per_chain'] for r in c],[r[metric] for r in c],color=colors.get(mode, '#2463a0' if 'fixed' in mode else '#c87921'),label=mode,marker='.',lw=1.3)
  ax.axhline(1.05 if metric=='split_rhat_median' else 100,color='black',ls='--',lw=.7)
  ax.set_xscale('log');ax.set_yscale('log');ax.set_title(case['condition_id'],fontsize=10);ax.grid(alpha=.15)
  ax.tick_params(labelsize=8);ax.set_xlabel('Retained draws / chain',fontsize=8);ax.set_ylabel(label,fontsize=8)
 for ax in list(axes.ravel())[len(manifest['cases']):]:ax.set_visible(False)
 handles,labels=next(((ax.get_legend_handles_labels()) for ax in axes.ravel() if ax.get_legend_handles_labels()[0]),([],[]))
 if handles:fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
 fig.tight_layout(rect=(0,0,1,.97))
 for ext in ('svg','pdf','png'):fig.savefig(root/(filename+'.'+ext),dpi=160)
 plt.close(fig)
lines=['# Same-GPU conditional barrier ablation','',f"Updated{report['generated_utc']}",'',report['selection'],'',report['protocol'],'','| Condition | Mode | Status | Selected draws | Sampler s | Correction s | Rhat | ESS |','|---|---|---|---:|---:|---:|---:|---:|']
for r in rows:
 def v(k):
  x=r.get(k);return '—' if x is None else f'{x:.5g}' if isinstance(x,(float,int)) else str(x)
 lines.append('| '+' | '.join([r['condition_id'],r['mode'],r['status'],v('selected_draws'),v('measured_sampler_seconds'),v('correction_seconds'),v('split_rhat_median'),v('bulk_ess_median')])+' |')
observed_rows=[r for r in rows if r.get('observed_status')]
if observed_rows:
 lines+=['','## Actual full-budget observations','', 'These are directly measured full-budget runs, not selected-prefix timing estimates. Fresh selected-budget validation remains separately labeled above.','', '| Condition | Full retained budget | Status | Sampler s | Rhat | ESS |','|---|---:|---|---:|---:|---:|']
 for r in observed_rows:lines.append(f"| {r['condition_id']} | {r['observed_budget']} | {r['observed_status']} | {r['observed_sampler_seconds']:.3f} | {r['observed_rhat']:.5f} | {r['observed_ess']:.1f} |")
if set(manifest['mode_order'])=={'off','rotate','whiten'}:
 lines+=['','With only coordinate moves and fixed thinning, exact CHRR is invariant under positive diagonal scaling after a fixed rotation: x=QSw and a coordinate step along S_j e_j spans the same physical chord as e_j after rescaling its uniform step. Thus whitening alone is a numerical/control ablation here; changes in finite-precision trajectories do not establish a different exact mixing kernel. Pair/dense directions can change under whitening.']
lines+=['','[Rhat SVG](rhat_vs_chain_length.svg) · [PDF](rhat_vs_chain_length.pdf) · [ESS SVG](ess_vs_chain_length.svg) · [Curve CSV](curves.csv) · [Full result states](summary.json) · [Exact manifest and selection](manifest.json).','',report['note']]
(root/'ABLATION.md').write_text('\n'.join(lines)+'\n');print(root/'summary.json')
