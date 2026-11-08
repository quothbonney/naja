#!/usr/bin/env python3
"""Split GPU capture and frozen CPU quality analysis for conditional200k follow-ups."""
import argparse,json,sys,time,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
B=Path(__file__).resolve().parents[1];sys.path[:0]=[str(B),str(B/'scripts')]
import conditioned_quality as frozen
from conditional_model_api import sha,write_json
GRID=(250,500,1000,2000,5000,10000,20000,50000,200000)
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--phase',choices=['capture-pilot','analyze-pilot','capture-selected','analyze-selected'],required=True);a=p.parse_args();study=json.loads(a.manifest.read_text());case=study['cases'][a.index];root=Path(case['condition_root']);model=root/'models/shared';out=Path(study['output_directory'])/case['condition_id'];out.mkdir(parents=True,exist_ok=True);path=out/'pipeline_state.json';state=json.loads(path.read_text()) if path.exists() else {};qpath=out/'rotate/quality.json';qpath.parent.mkdir(exist_ok=True)
if qpath.exists() and not qpath.with_name('quality.before_decoupling.json').exists():write_json(qpath.with_name('quality.before_decoupling.json'),json.loads(qpath.read_text()))
def call(cmd,log):
 remaining=(datetime.fromisoformat(study['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()
 if remaining<=0:raise TimeoutError('Extension22UTCdeadline')
 start=time.perf_counter();log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('w') as f:subprocess.run(list(map(str,cmd)),check=True,stdout=f,stderr=subprocess.STDOUT,timeout=min(2700,remaining))
 return time.perf_counter()-start
frozen.call=call
assert sha(model/'rounding/polytope.npz')==case['geometry_sha256']
base=dict(condition_id=case['condition_id'],condition_root=str(root),mode='rotate',seed=101,geometry_sha256=case['geometry_sha256'],pipeline_worker_sha256=sha(Path(__file__)),selection_note=study['selection'],criterion='masked medianclassical splitRhat<=1.05,bulkESS>=100,unmaskedbounds/massbalance1e-7; unchangedfrozenmetrics',protocol='16chains,1000warmup,thin1000,TPB512,float64')
selected='selected' in a.phase;draws=int(state['selected_draws']) if selected else study['pilot_draws'];dest=out/'rotate'/f'{"selected" if selected else "pilot"}-budget{draws}'
try:
 write_json(qpath,dict(**base,status='running',current_phase=a.phase))
 if a.phase.startswith('capture'):
  record=dest/'run.json'
  if not record.exists():
   cmd=[B/'scripts/enter.sh',B/'env-gpu/bin/python',B/'naja_runner.py','--exe',B/'build-h100-f64/naja','--model',model,'--out',dest,'--seed',101,'--chains',16,'--draws',draws,'--warmup',1000,'--thinning',1000,'--tpb',512,'--barrier','rotate','--sample-dtype','float64','--timeout',1500]
   seconds=call(cmd,dest.parent/(dest.name+'.capture.log'));write_json(dest/'decoupled_capture_command.json',dict(argv=list(map(str,cmd)),external_process_seconds=seconds,source_sha256=sha(Path(__file__))))
  run=json.loads(record.read_text());assert run['status']=='complete' and run['draws']==draws and run['barrier']=='rotate' and run['sample_dtype']=='float64' and run['exe_sha256']==study['binary_sha256']
  state['selected_captured' if selected else 'pilot_captured']=True
 else:
  run=json.loads((dest/'run.json').read_text());assert run['status']=='complete'
  metric=frozen.diagnostic(dest/'samples.npy',model,dest,[n for n in GRID if n<=draws] if not selected else [draws])
  masspath=dest/'mass_balance.json'
  if not masspath.exists():write_json(masspath,dict(maximum_stoichiometric_residual=frozen.mass_balance(dest/'samples.npy',model)))
  eq=json.loads(masspath.read_text())['maximum_stoichiometric_residual'];good=metric['split_rhat_median']<=1.05 and metric['bulk_ess_median']>=100 and metric['sample_bound_violation_fraction']==0 and eq<=1e-7
  timing=dict(measured_sampler_seconds=run['profile']['timing']['gpu_sampling_s'],correction_seconds=run['profile']['timing']['barrier_rotation_s'],split_rhat_median=metric['split_rhat_median'],bulk_ess_median=metric['bulk_ess_median'],sample_bound_violation_fraction=metric['sample_bound_violation_fraction'],maximum_stoichiometric_residual=eq,timer_scope='GPU sampler includesallocation/slack/RNG+CHRR;excludesbackmap/download/write andseparatecorrection')
  if not selected:
   curve=[]
   for r in metric['prefix_quality']:
    z=dict(r,condition_id=case['condition_id'],mode='rotate',seed=101,actual_thinning=1000,total_transitions_per_chain=(r['draws_per_chain']+1000)*1000,sampled_time_at_prefix=None,linear_time_estimate_used=False)
    z['quality_met']=r['split_rhat_median']<=1.05 and r['bulk_ess_median']>=100 and metric['sample_bound_violation_fraction']==0 and eq<=1e-7;curve.append(z)
   curve.sort(key=lambda r:r['draws_per_chain']);write_json(qpath.with_name('curve.json'),curve)
   eligible=[r['draws_per_chain'] for i,r in enumerate(curve) if r['quality_met'] and all(t['quality_met'] for t in curve[i:])]
   state['pilot_analyzed']=True;state['selected_draws']=min(eligible) if eligible else None
   observed=dict(**base,**timing,status='observed_budget_pass' if good else 'observed_budget_failed',observed_budget=draws,actual_run=str(dest),note='Measuredfull200kbudget withqualityfromsamefullrun;notminimalTTT,noprefixwallinterpolation,nofreshselectedbudgetvalidationyet')
   write_json(out/'rotate/quality_200k_observed.json',observed)
   if not eligible:state['terminal_status']='no_quality_within_cap';write_json(qpath,dict(**base,**timing,status='no_quality_within_cap',observed_budget=draws,prefix_curve_json=str(qpath.with_name('curve.json'))))
  else:
   state['selected_analyzed']=True;state['terminal_status']='complete' if good else 'fresh_selected_budget_failed'
   write_json(qpath,dict(**base,**timing,status=state['terminal_status'],selected_draws=draws,measured_run=str(dest),median_ess_per_sampler_second=metric['bulk_ess_median']/timing['measured_sampler_seconds'],samples_sha256=sha(dest/'samples.npy'),prefix_curve_json=str(qpath.with_name('curve.json'))))
 state.update(updated_utc=datetime.now(timezone.utc).isoformat(),latest_phase=a.phase);write_json(path,state)
except Exception as error:
 state.update(failed_phase=a.phase,error=repr(error),terminal_status='failed',updated_utc=datetime.now(timezone.utc).isoformat());write_json(path,state);write_json(qpath,dict(**base,status='failed',failed_phase=a.phase,error=repr(error)));raise
