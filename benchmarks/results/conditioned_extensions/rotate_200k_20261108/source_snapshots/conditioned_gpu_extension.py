#!/usr/bin/env python3
"""Fresh same-GPU off/rotate/whiten study; never edits original panel outputs."""
import argparse,json,os,sys,time,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
B=Path(__file__).resolve().parents[1];sys.path[:0]=[str(B),str(B/'scripts')]
from conditional_model_api import sha,write_json
import conditioned_quality as frozen_quality
from conditioned_quality import diagnostic,mass_balance
def call(command,log):
 deadline=datetime.fromisoformat(study['deadline_utc'])
 remaining=(deadline-datetime.now(timezone.utc)).total_seconds()
 if remaining<=0:raise TimeoutError('Extension deadline2026-11-08T22:00UTC')
 log=Path(log);log.parent.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
 with log.open('w') as handle:
  subprocess.run(list(map(str,command)),check=True,stdout=handle,stderr=subprocess.STDOUT,timeout=min(2700,remaining))
 return time.perf_counter()-started
# Only the subprocess orchestration callback changes in this process; frozen
# metric implementations, models, thresholds and source files remain untouched.
frozen_quality.call=call
GRID=(250,500,1000,2000,5000,10000,20000,50000,200000)
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--index',type=int,required=True);a=p.parse_args()
study=json.loads(a.manifest.read_text());case=study['cases'][a.index];root=Path(case['condition_root']);model=root/'models/shared';out=Path(study['output_directory'])/case['condition_id'];out.mkdir(parents=True,exist_ok=True)
geometry_hash=sha(model/'rounding/polytope.npz');code_hash=sha(Path(__file__));seed=101
pilot_budget=int(study.get('pilot_draws',50000))
assert geometry_hash==case['geometry_sha256']

def execute(mode,draws,phase):
 dest=out/mode/f'{phase}-budget{draws}';dest.parent.mkdir(parents=True,exist_ok=True)
 if not (dest/'run.json').exists():
  cmd=[B/'scripts/enter.sh',B/'env-gpu/bin/python',B/'naja_runner.py','--exe',B/'build-h100-f64/naja','--model',model,'--out',dest,'--seed',seed,'--chains',16,'--draws',draws,'--warmup',1000,'--thinning',1000,'--tpb',512,'--barrier',mode,'--sample-dtype','float64','--timeout',1500]
  elapsed=call(cmd,dest.parent/(dest.name+'.sampling.log'))
  write_json(dest/'exact_command.json',dict(argv=list(map(str,cmd)),external_process_seconds=elapsed,source_sha256=code_hash))
 run=json.loads((dest/'run.json').read_text());assert run['status']=='complete' and run['barrier']==mode and run['sample_dtype']=='float64'
 assert run['draws']==draws and run['chains']==16 and run['warmup']==1000 and run['requested_thinning']==1000
 assert run['exe_sha256']==study['binary_sha256']
 if mode=='off':assert run['profile']['timing']['barrier_rotation_s']==0 and not any(z.startswith('--barrier-') for z in run['command'])
 metrics=diagnostic(dest/'samples.npy',model,dest,[k for k in GRID if k<=draws] if phase=='pilot' else [draws])
 if not (dest/'mass_balance.json').exists():write_json(dest/'mass_balance.json',dict(maximum_stoichiometric_residual=mass_balance(dest/'samples.npy',model)))
 eq=json.loads((dest/'mass_balance.json').read_text())['maximum_stoichiometric_residual']
 return dest,run,metrics,eq

for mode in study['mode_order']:
 state=out/mode/'quality.json'
 if state.exists() and json.loads(state.read_text()).get('status') in ('complete','no_quality_within_cap','fresh_selected_budget_failed'):continue
 state.parent.mkdir(parents=True,exist_ok=True)
 base=dict(condition_id=case['condition_id'],condition_root=str(root),mode=mode,seed=seed,protocol='16chains,1000warmup,thinning1000,TPB512,float64; all modes same allocated GPU',source_sha256=code_hash,geometry_sha256=geometry_hash,selection_note=study['selection'],criterion='median classical splitRhat<=1.05,bulkESS>=100,unmasked original/conditional bounds and mass balance1e-7')
 write_json(state,dict(**base,status='running',started_utc=datetime.now(timezone.utc).isoformat()))
 try:
  pilot,run,metric,eq=execute(mode,pilot_budget,'pilot')
  curve=[]
  for x in metric['prefix_quality']:
   item=dict(x,mode=mode,condition_id=case['condition_id'],seed=seed,actual_thinning=1000,total_transitions_per_chain=(x['draws_per_chain']+1000)*1000,sampled_time_at_prefix=None,linear_time_estimate_used=False)
   item['quality_met']=x['split_rhat_median']<=1.05 and x['bulk_ess_median']>=100 and metric['sample_bound_violation_fraction']==0 and eq<=1e-7
   curve.append(item)
  curve.sort(key=lambda x:x['draws_per_chain']);write_json(state.with_name('curve.json'),curve)
  selected=[x['draws_per_chain'] for i,x in enumerate(curve) if x['quality_met'] and all(y['quality_met'] for y in curve[i:])]
  base.update(pilot_run=str(pilot),prefix_curve_json=str(state.with_name('curve.json')),pilot_sampler_seconds=run['profile']['timing']['gpu_sampling_s'],pilot_correction_seconds=run['profile']['timing']['barrier_rotation_s'],pilot_final_rhat=metric['split_rhat_median'],pilot_final_ess=metric['bulk_ess_median'],pilot_sample_bound_violation_fraction=metric['sample_bound_violation_fraction'],pilot_mass_balance_max=eq)
  if not selected:
   write_json(state,dict(**base,status='no_quality_within_cap',observed_budget=pilot_budget));continue
  chosen=min(selected);dest,run,metric,eq=execute(mode,chosen,'selected')
  passed=metric['split_rhat_median']<=1.05 and metric['bulk_ess_median']>=100 and metric['sample_bound_violation_fraction']==0 and eq<=1e-7
  seconds=run['profile']['timing']['gpu_sampling_s']
  write_json(state,dict(**base,status='complete' if passed else 'fresh_selected_budget_failed',selected_draws=chosen,measured_run=str(dest),measured_sampler_seconds=seconds,correction_seconds=run['profile']['timing']['barrier_rotation_s'],split_rhat_median=metric['split_rhat_median'],bulk_ess_median=metric['bulk_ess_median'],sample_bound_violation_fraction=metric['sample_bound_violation_fraction'],maximum_stoichiometric_residual=eq,median_ess_per_sampler_second=metric['bulk_ess_median']/seconds,samples_sha256=sha(dest/'samples.npy'),timer_scope='GPU sampler includes device allocation/slack/RNG and CHRR; excludes backmap/download/write and separate barrier correction'))
 except Exception as error:
  write_json(state,dict(**base,status='failed',error=repr(error)));continue
 assert sha(model/'rounding/polytope.npz')==geometry_hash
