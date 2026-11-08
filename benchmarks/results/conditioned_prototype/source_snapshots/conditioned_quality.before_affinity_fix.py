#!/usr/bin/env python3
"""Exact cached-target quality curves and fresh selected-budget retiming."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
B=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(B),str(B/'scripts')]
from conditional_model_api import write_json,sha,ensure_cached_mask,install_cached_mask,transport_mask
from conditioned_prototype import physical_affinity,call
from quality_diagnostics import rhat_precheck
GRID=(250,500,1000,2000,5000,10000,20000,50000,200000)
CODE_SHA=sha(Path(__file__))
METRICS_SHA=sha(B/'metrics.py')

def diagnostic(samples,model,output,prefixes):
    output.mkdir(parents=True,exist_ok=True)
    path=output/'metrics.json'
    if not path.exists():
        call([B/'env-cpu/bin/python',B/'metrics.py','--samples',samples,'--model-dir',model,'--output',output,
              '--prefixes',*prefixes],output.parent/(output.name+'.metrics.log'))
    result=json.loads(path.read_text())
    assert result['schema_version']==2 and result['diagnostic_mask_status']=='model_lp_verified'
    assert result['metrics_code_sha256']==METRICS_SHA
    return result

def mass_balance(samples,model):
    from scipy import sparse
    p=dict(np.load(model/'rounding/polytope.npz'));S=sparse.csr_matrix(np.loadtxt(model/'gem/A_eq.csv',delimiter=','))
    x=np.load(samples,mmap_mode='r');maximum=0.
    for chain in x:
        for begin in range(0,x.shape[1],1024):
            flux=chain[begin:begin+1024]@p['T'].T+p['shift'].reshape(-1)
            maximum=max(maximum,float(np.abs(S@flux.T).max()))
    return maximum

def run_budget(root,model,variant,seed,draws,phase,with_diagnostics=True):
    destination=root/'quality'/variant/f'seed{seed}'/f'{phase}-budget{draws}'
    destination.parent.mkdir(parents=True,exist_ok=True)
    cpu=variant!='naja_rotate'
    if not (destination/'run.json').exists():
        if cpu:
            affinity=physical_affinity()
            os.environ['HOPSY_CXXFLAGS']='-O3 -march=native -mtune=native'
            os.environ['HOPSY_BUILD_MANIFEST']=str(B/'audit/hopsy_native_build.json')
            proof=json.loads((B/'audit/hopsy-worker-timing-verification/iJO1366-real-parity.json').read_text())
            reference=json.loads((B.parent/proof['instrumented_run']/'sampling_math_timing.json').read_text())
            assert proof['status']=='passed' and proof['bitwise_equal']
            assert reference['proof']['instrumentation_module_sha256']==sha(B/'scripts/hopsy_worker_timing.py')
            command=[B/'env-cpu-native/bin/python',B/'scripts/timed_cpu_baseline.py','sample','--model',model,
                '--out',destination,'--seed',seed,'--chains',16,'--draws',draws,'--warmup',1000,'--thinning',2000,
                '--workers',16,'--chunk',draws+1000,'--build-variant','native']
        else:
            affinity=None
            gpu64=(root/'condition.json').exists()
            command=[B/'scripts/enter.sh',B/'env-gpu/bin/python',B/'naja_runner.py','--exe',B/('build-h100-f64/naja' if gpu64 else 'build-h100/naja'),
                '--model',model,'--out',destination,'--seed',seed,'--chains',16,'--draws',draws,'--warmup',1000,
                '--thinning',1000,'--tpb',512,'--barrier','rotate','--timeout',1200]
            if gpu64:command+=['--sample-dtype','float64']
        elapsed=call(command,destination.parent/(destination.name+'.sampling.log'))
        if affinity:write_json(destination/'physical_affinity.json',affinity)
        write_json(destination/'sampling_command.json',dict(command=list(map(str,command)),external_process_seconds=elapsed,
            orchestration_sha256=CODE_SHA))
    run=json.loads((destination/'run.json').read_text())
    assert run['status']=='complete' and run['draws']==draws and run['seed']==seed
    metric=None
    if with_diagnostics:
        prefixes=[n for n in GRID if n<=draws] if phase=='pilot' else [draws]
        metric=diagnostic(destination/'samples.npy',model,destination,prefixes)
        if not (destination/'mass_balance.json').exists():
            write_json(destination/'mass_balance.json',dict(maximum_stoichiometric_residual=mass_balance(destination/'samples.npy',model)))
    if cpu:
        timing=json.loads((destination/'sampling_math_timing.json').read_text())
        seconds=timing['worker_draw_critical_path_seconds'];scope='CPU max accumulated MarkovChain.draw intervals; binding/state-copy included, startup/IPC/list assembly excluded'
        secondary=timing['instrumented_wrapper_process_seconds'];correction=0.
    else:
        seconds=run['profile']['timing']['gpu_sampling_s'];scope='GPU sampler phase: device allocation/slack/RNG init plus CHR kernel; excludes backmap/download/output/adapter'
        secondary=run['wall_including_normalization_seconds'];correction=run['profile']['timing']['barrier_rotation_s']
    return destination,metric,dict(measured_sampler_seconds=seconds,timer_scope=scope,secondary_process_seconds=secondary,
                                  correction_seconds=correction)

def prefix_copy(samples,destination,draws):
    source=np.load(samples,mmap_mode='r')
    if not destination.exists():
        out=np.lib.format.open_memmap(destination,mode='w+',dtype=source.dtype,shape=(source.shape[0],draws,source.shape[2]))
        for begin in range(0,draws,512):out[:,begin:begin+512]=source[:,begin:min(begin+512,draws)]
        out.flush()
    return destination

def curve_rows(root,model,pilot,variant,seed):
    original=model/'active_reactions.before_cached_target.npy'
    changed=original.exists() and not np.array_equal(np.load(original),np.load(model/'active_reactions.npy'))
    override=root/'quality'/variant/f'seed{seed}'/'pilot_recomputed_common_mask'
    if changed and not str(pilot).startswith(str(root/'quality')):
        n=np.load(pilot/'samples.npy',mmap_mode='r').shape[1]
        metric=diagnostic(pilot/'samples.npy',model,override,[k for k in GRID if k<=n])
        metric_path=override/'metrics.json'
    else:
        metric=json.loads((pilot/'metrics.json').read_text());metric_path=pilot/'metrics.json'
    assert metric['metrics_code_sha256']==METRICS_SHA
    n=np.load(pilot/'samples.npy',mmap_mode='r').shape[1]
    by_budget={row['draws_per_chain']:row for row in metric['prefix_quality']}
    run=json.loads((pilot/'run.json').read_text())
    thin=run.get('thinning',run.get('profile',{}).get('config',{}).get('thinning'))
    rows=[]
    for draws in GRID:
        if draws>n:continue
        if draws in by_budget:
            x=by_budget[draws]
            record=dict(draws_per_chain=draws,split_rhat_median=x['split_rhat_median'],split_rhat_p95=x['split_rhat_p95'],
                        bulk_ess_median=x['bulk_ess_median'],sample_bound_violation_fraction=metric['sample_bound_violation_fraction'] if metric['sample_bound_violation_fraction']==0 else None,
                        diagnostic_source=str(metric_path),cached_frozen_diagnostics=True)
        else:
            p=root/'quality'/variant/f'seed{seed}'/f'prefix{draws}'
            p.mkdir(parents=True,exist_ok=True)
            checkpath=p/'rhat_precheck.json'
            if not checkpath.exists():write_json(checkpath,rhat_precheck(pilot/'samples.npy',model,draws_limit=draws))
            check=json.loads(checkpath.read_text());record=dict(draws_per_chain=draws,**{k:check.get(k) for k in ['split_rhat_median','split_rhat_p95','bulk_ess_median','sample_bound_violation_fraction']},diagnostic_source=str(checkpath),cached_frozen_diagnostics=False)
            if check['split_rhat_median']<=1.05 and check['sample_bound_violation_fraction']==0:
                samples=prefix_copy(pilot/'samples.npy',p/'samples.npy',draws)
                full=diagnostic(samples,model,p,[draws]);record.update(bulk_ess_median=full['bulk_ess_median'],diagnostic_source=str(p/'metrics.json'))
        record.update(condition_id=root.name,variant=variant,seed=seed,actual_thinning=thin,retained_transitions_per_chain=draws*thin,total_transitions_per_chain=(draws+1000)*thin,pilot_run=str(pilot),actual_execution_at_this_budget=False,
                      sampled_time_at_prefix=None,linear_time_estimate_used=False)
        record['quality_met']=bool(record['split_rhat_median'] is not None and record['split_rhat_median']<=1.05 and record['bulk_ess_median'] is not None and record['bulk_ess_median']>=100 and record['sample_bound_violation_fraction']==0)
        rows.append(record)
    return rows

def main(a):
    root=a.condition_root.resolve();shared=root/'models/shared';glob=root/'models/global_reround'
    if a.sample_only:
        model=glob if a.variant=='global_cpu' else shared
        run_budget(root,model,a.variant,a.seed,10000,'pilot',with_diagnostics=False)
        return
    # This oracle is the exact cached target. No raw-model equality reduction.
    ensure_cached_mask(shared)
    if (glob/'global_reround_proof.json').exists():transport_mask(shared,glob)
    model=glob if a.variant=='global_cpu' else shared
    assert (model/'diagnostic_mask.json').exists()
    if a.variant=='global_cpu':
        proof=json.loads((glob/'global_reround_proof.json').read_text());assert proof['source_geometry_sha256']==sha(shared/'rounding/polytope.npz')
    state=root/'quality'/a.variant/f'seed{a.seed}'/'quality.json';state.parent.mkdir(parents=True,exist_ok=True)
    write_json(state,dict(status='running',condition_id=root.name,variant=a.variant,seed=a.seed))
    pilot=a.pilot_source.resolve() if a.pilot_source else None
    max_budgets=[10000,50000]+([200000] if a.allow_200k else [])
    rows=[];chosen=None
    for budget in max_budgets:
        if pilot is None or np.load(pilot/'samples.npy',mmap_mode='r').shape[1]<budget:
            if pilot is not None and budget==10000:continue
            pilot,metric,_=run_budget(root,model,a.variant,a.seed,budget,'pilot')
        else:metric=json.loads((pilot/'metrics.json').read_text())
        rows=curve_rows(root,model,pilot,a.variant,a.seed)
        eligible=[r['draws_per_chain'] for i,r in enumerate(rows) if r['quality_met'] and all(x['quality_met'] for x in rows[i:])]
        if eligible:chosen=min(eligible);break
        if np.load(pilot/'samples.npy',mmap_mode='r').shape[1]>=max_budgets[-1]:break
        pilot=None
    write_json(state.with_name('curve.json'),rows)
    base=dict(condition_id=root.name,variant=a.variant,seed=a.seed,pilot_run=str(pilot),prefix_curve_json=str(state.with_name('curve.json')),
        selection_policy='Earliest evaluated passing prefix with no later observed failure; avoids unstable early crossings. Actual selected budget is then rerun.',
        criterion='masked median classical split-Rhat<=1.05 and pooled bulk ESS>=100; original/conditional bounds and mass balance<=1e-7',
        curve_only_has_actual_chain_lengths=True,interpolated_times_used=False,orchestration_sha256=CODE_SHA)
    if chosen is None:
        write_json(state,dict(**base,status='no_quality_within_cap',observed_budget=max_budgets[-1]));return
    measured,metric,timing=run_budget(root,model,a.variant,a.seed,chosen,'selected')
    eq=json.loads((measured/'mass_balance.json').read_text())['maximum_stoichiometric_residual']
    passed=metric['split_rhat_median']<=1.05 and metric['bulk_ess_median']>=100 and metric['sample_bound_violation_fraction']==0 and eq<=1e-7
    write_json(state,dict(**base,status='complete' if passed else 'fresh_selected_budget_failed',selected_draws=chosen,
        measured_run=str(measured),**timing,split_rhat_median=metric['split_rhat_median'],split_rhat_p95=metric['split_rhat_p95'],
        bulk_ess_median=metric['bulk_ess_median'],median_ess_per_sampler_second=metric['bulk_ess_median']/timing['measured_sampler_seconds'],
        sample_bound_violation_fraction=metric['sample_bound_violation_fraction'],maximum_stoichiometric_residual=eq,
        setup_proof=str(glob/'global_reround_proof.json'),geometry_sha256=sha(model/'rounding/polytope.npz'),
        samples_sha256=sha(measured/'samples.npy')))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--condition-root',type=Path,required=True)
    p.add_argument('--variant',choices=['global_cpu','stale_cpu','naja_rotate'],required=True);p.add_argument('--seed',type=int,default=101)
    p.add_argument('--pilot-source',type=Path);p.add_argument('--allow-200k',action='store_true');p.add_argument('--sample-only',action='store_true');a=p.parse_args()
    try:main(a)
    except Exception as e:
        target=a.condition_root/'quality'/a.variant/f'seed{a.seed}'/'quality.json';target.parent.mkdir(parents=True,exist_ok=True)
        write_json(target,dict(status='failed',condition_id=a.condition_root.name,variant=a.variant,seed=a.seed,error=repr(e)));raise
