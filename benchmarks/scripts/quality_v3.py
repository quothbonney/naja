#!/usr/bin/env python3
"""Measure exact sample budgets that meet the shared schema-2 quality target.

A trajectory prefix selects a budget; only separately executed exact-budget
wall times count as measured time-to-quality. Fixed warmup is 1000 retained
samples per chain, so warmup move counts differ across thinning settings.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B))
VARIANTS={
 'naja128':dict(method='naja',tpb=128),
 'naja512':dict(method='naja',tpb=512,pilot=True),
 'cpu_default16':dict(method='hopsy',label='default',workers=16,source='default'),
 'cpu_default1':dict(method='hopsy',label='default',workers=1,source='default_workers1'),
 'cpu_jade16':dict(method='hopsy',label='jadebeck',workers=16,source='jadebeck'),
 'cpu_jade1':dict(method='hopsy',label='jadebeck',workers=1,source='jadebeck_workers1'),
 'cpu_fixed100':dict(method='hopsy',label='100',workers=16,source='100',extend=True,maxdraws=200000,extension_grid=[50000,100000,200000]),
 'cpu_fixed500':dict(method='hopsy',label='500',workers=16,source='500',extend=True,maxdraws=100000),
 'cpu_fixed2000':dict(method='hopsy',label='2000',workers=16,pilot=True),
 'cpu_fixed5000':dict(method='hopsy',label='5000',workers=16,pilot=True),
}
GRID=[100,250,500,1000,2000,5000,10000,20000,50000,100000,200000]


def source_run(model,variant,seed):
    v=VARIANTS[variant]
    if v.get('pilot'): return None
    parent=B/'results'/('naja' if v['method']=='naja' else 'hopsy')/model
    if v['method']=='hopsy': parent/=v['source']
    return parent/f'seed{seed}'


def read(path): return json.loads(path.read_text())
def write(path,data): path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def qualifies(m):
    r,e=m.get('split_rhat_median'),m.get('bulk_ess_median')
    return bool(m.get('schema_version')==2 and m.get('diagnostic_mask_status')=='model_lp_verified'
                and r is not None and e is not None and r<=1.01 and e>=100)


def run_budget(model,variant,seed,draws,out,find_prefix=False,phase='final'):
    """Only sampling process time contributes to measured time, not diagnostics."""
    import metrics
    from quality_diagnostics import reuse,save_identity,rhat_precheck
    v=VARIANTS[variant]
    path=out/f'{phase}-budget{draws}'
    path.mkdir(parents=True,exist_ok=True)
    rp=path/'run.json'
    if not rp.exists() or read(rp).get('status')!='complete':
        if v['method']=='naja':
            cmd=['/usr/bin/env','bash',str(B/'scripts/enter.sh'),'python',str(B/'naja_runner.py'),
                 '--exe',str(B/'build-h100/naja'),'--model',str(B/'data/models'/model),
                 '--out',str(path),'--seed',str(seed),'--chains','16','--draws',str(draws),
                 '--warmup','1000','--tpb',str(v['tpb']),'--timeout','1500','--resume']
            # Runner refuses nonempty output directories; command logs live outside.
            if not rp.exists() and not any(path.iterdir()): path.rmdir()
        else:
            cmd=[str(B/'env-cpu/bin/python'),str(B/'cpu_baseline.py'),'sample',
                 '--model',str(B/'data/models'/model),'--out',str(path),'--seed',str(seed),
                 '--chains','16','--draws',str(draws),'--warmup','1000','--thinning',v['label'],
                 '--workers',str(v['workers']),'--chunk',str(draws+1000)]
        env=os.environ.copy()
        env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (out/f'{phase}-budget{draws}.sampling.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=1550)
    record=read(rp)
    wall=record.get('wall_including_normalization_seconds',record.get('process_wall_seconds'))
    if wall is None: raise RuntimeError('Missing measured full process wall time')
    mp=path/'metrics.json'
    if not mp.exists() or read(mp).get('schema_version')!=2:
        pilot=out/f'pilot-budget{draws}'
        reused=phase=='final' and reuse(path/'samples.npy',B/'data/models'/model,path,wall,pilot)
        if not reused:
            prefixes=[n for n in GRID if n<=draws] if find_prefix else [draws]
            passed_rhat=prefixes
            if phase=='pilot':
                checks=[rhat_precheck(path/'samples.npy',B/'data/models'/model,draws_limit=n) for n in prefixes]
                write(path/'rhat-precheck.json',metrics.json_safe({'prefixes':checks,
                    'note':'ESS omitted only when masked split Rhat already fails the AND target.'}))
                passed_rhat=[n for n,check in zip(prefixes,checks)
                    if check['split_rhat_median'] is None or check['split_rhat_median']<=1.0100000001]
                if not passed_rhat:
                    check=checks[-1]
                    check.update(model=model,samples=str(path/'samples.npy'),wall_seconds=wall,
                        first_quality_prefix_draws=None,quality_met=False,
                        bulk_ess_not_computed_reason='All evaluated masked Rhat medians exceed 1.01; AND target already fails.',
                        prefix_quality=[dict(draws_per_chain=n,split_rhat_median=c['split_rhat_median'],bulk_ess_median=None,
                            bulk_ess_computed=False,quality_met=False) for n,c in zip(prefixes,checks)])
                    write(mp,metrics.json_safe(check))
            if passed_rhat:
                metrics.evaluate(path/'samples.npy',B/'data/models'/model,path,prefixes=passed_rhat)
                save_identity(path/'samples.npy',B/'data/models'/model,path)
    m=read(mp)
    if m['diagnostic_mask_status']!='model_lp_verified': raise RuntimeError('Unverified diagnostic mask')
    return dict(draws_per_chain=draws,wall_seconds=wall,quality_met=qualifies(m),
        split_rhat_median=m['split_rhat_median'],bulk_ess_median=m['bulk_ess_median'],
        first_passing_prefix=m.get('first_quality_prefix_draws'),run_dir=str(path),
        bulk_ess_computed=m.get('bulk_ess_computed',True),diagnostics_reused_from=m.get('diagnostics_reused_from'),
        source_sampling_command=record.get('command'),thinning=record.get('thinning',record.get('profile',{}).get('config',{}).get('thinning')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--remeasure-final',action='store_true')
    args=p.parse_args()
    model,variant,seed=args.model,args.variant,args.seed
    out=B/'results/quality'/model/variant/f'seed{seed}'
    out.mkdir(parents=True,exist_ok=True)
    result_path=out/'quality.json'
    if result_path.exists() and not args.remeasure_final and read(result_path).get('status') in ['complete','cap_reached']:
        print(result_path);return
    v=VARIANTS[variant]
    source=source_run(model,variant,seed)
    source_metrics=read(source/'metrics.json') if source else None
    if source_metrics and (source_metrics.get('schema_version')!=2 or source_metrics.get('diagnostic_mask_status')!='model_lp_verified'):
        raise RuntimeError('Source trajectory must have verified schema-2 diagnostics')
    if args.remeasure_final:
        result=read(result_path)
        candidate=result['measured']['draws_per_chain']
        measured=run_budget(model,variant,seed,candidate,out,phase='final')
        result['attempts'].append(dict(role='independent_final_remeasurement',**measured))
        result['measured']=measured
        result['status']='complete' if measured['quality_met'] else 'selected_budget_failed_revalidation'
        result['finished_utc']=datetime.now(timezone.utc).isoformat()
        result['independent_final_remeasurement']=True
        write(result_path,result)
        print(json.dumps(result,indent=2),flush=True)
        return
    result=dict(model=model,variant=variant,seed=seed,status='running',chains=16,warmup=1000,
      started_utc=datetime.now(timezone.utc).isoformat(),worker_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      diagnostics_helper_sha256=hashlib.sha256((B/'scripts/quality_diagnostics.py').read_bytes()).hexdigest(),criterion='masked median classical split-Rhat<=1.01 and pooled rank-normalized bulkESS median>=100',
      warmup_note='1000 retained draws per chain, so warmup move counts differ with thinning',
      source_run=str(source) if source else None,
      source_metrics_sha256=hashlib.sha256((source/'metrics.json').read_bytes()).hexdigest() if source else None,
      source_first_passing_prefix=source_metrics.get('first_quality_prefix_draws') if source_metrics else None,
      attempts=[],timing_note='Independent exact-budget sampling process wall, including warmup/load/write and Naja normalization; excludes diagnostics and exploratory pilots; no interpolation.',
      slurm={k:val for k,val in os.environ.items() if k.startswith('SLURM_')})
    write(result_path,result)
    try:
        candidate=source_metrics.get('first_quality_prefix_draws') if source_metrics else None
        extend=v.get('extend',False) or v['method']=='naja' or (model=='e_coli_core' and variant=='cpu_default1')
        maxdraws=v.get('maxdraws',200000 if v['method']=='naja' else 100000)
        if source_metrics and candidate is None and not extend:
            result['status']='no_quality_within_source_budget'
        else:
            # No original same-settings trajectory for TPB512: measure its 10k pilot first.
            if source is None:
                pilot=run_budget(model,variant,seed,10000,out,find_prefix=True,phase='pilot')
                result['attempts'].append(dict(role='budget_finding_pilot',**pilot))
                candidate=pilot['first_passing_prefix']
            if candidate is None and extend:
                for budget in v.get('extension_grid',[20000,50000,100000,200000]):
                    if budget>maxdraws: break
                    pilot=run_budget(model,variant,seed,budget,out,find_prefix=False,phase='pilot')
                    result['attempts'].append(dict(role='extension_budget',**pilot))
                    write(result_path,result)
                    if pilot['quality_met']:
                        candidate=budget;break
            if candidate is not None:
                measured=run_budget(model,variant,seed,candidate,out,find_prefix=False)
                result['attempts'].append(dict(role='measured_exact_budget',**measured))
                result['measured']=measured
                result['status']='complete' if measured['quality_met'] else 'selected_budget_failed_revalidation'
            else: result['status']='cap_reached'
    except Exception as exc:
        result['status']='failed';result['error']=repr(exc)
        write(result_path,result)
        raise
    result['finished_utc']=datetime.now(timezone.utc).isoformat()
    write(result_path,result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
