#!/usr/bin/env python3
"""Three new seed pairs at frozen conservative budgets; no adaptive extensions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

B=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(B),str(B/'scripts')]
import metrics
from conditioned_prototype import physical_affinity
R=B/'results/fresh16_conservative/iJO1366'
MODEL=B/'results/high_throughput/iJO1366/cached_target_audit/model'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def run_case(method,seed,output):
    if (output/'quality.json').exists():
        return json.loads((output/'quality.json').read_text())
    if output.exists():
        assert method=='gpu' and not (output/'run.json').exists(), 'Review incomplete sampling before retry'
        previous=output.with_name(output.name+'.failed_before_sampling')
        assert not previous.exists(), 'Only one known preflight retry allowed'
        output.rename(previous)
    draws=15000 if method=='cpu' else 3000
    if method=='cpu':
        command=[B/'env-cpu-native/bin/python',B/'scripts/timed_cpu_baseline.py','sample',
            '--model',MODEL,'--out',output,'--seed',str(seed),'--chains','16',
            '--draws',str(draws),'--warmup','1000','--thinning','2000','--workers','16',
            '--chunk',str(draws+1000),'--build-variant','native']
    else:
        exe=B/'build-h100/naja'
        with exe.open('rb') as handle:
            assert hashlib.file_digest(handle,'sha256').hexdigest()=='c7baf4f0ed282b936ac0dbac4b896bfbfc040eb06da6f84aff6605fddf4b0fc6'
        command=['/usr/bin/env','bash',B/'scripts/enter.sh',B/'env-gpu/bin/python',B/'naja_runner.py',
            '--exe',exe,'--model',MODEL,'--out',output,'--seed',str(seed),'--chains','16',
            '--draws',str(draws),'--warmup','1000','--tpb','512','--thinning','1000',
            '--barrier','rotate','--pair-prob','0','--timeout','600']
    command=list(map(str,command))
    output.mkdir(parents=True,exist_ok=False)
    write(output.parent/(method+'.orchestration.json'),dict(command=command,started_utc=datetime.now(timezone.utc).isoformat(),
        protocol=str(R/'protocol.json'),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    if method=='gpu':assert not any(output.iterdir()), 'Naja requires an empty output directory'
    with (output.parent/(method+'.execution.log')).open('w') as log:
        subprocess.run(command,check=True,timeout=900,stdout=log,stderr=subprocess.STDOUT)
    run=json.loads((output/'run.json').read_text());assert run['status']=='complete'
    timing=json.loads((output/'sampling_math_timing.json').read_text()) if method=='cpu' else None
    wall=timing['instrumented_wrapper_process_seconds'] if timing else run['wall_including_normalization_seconds']
    metrics.evaluate(output/'samples.npy',MODEL,output,prefixes=[draws],workers=4,wall_seconds=wall)
    m=json.loads((output/'metrics.json').read_text())
    assert m['solver_detected_nearconstant_reactions_excluded']==478
    passed=(m['split_rhat_median']<=1.01 and m['bulk_ess_median']>=100
            and m['sample_bound_violation_fraction']==0 and m['bounds_tolerance']==1e-7)
    compute=timing['worker_draw_critical_path_seconds'] if timing else run['profile']['timing']['gpu_sampling_s']
    result=dict(method=method,seed=seed,status='PASS' if passed else 'FAIL',quality_met=passed,
        chains=16,retained_draws_per_chain=draws,warmup_per_chain=1000,
        thinning=2000 if method=='cpu' else 1000,sampler_compute_seconds=compute,
        median_split_rhat=m['split_rhat_median'],p95_split_rhat=m['split_rhat_p95'],
        median_bulk_ess=m['bulk_ess_median'],median_ess_per_compute_second=m['bulk_ess_median']/compute,
        sample_bound_violation_fraction=m['sample_bound_violation_fraction'],
        maximum_absolute_bound_violation=m['max_absolute_bound_violation'],process_seconds_secondary=wall,
        diagnostic_mask='conservative478_exact_cached_target',source_run=str(output))
    write(output/'quality.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=int,required=True);a=p.parse_args()
    protocol=json.loads((R/'protocol.json').read_text())
    seed=protocol['seeds'][a.task]
    root=R/f'seed{seed}';root.mkdir(parents=True,exist_ok=True)
    affinity_path=root/'physical_affinity.json'
    if affinity_path.exists():affinity_path=root/f"physical_affinity.retry{os.environ.get('SLURM_JOB_ID','local')}.json"
    write(affinity_path,physical_affinity())
    previous=root/'validation.json'
    if previous.exists() and json.loads(previous.read_text()).get('status')=='failed_backend_or_diagnostics':
        assert not (root/'validation.failed_preflight.json').exists()
        (root/'validation.failed_preflight.json').write_bytes(previous.read_bytes())
    result=dict(seed=seed,status='running',protocol=str(R/'protocol.json'),extension_policy='none')
    try:
        for method in ('cpu','gpu'):
            result[method]=run_case(method,seed,root/method)
            write(root/'validation.json',result)
        result['status']='complete'
    except Exception as error:
        result.update(status='failed_backend_or_diagnostics',error=repr(error));raise
    finally:
        write(root/'validation.json',result)


if __name__=='__main__':main()
