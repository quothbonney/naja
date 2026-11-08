#!/usr/bin/env python3
"""Optional full-node128-logical-worker retime, guarded by whole-stream identity."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

B=Path(__file__).resolve().parents[1]
R=B/'results/high_throughput/iJO1366'
def read(path):return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=101001);a=p.parse_args()
    reference=next(row for row in read(R/'summary.json')['groups']['cpu']['rows'] if row['seed']==a.seed)
    source=Path(reference['run_dir'])
    output=R/'cpu_native512_logical128_thin2000'/f'seed{a.seed}'
    output.mkdir(parents=True,exist_ok=False)
    allowed=sorted(os.sched_getaffinity(0));topology=[]
    for cpu in allowed:
        t=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        topology.append(dict(cpu=cpu,socket=int((t/'physical_package_id').read_text()),
            core=int((t/'core_id').read_text()),siblings=list(map(int,(t/'thread_siblings_list').read_text().strip().split(',')))))
    assert len(allowed)==128 and len({(r['socket'],r['core']) for r in topology})==64
    assert int(os.environ['SLURM_CPUS_PER_TASK'])==128
    assert all(set(row['siblings'])<=set(allowed) for row in topology)
    write(output/'physical_affinity.json',dict(job=os.environ.get('SLURM_JOB_ID'),allowed=allowed,
        topology=topology,physical_cores=64,logical_cpus=128,requested_workers=128))
    command=[B/'env-cpu-native/bin/python',B/'scripts/timed_cpu_baseline.py','sample',
        '--model',B/'data/models/iJO1366','--out',output,'--seed',str(a.seed),'--chains','512',
        '--draws','12000','--warmup','1000','--thinning','2000','--workers','128','--chunk','13000','--build-variant','native']
    subprocess.run(list(map(str,command)),check=True)
    began=time.perf_counter()
    expected=read(source/'diagnostics_identity.json')['data']['samples_sha256']
    actual=sha(output/'samples.npy')
    proof=dict(status='passed' if actual==expected else 'failed',source_run=str(source),
        source_expected_sha256=expected,actual_retained_sha256=actual,
        verification_seconds=time.perf_counter()-began,
        note='Whole retained-file identity against verified64-physical-worker run; same-seed retime, not a new RNG replicate.')
    write(output/'shared_stream_identity.json',proof)
    assert proof['status']=='passed', 'Require independent diagnostics if stream identity fails'
    metric=read(R/'cached_target_aggregate/cpu'/f'seed{a.seed}'/'metrics.json')
    assert Path(metric['samples']).resolve()==(source/'samples.npy').resolve()
    assert metric['metrics_code_sha256']==sha(B/'metrics.py')
    assert metric['solver_detected_nearconstant_reactions_excluded']==478
    assert metric['split_rhat_median']<=1.01 and metric['bulk_ess_median']>=100
    assert metric['sample_bound_violation_fraction']==0 and metric['bounds_tolerance']==1e-7
    timing=read(output/'sampling_math_timing.json')
    compute=timing['worker_draw_critical_path_seconds']
    result=dict(status='passed',seed=a.seed,chains=512,retained_per_chain=12000,warmup=1000,thinning=2000,
        requested_workers=128,observed_worker_count=timing['worker_count'],physical_cores=64,logical_cpus=128,
        worker_draw_critical_path_seconds=compute,
        inclusive_sampler_seconds=read(output/'run.json')['timings']['sample_seconds'],
        process_seconds_secondary=timing['instrumented_wrapper_process_seconds'],
        median_split_rhat=metric['split_rhat_median'],median_bulk_ess=metric['bulk_ess_median'],
        sample_bound_violation_fraction=metric['sample_bound_violation_fraction'],
        median_ess_per_compute_second=metric['bulk_ess_median']/compute,
        source_run=str(source),source_conservative_metrics=str(R/'cached_target_aggregate/cpu'/f'seed{a.seed}'/'metrics.json'),
        identity_proof=str(output/'shared_stream_identity.json'),finished_utc=datetime.now(timezone.utc).isoformat())
    write(output/'smt_quality.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
