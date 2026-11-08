"""Timed guideline repeats plus explicitly capped thin2000 missing seeds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B))
import metrics
from quality_diagnostics import rhat_precheck

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=int,required=True);a=p.parse_args()
    if a.task<3:
        seed=[101,202,303][a.task]
        reference=B/f'results/larger_quality/iML1515/cpujade/seed{seed}/final'
        q=json.loads((reference/'run.json').read_text());assert q['draws']==500 and q['thinning']==67205
        subprocess.run([str(B/'env-cpu-native/bin/python'),str(B/'scripts/native_physical_pilot.py'),
            '--worker-timing','--model','iML1515','--build','native','--label','physical16_mathinstrumented_native_jade',
            '--seed',str(seed),'--draws','500','--thinning','jadebeck','--reuse-diagnostics-from',str(reference)],check=True)
        out=B/f'results/native_cpu/iML1515/physical16_mathinstrumented_native_jade/seed{seed}'
        current,expected=sha(out/'samples.npy'),sha(reference/'samples.npy')
        proof=dict(status='passed' if current==expected else 'failed',bitwise_equal=current==expected,
            samples_sha256=current,reference_samples_sha256=expected,reference_run=str(reference),instrumented_run=str(out),seed=seed,model='iML1515')
        (out/'worker_timing_parity.json').write_text(json.dumps(proof,indent=2)+'\n');assert current==expected
    else:
        seed=[202,303][a.task-3]
        out=B/f'results/iml_time_to_target/cpu_native_thin2000/seed{seed}'
        out.mkdir(parents=True,exist_ok=False)
        allowed=sorted(os.sched_getaffinity(0));unique={};topology=[]
        for cpu in allowed:
            top=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
            row=dict(cpu=cpu,socket=int((top/'physical_package_id').read_text()),core=int((top/'core_id').read_text()),siblings=(top/'thread_siblings_list').read_text().strip())
            topology.append(row);unique.setdefault((row['socket'],row['core']),cpu)
        assert len(allowed)==32 and len(unique)==16
        os.sched_setaffinity(0,sorted(unique.values()))
        (out/'physical_affinity.json').write_text(json.dumps(dict(reserved_logical_cpus=32,selected_physical_core_count=16,allowed_logical_cpus=allowed,selected_logical_cpus=sorted(unique.values()),topology=topology),indent=2)+'\n')
        sample=out/'fixed10000'
        subprocess.run([str(B/'env-cpu-native/bin/python'),str(B/'scripts/timed_cpu_baseline.py'),'sample',
            '--model',str(B/'data/models/iML1515'),'--out',str(sample),'--seed',str(seed),'--chains','16',
            '--draws','10000','--warmup','1000','--thinning','2000','--workers','16','--chunk','11000','--build-variant','native'],check=True)
        check=rhat_precheck(sample/'samples.npy',B/'data/models/iML1515')
        (sample/'rhat-precheck.json').write_text(json.dumps(metrics.json_safe(check),indent=2)+'\n')
        result=dict(model='iML1515',seed=seed,method='cpu_native_thin2000',draws=10000,warmup=1000,thinning=2000,
            status='fixed_budget_Rhat_fail',quality_met=False,precheck=check,run_dir=str(sample),
            note='Actual fixed10k retained-draw cap. A failure is no claim about eventual convergence; ESS omitted only when Rhat already rejects the AND target.')
        if check['split_rhat_median']<=1.01 and check['sample_bound_violation_fraction']==0:
            t=json.loads((sample/'sampling_math_timing.json').read_text())
            metrics.evaluate(sample/'samples.npy',B/'data/models/iML1515',sample,prefixes=[10000],workers=4,wall_seconds=t['instrumented_wrapper_process_seconds'])
            m=json.loads((sample/'metrics.json').read_text());passed=m['split_rhat_median']<=1.01 and m['bulk_ess_median']>=100 and m['sample_bound_violation_fraction']==0
            result.update(status='PASS' if passed else 'fixed_budget_quality_fail',quality_met=passed,metrics=m)
        (out/'quality.json').write_text(json.dumps(result,indent=2)+'\n')
    subprocess.run([sys.executable,str(B/'scripts/summarize_iml_time_to_target.py')],check=True)
    subprocess.run([sys.executable,str(B/'scripts/summarize_worker_math.py')],check=True)

if __name__=='__main__':main()
