"""Additional frozen-protocol model quality searches; isolated from canonical runs."""
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
import metrics
from quality import qualifies
from quality_diagnostics import rhat_precheck,save_identity,reuse

MODELS=['iAF1260','iML1515','iMM904']
SEEDS=[101,202,303]
GRID=[100,250,500,1000,2000,5000,10000]

def write(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(metrics.json_safe(x),indent=2,allow_nan=False)+'\n');t.replace(p)
def read(p):return json.loads(p.read_text())

def pin(out):
    allowed=sorted(os.sched_getaffinity(0));top=[];by_socket={}
    for cpu in allowed:
        p=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        top.append(dict(cpu=cpu,socket=int((p/'physical_package_id').read_text()),core=int((p/'core_id').read_text()),siblings=(p/'thread_siblings_list').read_text().strip()))
    unique={}
    for x in top:unique.setdefault((x['socket'],x['core']),x['cpu'])
    for (socket,core),cpu in sorted(unique.items()):by_socket.setdefault(socket,[]).append(cpu)
    chosen=[]
    while len(chosen)<16 and any(by_socket.values()):
        for s in sorted(by_socket):
            if by_socket[s] and len(chosen)<16:chosen.append(by_socket[s].pop(0))
    assert len(allowed)>=32 and len(chosen)==16
    os.sched_setaffinity(0,chosen)
    write(out/'physical_affinity.json',dict(reserved_logical_cpus=int(os.environ['SLURM_CPUS_PER_TASK']),allowed_logical_cpus=allowed,topology=top,selected_logical_cpus=chosen,selected_physical_core_count=16,selection='one thread per unique physical core, round robin sockets'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=int,required=True);p.add_argument('--method',choices=['gpu','cpu2000','cpujade'],required=True);a=p.parse_args()
    model=MODELS[a.task//3];seed=SEEDS[a.task%3];method=a.method
    out=B/'results/larger_quality'/model/method/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'quality.json').exists() and read(out/'quality.json').get('status') in ['complete','pilot_cap_no_passing_rhat','pilot_cap_no_passing_quality']:return
    geo=B/'data/models'/model
    thin=1000 if method=='gpu' else 2000 if method=='cpu2000' else 'jadebeck'
    result=dict(model=model,method=method,seed=seed,chains=16,warmup=1000,thinning=thin,tpb=512 if method=='gpu' else None,status='running',started_utc=datetime.now(timezone.utc).isoformat(),note='Actual fresh selected-budget process time excludes pilot/search/diagnostics. Median split Rhat<=1.01 and bulk ESS>=100; bound metrics are independent. Fixed retained warmup1000 implies differing warmup moves.')
    if method!='gpu':pin(out)
    exe=B/'build-h100/naja'
    expected='c7baf4f0ed282b936ac0dbac4b896bfbfc040eb06da6f84aff6605fddf4b0fc6'
    if method=='gpu':assert hashlib.file_digest(exe.open('rb'),'sha256').hexdigest()==expected
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if method!='gpu':env.update(HOPSY_CXXFLAGS='-O3 -march=native -mtune=native',HOPSY_BUILD_MANIFEST=str(B/'audit/hopsy_native_build.json'))
    def sample(label,draws):
        d=out/label
        if (d/'run.json').exists() and read(d/'run.json').get('status')=='complete':return d,read(d/'run.json')
        if method=='gpu':
            cmd=['bash',str(B/'scripts/enter.sh'),'python',str(B/'naja_runner.py'),'--exe',str(exe),'--model',str(geo),'--out',str(d),'--seed',str(seed),'--chains','16','--draws',str(draws),'--warmup','1000','--tpb','512','--thinning',str(thin),'--timeout','1200','--barrier','rotate','--pair-prob','0']
        else:
            cmd=[str(B/'env-cpu-native/bin/python'),str(B/'cpu_baseline.py'),'sample','--model',str(geo),'--out',str(d),'--seed',str(seed),'--chains','16','--draws',str(draws),'--warmup','1000','--thinning',str(thin),'--workers','16','--chunk',str(draws+1000),'--build-variant','native']
        with (out/f'{label}-sampling.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1250)
        r=read(d/'run.json');assert r['status']=='complete'
        return d,r
    write(out/'quality.json',result)
    try:
        # Existing official-wheel Jadebeck prefix evidence selects a cheap native screening budget;
        # native output is always freshly validated before it can contribute a result.
        pilot_draws=10000
        if method=='cpujade':
            src=B/'results/hopsy'/model/'jadebeck'/f'seed{seed}'/'metrics.json';m=read(src)
            assert m['schema_version']==2 and m['diagnostic_mask_status']=='model_lp_verified'
            pilot_draws=m['first_quality_prefix_draws'] or 10000
            result['screening_budget_source']=str(src)
        start=time.perf_counter();pilot,r=sample('pilot',pilot_draws)
        prefixes=[n for n in GRID if n<=pilot_draws]
        checks=[rhat_precheck(pilot/'samples.npy',geo,draws_limit=n) for n in prefixes];write(pilot/'rhat-precheck.json',dict(prefixes=checks))
        eligible=[n for n,c in zip(prefixes,checks) if c['split_rhat_median'] is not None and c['split_rhat_median']<=1.0100000001]
        result['pilot_process_seconds']=r.get('wall_including_normalization_seconds',r.get('process_wall_seconds'))
        if not eligible:result['status']='pilot_cap_no_passing_rhat'
        else:
            metrics.evaluate(pilot/'samples.npy',geo,pilot,prefixes=eligible);save_identity(pilot/'samples.npy',geo,pilot)
            passing=[x['draws_per_chain'] for x in read(pilot/'metrics.json')['prefix_quality'] if x['quality_met']]
            if not passing:result['status']='pilot_cap_no_passing_quality'
            else:
                selected=min(passing);result['selected_draws']=selected;result['pilot_and_selection_seconds']=time.perf_counter()-start;write(out/'quality.json',result)
                final,fr=sample('final',selected);wall=fr.get('wall_including_normalization_seconds',fr.get('process_wall_seconds'))
                if not (selected==pilot_draws and reuse(final/'samples.npy',geo,final,wall,pilot)):metrics.evaluate(final/'samples.npy',geo,final,prefixes=[selected])
                save_identity(final/'samples.npy',geo,final);fm=read(final/'metrics.json')
                result['status']='complete' if qualifies(fm) else 'selected_budget_failed_revalidation'
                result['measured']=dict(draws_per_chain=selected,wall_seconds=wall,split_rhat_median=fm['split_rhat_median'],bulk_ess_median=fm['bulk_ess_median'],quality_met=qualifies(fm),sample_bound_violation_fraction=fm['sample_bound_violation_fraction'],max_absolute_bound_violation=fm['max_absolute_bound_violation'],bounds_tolerance=fm['bounds_tolerance'],run_dir=str(final))
    except Exception as e:result.update(status='failed',error=repr(e));write(out/'quality.json',result);raise
    result['finished_utc']=datetime.now(timezone.utc).isoformat();write(out/'quality.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
