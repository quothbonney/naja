#!/usr/bin/env python3
"""One pilot-trained Recon3D attempt, at most59 allocated GPUminutes, no retries."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

B=Path(__file__).resolve().parents[1]
R=B/'results/closeout/recon3d_bounded'
sys.path[:0]=[str(B),str(B/'scripts')]
from naja_runner import normalize_samples,backend_errors
def read(path):return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def device_info():
    code='''import ctypes,json,uuid
c=ctypes.CDLL("libcudart.so.12");p=ctypes.create_string_buffer(4096)
c.cudaGetDeviceProperties_v2.argtypes=[ctypes.c_void_p,ctypes.c_int]
assert c.cudaGetDeviceProperties_v2(p,0)==0
c.cudaFree.argtypes=[ctypes.c_void_p];assert c.cudaFree(None)==0
f,t=ctypes.c_size_t(),ctypes.c_size_t();c.cudaMemGetInfo.argtypes=[ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
assert c.cudaMemGetInfo(ctypes.byref(f),ctypes.byref(t))==0
print(json.dumps(dict(uuid="GPU-"+str(uuid.UUID(bytes=p.raw[256:272])),free_bytes=f.value,total_bytes=t.value)))
'''
    return json.loads(subprocess.check_output([str(B/'scripts/enter.sh'),str(B/'env-gpu/bin/python'),'-c',code],text=True,timeout=30))


def native_run(protocol,name,draws,warmup,thin,timeout):
    out=R/name;out.mkdir(exist_ok=False)
    exe=Path(protocol['exe']);assert sha(exe)==protocol['binary_sha256']
    command=[exe,'sample','run','--model-dir',protocol['model'],'--out-root',out/'native',
        '--gpu','0','--n-chains','512','--n-samples',str(draws+warmup),'--seed','101','--tpb','128',
        '--thinning',str(thin),'--constraint-eps','0','--extra-constraint-eps','0',
        '--pair-prob','0','--bounds-policy','ignore','--write-npy','--sample-dtype','float64',
        '--barrier-whiten','--barrier-rotate-backend','gpu','--start-policy','file','--resync-interval','0']
    record=dict(status='running',method='naja',model_dir=protocol['model'],seed=101,chains=512,
        draws=draws,warmup=warmup,thinning=thin,sample_dtype='float64',barrier='whiten',
        command=list(map(str,command)),exe_sha256=protocol['binary_sha256'],source_commit=protocol['source_commit'],
        index_verification='Merged production GPU-wide-index and FP64 host tests; copied test records. Legacy experimental wrapper parity flags are not asserted.',
        started_utc=datetime.now(timezone.utc).isoformat(),samples=str(out/'samples.npy'))
    save(out/'run.json',record);started=time.monotonic()
    try:
        with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
            done=subprocess.run([str(B/'scripts/enter.sh'),*map(str,command)],stdout=stdout,stderr=stderr,
                                timeout=timeout,check=False)
        record['native_process_seconds']=time.monotonic()-started
        assert done.returncode==0,f'Native returncode {done.returncode}'
        assert not backend_errors(out),'Native backend error in logs'
        profiles=list((out/'native').rglob('profile.json'));assert len(profiles)==1
        profile=read(profiles[0]);cfg=profile['config']
        assert cfg['seed']==101 and cfg['sample_dtype']=='float64' and cfg['n_chains']==512
        assert cfg['n_samples']==draws+warmup and cfg['thinning']==thin and cfg['reduced_dim']==4861
        assert not cfg['back_transform'],'Keep native output reduced to fit device memory'
        record.update(status='native_complete',profile=profile,profile_path=str(profiles[0]),
            raw_samples=str(profiles[0].parent/'samples.npy'))
    except Exception as error:
        record.update(status='failed',error=repr(error),native_process_seconds=time.monotonic()-started);raise
    finally:save(out/'run.json',record)
    return record


def normalize(folder,record):
    started=time.monotonic()
    record.update(normalize_samples(Path(record['raw_samples']),folder/'samples.npy',512,
        record['draws'],record['warmup'],expected_dimension=4861,expected_dtype='float64'))
    record.update(status='complete',normalization_seconds=time.monotonic()-started,
        wall_including_normalization_seconds=record['native_process_seconds']+time.monotonic()-started)
    save(folder/'run.json',record)
    return record


def pilot_lag1(path):
    x=np.load(path,mmap_mode='r');values=[]
    for start in range(0,x.shape[2],64):
        block=np.asarray(x[:,:,start:start+64]);center=block.mean(axis=(0,1))
        a=block[:,:-1]-center;b=block[:,1:]-center
        denominator=np.sqrt((a*a).sum(axis=(0,1))*(b*b).sum(axis=(0,1)))
        valid=denominator>0
        values.extend(((a*b).sum(axis=(0,1))[valid]/denominator[valid]).tolist())
    assert values
    return dict(median=float(np.median(values)),p95=float(np.quantile(values,.95)),coordinates=len(values),
        definition='Pooled lag1 over512chains after64 discarded warmup draws; each coordinate uses one pooled mean. Pilot estimate only, not a convergence claim.')


def capture(protocol):
    started=time.monotonic();state=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),
        job=os.environ.get('SLURM_JOB_ID'),protocol_sha256=sha(R/'protocol.json'),source_sha256=sha(Path(__file__)))
    save(R/'capture.json',state)
    try:
        assert sha(Path(protocol['model'])/'rounding/polytope.npz')==protocol['geometry_sha256']
        info=device_info();state['device']=info;save(R/'capture.json',state)
        minimum=protocol['memory']['two_output_buffers_bytes']+protocol['memory']['workspace_reserve_bytes']
        assert info['free_bytes']>=minimum,f'Insufficient device memory: {info["free_bytes"]} < {minimum}'
        pilot=native_run(protocol,'pilot',64,64,810,600)
        normalize(R/'pilot',pilot)
        lag=pilot_lag1(R/'pilot/samples.npy')
        pilot_seconds=pilot['profile']['timing']['gpu_sampling_s'];assert pilot_seconds>0
        rho=float(np.clip(lag['median'],.001,.9999))
        desired=max(810,math.ceil(810*math.log(.1)/math.log(rho)))
        elapsed=time.monotonic()-started
        remaining=3540-elapsed
        main_sampler_budget=min(2700,max(0,remaining-180))
        maximum=math.floor(main_sampler_budget*128*810/(1800*pilot_seconds))
        assert maximum>=1,'No budget for one main attempt'
        thinning=min(desired,maximum)
        prediction=pilot_seconds*(1800*thinning)/(128*810)
        decision=dict(lag1=lag,desired_thinning=desired,budget_capped_thinning=maximum,
            selected_thinning=thinning,pilot_sampler_seconds=pilot_seconds,
            predicted_main_sampler_seconds=prediction,elapsed_allocation_proxy_seconds=elapsed,
            main_sampler_budget_seconds=main_sampler_budget,rule=protocol['thinning_rule'])
        save(R/'pilot_decision.json',decision)
        state['decision']=decision;save(R/'capture.json',state)
        timeout=min(remaining-30,prediction+180)
        main=native_run(protocol,'main',800,1000,thinning,timeout)
        state.update(status='native_capture_complete',main_run=str(R/'main'),
            main_sampler_seconds=main['profile']['timing']['gpu_sampling_s'],
            allocated_gpu_wall_proxy_seconds=time.monotonic()-started,
            note='Main normalization and all-reaction diagnostics run after this GPU allocation ends.')
    except Exception as error:
        state.update(status='failed_or_budget_cap',error=repr(error),allocated_gpu_wall_proxy_seconds=time.monotonic()-started)
    finally:
        state['finished_utc']=datetime.now(timezone.utc).isoformat();save(R/'capture.json',state)


def mass_check(samples,model):
    from scipy import sparse
    with np.load(model/'rounding/polytope.npz') as p:T,shift=p['T'],p['shift'].reshape(-1)
    S=sparse.csr_matrix(np.loadtxt(model/'gem/A_eq.csv',delimiter=','))
    x=np.load(samples,mmap_mode='r').reshape(-1,T.shape[1]);maximum=0.
    for begin in range(0,len(x),2048):
        flux=np.asarray(x[begin:begin+2048])@T.T+shift
        maximum=max(maximum,float(np.abs(S@flux.T).max()))
    return dict(maximum_stoichiometric_residual=maximum,points_checked=len(x),
        method='Every retained point explicitly backmapped in float64 blocks, then multiplied by original sparseS; no bound filtering.')


def diagnose(protocol):
    capture_state=read(R/'capture.json')
    if capture_state['status']!='native_capture_complete':
        save(R/'summary.json',dict(status='no_complete_main_capture',capture=capture_state,scope=protocol['scope']));return
    import metrics
    record=normalize(R/'main',read(R/'main/run.json'))
    started=time.monotonic()
    metric=metrics.evaluate(R/'main/samples.npy',Path(protocol['model']),R/'main',prefixes=[800],workers=4)
    mass=mass_check(R/'main/samples.npy',Path(protocol['model']));save(R/'main/mass_balance.json',mass)
    passed=(metric['split_rhat_median']<=1.01 and metric['bulk_ess_median']>=100
        and metric['sample_bound_violation_fraction']==0 and mass['maximum_stoichiometric_residual']<=1e-7)
    summary=dict(status='PASS' if passed else 'DID_NOT_MEET_TARGET',quality_met=passed,
        source_commit=protocol['source_commit'],binary_sha256=protocol['binary_sha256'],
        n=10600,d=4861,m=18477,chains=512,retained_per_chain=800,warmup_per_chain=1000,
        thinning=record['thinning'],median_split_rhat=metric['split_rhat_median'],p95_split_rhat=metric['split_rhat_p95'],
        median_bulk_ess=metric['bulk_ess_median'],sample_bound_violation_fraction=metric['sample_bound_violation_fraction'],
        maximum_absolute_bound_violation=metric['max_absolute_bound_violation'],mass_balance=mass,
        diagnosed_reactions=metric['diagnosed_reactions'],solver_exclusions=0,
        exactly_constant_sampled_reactions=metric['exactly_constant_sampled_reactions'],
        sampler_seconds=record['profile']['timing']['gpu_sampling_s'],barrier_seconds=record['profile']['timing']['barrier_rotation_s'],
        native_process_wall_seconds=record['native_process_seconds'],
        normalization_seconds=record['normalization_seconds'],cpu_diagnostics_seconds=time.monotonic()-started,
        allocated_gpu_wall_proxy_seconds=capture_state['allocated_gpu_wall_proxy_seconds'],
        preprocessing=protocol['preprocessing'],scope=protocol['scope'],
        criterion=protocol['criterion'],pilot_decision=read(R/'pilot_decision.json'),
        source_metrics=str(R/'main/metrics.json'),finished_utc=datetime.now(timezone.utc).isoformat())
    save(R/'summary.json',summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['capture','metrics'],required=True);a=parser.parse_args()
    protocol=read(R/'protocol.json')
    if a.stage=='capture':capture(protocol)
    else:diagnose(protocol)
