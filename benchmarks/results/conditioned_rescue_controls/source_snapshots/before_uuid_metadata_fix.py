#!/usr/bin/env python3
"""Controlled same-GPU off/rotate pairs for all13 original rotate-only conditions."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

B=Path(__file__).resolve().parents[1]
R=B/'results/conditioned_rescue_controls'
sys.path[:0]=[str(B),str(B/'scripts')]
def read(path):return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def gpu_uuid():
    code='''import ctypes,json,uuid
class U(ctypes.Structure):_fields_=[("bytes",ctypes.c_ubyte*16)]
u=U();c=ctypes.CDLL("libcudart.so.12");c.cudaDeviceGetUuid.argtypes=[ctypes.POINTER(U),ctypes.c_int]
rc=c.cudaDeviceGetUuid(ctypes.byref(u),0);assert rc==0,rc
print(json.dumps({"cuda_device_0_uuid":"GPU-"+str(uuid.UUID(bytes=bytes(u.bytes)))}))
'''
    result=subprocess.check_output([str(B/'scripts/enter.sh'),str(B/'env-gpu/bin/python'),'-c',code],text=True,timeout=30)
    return json.loads(result)['cuda_device_0_uuid']


def verify(case):
    model=Path(case['model'])
    for name,key in [('rounding/polytope.npz','geometry_sha256'),('active_reactions.npy','mask_sha256'),('diagnostic_mask.json','certificate_sha256')]:
        assert sha(model/name)==case[key],f'Frozen target changed: {name}'
    assert read(model/'diagnostic_mask.json')['target_geometry_kind']=='exact_cached_polytope'
    return model


def capture(case,protocol):
    output=R/case['gene'];output.mkdir(parents=True,exist_ok=True)
    state=output/'capture.json'
    if datetime.now(timezone.utc)>=datetime.fromisoformat(protocol['stop_launch_utc']):
        save(state,dict(status='not_launched_cutoff',gene=case['gene']));return
    model=verify(case);exe=B/'build-h100-f64/naja';assert sha(exe)==protocol['binary_sha256']
    identity=gpu_uuid()
    value=dict(status='running',gene=case['gene'],order=case['order'],draws=case['draws'],
        cuda_uuid=identity,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        job=os.environ.get('SLURM_JOB_ID'),protocol_sha256=sha(R/'protocol.json'),
        source_sha256=sha(Path(__file__)),started_utc=datetime.now(timezone.utc).isoformat(),runs=[])
    save(state,value)
    try:
        for mode in case['order']:
            destination=output/mode
            assert not destination.exists(),'Fresh pair output required'
            command=[B/'scripts/enter.sh',B/'env-gpu/bin/python',B/'naja_runner.py','--exe',exe,
                '--model',model,'--out',destination,'--seed','101','--chains','16','--draws',str(case['draws']),
                '--warmup','1000','--thinning','1000','--tpb','512','--barrier',mode,
                '--sample-dtype','float64','--pair-prob','0','--timeout','600']
            with (output/(mode+'.capture.log')).open('w') as log:
                subprocess.run(list(map(str,command)),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=660)
            run=read(destination/'run.json')
            assert run['status']=='complete' and run['seed']==101 and run['draws']==case['draws']
            assert run['barrier']==mode and run['sample_dtype']=='float64' and run['exe_sha256']==protocol['binary_sha256']
            if mode=='off':assert run['profile']['timing']['barrier_rotation_s']==0
            value['runs'].append(dict(mode=mode,command=list(map(str,command)),run=str(destination),
                samples_sha256=sha(destination/'samples.npy'),profile=run['profile'],
                process_seconds_secondary=run['wall_including_normalization_seconds']))
            save(state,value)
        assert gpu_uuid()==identity
        verify(case)
        value.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        value.update(status='failed',error=repr(error));raise
    finally:save(state,value)


def diagnose(case,protocol):
    import metrics
    from conditioned_quality import mass_balance
    output=R/case['gene'];capture_state=read(output/'capture.json')
    assert capture_state['status']=='complete'
    model=verify(case);records=[]
    for mode in ('off','rotate'):
        folder=output/mode;run=read(folder/'run.json')
        metric=metrics.evaluate(folder/'samples.npy',model,folder,prefixes=[case['draws']],workers=4)
        residual=mass_balance(folder/'samples.npy',model)
        seconds=run['profile']['timing']['gpu_sampling_s']
        passed=(metric['split_rhat_median']<=1.05 and metric['bulk_ess_median']>=100
            and metric['sample_bound_violation_fraction']==0 and residual<=1e-7)
        value=dict(gene=case['gene'],mode=mode,seed=101,retained_draws_per_chain=case['draws'],
            quality_met=passed,status='PASS' if passed else 'FAIL',median_split_rhat=metric['split_rhat_median'],
            p95_split_rhat=metric['split_rhat_p95'],median_bulk_ess=metric['bulk_ess_median'],
            sample_bound_violation_fraction=metric['sample_bound_violation_fraction'],
            maximum_absolute_bound_violation=metric['max_absolute_bound_violation'],
            maximum_stoichiometric_residual=residual,sampler_seconds=seconds,
            correction_seconds=run['profile']['timing']['barrier_rotation_s'],
            median_ess_per_sampler_second=metric['bulk_ess_median']/seconds,
            gpu_uuid=capture_state['cuda_uuid'],pair_order=case['order'],run=str(folder),
            original_selected_run=case['original_selected_run'],geometry_sha256=case['geometry_sha256'],
            diagnostic_mask_sha256=case['mask_sha256'],process_seconds_secondary=run['wall_including_normalization_seconds'])
        save(folder/'quality.json',value);records.append(value)
    verify(case)
    save(output/'paired_quality.json',dict(status='complete',gene=case['gene'],rows=records,
        original_category='original_CPU_inherited_cap_Naja_rotate_pass',scope=protocol['scope']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['capture','metrics'],required=True)
    parser.add_argument('--index',type=int,required=True);args=parser.parse_args()
    study=read(R/'protocol.json');entry=study['cases'][args.index]
    if args.stage=='capture':capture(entry,study)
    else:diagnose(entry,study)
