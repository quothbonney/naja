#!/usr/bin/env python3
"""Paired export precision diagnosis on one unchanged GPU trajectory.

Compare T@Z64+shift against T@float64(float32(Z64))+shift. This isolates
export quantization even if the historical frozen-binary trajectory differs.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import time
import numpy as np

B=Path(__file__).resolve().parents[1]
MODEL=B/'data/models/RECON1';OUT=B/'results/precision/RECON1/seed303'
TOLERANCES=[0.,1e-7,1e-6,1e-5,1e-3]

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


class Stats:
    def __init__(self,reactions,active):
        self.active=active;self.count=0;self.worst=np.zeros(reactions)
        self.sample={str(t):0 for t in TOLERANCES};self.active_sample={str(t):0 for t in TOLERANCES}
        self.coordinate={str(t):0 for t in TOLERANCES};self.reduced_worst=0.;self.reduced_samples=0
    def update(self,flux,z,A,b,lower,upper):
        violation=np.maximum(np.maximum(lower-flux,flux-upper),0)
        self.count+=len(flux);self.worst=np.maximum(self.worst,violation.max(axis=0))
        vmax=violation.max(axis=1);amax=violation[:,self.active].max(axis=1)
        for tolerance in TOLERANCES:
            key=str(tolerance);self.sample[key]+=int(np.count_nonzero(vmax>tolerance))
            self.active_sample[key]+=int(np.count_nonzero(amax>tolerance))
            self.coordinate[key]+=int(np.count_nonzero(violation>tolerance))
        reduced=np.maximum(z@A.T-b,0).max(axis=1)
        self.reduced_worst=max(self.reduced_worst,float(reduced.max()))
        self.reduced_samples+=int(np.count_nonzero(reduced>1e-9))
    def result(self,ids):
        order=np.argsort(self.worst)[::-1]
        top=[{'reaction':ids[i],'max_violation':float(self.worst[i]),'lp_mask_excluded':bool(not self.active[i])}
             for i in order[:20] if self.worst[i]>0]
        return {'sample_count':self.count,'sample_bound_violation_fraction':{key:value/self.count for key,value in self.sample.items()},
          'nonexcluded_reaction_sample_violation_fraction':{key:value/self.count for key,value in self.active_sample.items()},
          'coordinate_bound_violation_fraction':{key:value/(self.count*len(ids)) for key,value in self.coordinate.items()},
          'max_absolute_reaction_bound_violation':float(self.worst.max()),
          'max_nonexcluded_reaction_bound_violation':float(self.worst[self.active].max()),
          'max_reduced_Az_minus_b_violation':self.reduced_worst,
          'reduced_sample_violation_fraction_eps_1e9':self.reduced_samples/self.count,'top_violations':top}


def main():
    begun=time.perf_counter()
    newdir=OUT/'f64';olddir=B/'results/naja/RECON1/seed303'
    newrun=json.loads((newdir/'run.json').read_text());oldrun=json.loads((olddir/'run.json').read_text())
    assert newrun['status']=='complete' and oldrun['status']=='complete'
    new=np.load(newdir/'samples.npy',mmap_mode='r');old=np.load(olddir/'samples.npy',mmap_mode='r')
    assert new.dtype==np.float64 and old.dtype==np.float32 and new.shape==old.shape
    assert new.shape[:2]==(16,10000)
    raw64=np.load(newrun['raw_samples'],mmap_mode='r');raw32=np.load(oldrun['raw_samples'],mmap_mode='r')
    assert raw64.shape==raw32.shape and raw64.dtype==np.float64 and raw32.dtype==np.float32
    mismatched_bits=mismatched_values=0;raw_cast_max_difference=0.
    for lo in range(0,len(raw64),2048):
        cast=np.asarray(raw64[lo:lo+2048],dtype=np.float32);reference=np.asarray(raw32[lo:lo+2048])
        mismatched_bits+=int(np.count_nonzero(cast.view(np.uint32)!=reference.view(np.uint32)))
        mismatched_values+=int(np.count_nonzero(cast!=reference))
        raw_cast_max_difference=max(raw_cast_max_difference,float(np.max(np.abs(cast.astype(np.float64)-reference.astype(np.float64)))))
    print('Historical cast comparison:',mismatched_bits,'different bit patterns',flush=True)
    with np.load(MODEL/'rounding/polytope.npz') as p:A,b,T,shift=p['A'],p['b'],p['T'],p['shift']
    lower=np.loadtxt(MODEL/'gem/l_bounds.csv',delimiter=',').reshape(-1)
    upper=np.loadtxt(MODEL/'gem/u_bounds.csv',delimiter=',').reshape(-1)
    ids=(MODEL/'gem/reaction_ids.txt').read_text().splitlines()
    active=np.load(MODEL/'active_reactions.npy');mask=json.loads((MODEL/'diagnostic_mask.json').read_text())
    assert mask['geometry_sha256']==sha(MODEL/'rounding/polytope.npz')
    cases={name:Stats(len(ids),active) for name in ['float64_export','paired_float32_quantization','historical_frozen_float32']}
    max_z_difference=max_flux_difference=0.;normalized_cast_mismatches=0
    for lo in range(0,new.shape[1],256):
        hi=min(lo+256,new.shape[1]);z64=np.asarray(new[:,lo:hi],dtype=np.float64).reshape(-1,new.shape[-1])
        zq=z64.astype(np.float32).astype(np.float64)
        zo=np.asarray(old[:,lo:hi],dtype=np.float64).reshape(z64.shape)
        assert np.isfinite(z64).all() and np.isfinite(zo).all()
        max_z_difference=max(max_z_difference,float(np.max(np.abs(z64-zq))))
        f64=z64@T.T+shift;fq=zq@T.T+shift
        max_flux_difference=max(max_flux_difference,float(np.max(np.abs(f64-fq))))
        cases['float64_export'].update(f64,z64,A,b,lower,upper)
        cases['paired_float32_quantization'].update(fq,zq,A,b,lower,upper)
        if np.array_equal(zo,zq):
            # Same values: statistics are identical, not another independent run.
            fo=fq
        else:
            normalized_cast_mismatches+=int(np.count_nonzero(zo!=zq));fo=zo@T.T+shift
        cases['historical_frozen_float32'].update(fo,zo,A,b,lower,upper)
    results={name:stats.result(ids) for name,stats in cases.items()}
    max64=results['float64_export']['max_absolute_reaction_bound_violation']
    maxq=results['paired_float32_quantization']['max_absolute_reaction_bound_violation']
    if max64<=1e-7 and maxq>1e-7:
        conclusion='Paired reduced-coordinate export quantization creates the material bound violations in this trajectory; float64 stays within1e-7.'
    elif maxq>max64:
        conclusion='Paired export quantization increases bound violations, but full-precision residual violations remain and require separate geometry/slack investigation.'
    else:
        conclusion='This paired experiment does not support export quantization as the explanation for material bound violations.'
    result={'model':'RECON1','seed':303,'started_utc':datetime.now(timezone.utc).isoformat(),
      'causal_protocol':'One GPU float64 trajectory: X64=T@Z64+shift; Xquant=T@float64(float32(Z64))+shift. Both backmaps float64, same matrix and sample points.',
      'primary_tolerance':1e-7,'no_bounds_filtering_or_clipping':True,
      'new_binary_sha256':newrun['exe_sha256'],'historical_binary_sha256':oldrun['exe_sha256'],
      'geometry_sha256':mask['geometry_sha256'],'mask_sha256':sha(MODEL/'active_reactions.npy'),
      'mask_policy':mask['policy'],'preprocessing':json.loads((MODEL/'meta.json').read_text()),
      'shape':list(new.shape),'raw_shape':list(raw64.shape),'new_raw_header_offset':raw64.offset,
      'historical_reproduction':{'raw_cast_bitwise_equal':mismatched_bits==0,'raw_cast_value_equal':mismatched_values==0,
         'raw_bit_mismatches':mismatched_bits,'raw_value_mismatches':mismatched_values,
         'raw_cast_max_absolute_difference':raw_cast_max_difference,'normalized_cast_value_mismatches':normalized_cast_mismatches},
      'max_absolute_reduced_export_rounding_error':max_z_difference,
      'max_absolute_flux_difference_from_paired_quantization':max_flux_difference,
      'T_max_absolute_entry':float(np.max(np.abs(T))),'T_max_row_l1_norm':float(np.max(np.sum(np.abs(T),axis=1))),
      'representations':results,'conclusion':conclusion,
      'new_measured_process_wall_seconds':newrun['wall_seconds'],'new_normalization_seconds':newrun['normalization_seconds'],
      'new_wall_including_normalization_seconds':newrun['wall_including_normalization_seconds'],
      'historical_wall_including_normalization_seconds':oldrun['wall_including_normalization_seconds'],
      'analysis_wall_seconds':time.perf_counter()-begun}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'conclusion':conclusion,'historical_reproduction':result['historical_reproduction'],
          'max_violations':{name:data['max_absolute_reaction_bound_violation'] for name,data in results.items()}},indent=2),flush=True)


if __name__=='__main__':main()
