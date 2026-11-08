#!/usr/bin/env python3
"""Empirical marginal/correlation agreement with independently seeded references.

This is an agreement check, not a proof of a uniform stationary distribution.
Backmaps use float64 and write reaction-major disk arrays for bounded RAM use.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/uniformity'
MODEL=ROOT/'data/models/e_coli_core'
METHODS=['hopsy1201','hopsy2401','naja_rotate','naja_whiten','naja_whiten_dikin_pair']


def backmap(name,T,shift,lb,ub):
    directory=OUT/name
    record=json.loads((directory/'run.json').read_text())
    if record['status']!='complete':
        raise ValueError(f'{name}: sampling is not complete')
    z=np.load(directory/'samples.npy',mmap_mode='r')
    if z.ndim!=3:
        raise ValueError(f'{name}: expected chain,draw,d')
    flat=z.reshape(-1,z.shape[-1])
    path=directory/'reaction_major_float64.npy'
    f=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=(T.shape[0],flat.shape[0]))
    invalid=0
    max_violation=0.
    for start in range(0,len(flat),8192):
        end=min(start+8192,len(flat))
        block=np.asarray(flat[start:end],dtype=np.float64)@T.T+shift
        f[:,start:end]=block.T
        violation=np.maximum(np.maximum(lb-block,block-ub),0).max(axis=1)
        invalid+=int(np.count_nonzero(violation>1e-6))
        max_violation=max(max_violation,float(violation.max()))
    f.flush()
    means=np.mean(f,axis=1)
    variance=np.zeros(len(means))
    for start in range(0,f.shape[1],8192):
        centered=np.asarray(f[:,start:start+8192])-means[:,None]
        variance+=np.sum(centered*centered,axis=1)
    variance/=f.shape[1]
    return f,means,variance,dict(sample_count=f.shape[1],dtype='float64',
               shape=list(z.shape),sample_bounds_violation_fraction_eps_1e6=invalid/f.shape[1],
               max_reaction_bound_violation=max_violation,
               command=record.get('command'),seed=record.get('seed'))


def correlation(f,mean,active):
    indices=np.flatnonzero(active)
    cov=np.zeros((len(indices),len(indices)))
    for start in range(0,f.shape[1],8192):
        block=np.asarray(f[indices,start:start+8192])-mean[indices,None]
        cov+=block@block.T
    std=np.sqrt(np.diag(cov))
    return cov/(std[:,None]*std[None,:])


def summary(values):
    return dict(median=float(np.median(values)),p95=float(np.quantile(values,.95)),
                max=float(np.max(values)))


def main():
    begun=time.perf_counter()
    with np.load(MODEL/'rounding/polytope.npz') as bundle:
        T,shift=bundle['T'],bundle['shift']
    lb=np.loadtxt(MODEL/'gem/l_bounds.csv',delimiter=',')
    ub=np.loadtxt(MODEL/'gem/u_bounds.csv',delimiter=',')
    ids=(MODEL/'gem/reaction_ids.txt').read_text().splitlines()
    mapped={}
    for name in METHODS:
        print(f'Backmapping {name}',flush=True)
        mapped[name]=backmap(name,T,shift,lb,ub)
    _,m1,v1,_=mapped[METHODS[0]]
    _,m2,v2,_=mapped[METHODS[1]]
    scale=np.sqrt((v1+v2)/2+(m1-m2)**2/4)
    # Use the same model-derived LP mask as the shared quality diagnostics.
    mask_report=json.loads((MODEL/'diagnostic_mask.json').read_text())
    geometry_hash=hashlib.sha256((MODEL/'rounding/polytope.npz').read_bytes()).hexdigest()
    if mask_report['geometry_sha256']!=geometry_hash:
        raise ValueError('Diagnostic mask geometry mismatch')
    model_active=np.load(MODEL/'active_reactions.npy')
    if model_active.dtype!=bool or model_active.shape!=scale.shape:
        raise ValueError('Invalid cached model mask')
    active=model_active&(scale>0)
    indices=np.flatnonzero(active)
    correlations={name:correlation(data[0],data[1],active) for name,data in mapped.items()}
    comparisons={}
    for name in METHODS[1:]:
        print(f'Comparing {name} against hopsy1201',flush=True)
        reference=mapped['hopsy1201'][0]
        candidate=mapped[name][0]
        if reference.shape!=candidate.shape:
            raise ValueError('Exact empirical W1 requires equal retained sample counts')
        w1=[]
        for reaction in indices:
            left=np.sort(np.asarray(reference[reaction]))
            right=np.sort(np.asarray(candidate[reaction]))
            w1.append(float(np.mean(np.abs(left-right))))
        w1=np.asarray(w1)
        distance=float(np.linalg.norm(correlations[name]-correlations['hopsy1201'],'fro'))
        comparisons[name]={
            'interpretation':'independent-reference floor' if name=='hopsy2401' else 'empirical agreement',
            'raw_wasserstein1':summary(w1),
            'wasserstein1_divided_by_pooled_reference_std':summary(w1/scale[indices]),
            'correlation_frobenius_distance':distance,
            'correlation_frobenius_per_active_reaction':distance/len(indices),
            'per_reaction':[{'reaction':ids[r], 'wasserstein1':float(w),
                             'pooled_reference_std':float(scale[r]),
                             'normalized_wasserstein1':float(w/scale[r])}
                            for r,w in zip(indices,w1)]}
    result={'description':'Empirical agreement on e_coli_core; does not prove uniformity.',
            'model':'e_coli_core',
            'model_bundle_sha256':hashlib.sha256((MODEL/'rounding/polytope.npz').read_bytes()).hexdigest(),
            'protocol':{'chains':16,'draws_per_chain':62500,'total_retained_per_method':1000000,
                        'warmup_retained_draws_per_chain':1000,'reference_thinning':96,
                        'naja_rotate':'rotate + adaptive thinning, coordinate-only',
                        'naja_whiten':'rotate+whiten + fixed pair probability .5, adaptive thinning',
                        'naja_whiten_dikin_pair':'rotate+whiten + barrier Jacobi pair probability .5 + verified Dikin probability .15, adaptive thinning; no GPU backmap',
                        'backmap':'float64 T*z+shift; Naja native stored samples are float32',
                        'wasserstein1':'mean absolute difference of sorted empirical samples; equal sample counts',
                        'normalization':'pooled reference population standard deviation, including between-reference mean difference',
                        'active_reactions':'cached model-derived LP-verified active_reactions.npy mask and pooled reference std>0',
                        'diagnostic_mask_geometry_sha256':geometry_hash,
                        'diagnostic_mask_source_sha256':mask_report['source_sha256'],
                        'diagnostic_mask_policy':mask_report['policy'],
                        'correlation_normalization':'Frobenius norm divided by number of active reactions (entrywise RMS)',
                        'reference_floor':'hopsy2401 vs hopsy1201, separate random seeds'},
            'active_reaction_count':len(indices),'excluded_fixed_reaction_count':len(ids)-len(indices),
            'excluded_reactions':[ids[i] for i in np.flatnonzero(~active)],
            'runs':{name:data[3] for name,data in mapped.items()},
            'comparisons':comparisons,'analysis_wall_seconds':time.perf_counter()-begun}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({name:{k:v for k,v in data.items() if k!='per_reaction'}
                      for name,data in comparisons.items()},indent=2),flush=True)


if __name__=='__main__':
    main()
