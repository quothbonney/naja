"""Quality pilot short-circuit and strict, same-budget diagnostics cache."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def identity(samples,model):
    samples,model=Path(samples),Path(model)
    x=np.load(samples,mmap_mode='r')
    return {'estimator_code_sha256':getattr(sys.modules.get('metrics'),'_METRICS_CODE_SHA256',None),'numpy_version':np.__version__,'arviz_version':importlib.metadata.version('arviz'),
      'samples_sha256':sha(samples),'shape':list(x.shape),'dtype':x.dtype.str,
      'geometry_sha256':sha(model/'rounding/polytope.npz'),
      'mask_sha256':sha(model/'active_reactions.npy'),
      'mask_policy_sha256':sha(model/'diagnostic_mask.json'),
      'lower_bounds_sha256':sha(model/'gem/l_bounds.csv'),
      'upper_bounds_sha256':sha(model/'gem/u_bounds.csv'),
      'reaction_ids_sha256':sha(model/'gem/reaction_ids.txt')}


def save_identity(samples,model,output):
    output=Path(output)
    m=json.loads((output/'metrics.json').read_text())
    if m.get('schema_version')!=2 or m.get('diagnostic_mask_status')!='model_lp_verified':
        raise ValueError('Only full, verified schema2 diagnostics may enter cache')
    loaded_sha=getattr(sys.modules.get('metrics'),'_METRICS_CODE_SHA256',None)
    if not loaded_sha or m.get('metrics_code_sha256')!=loaded_sha:
        return False # Older results remain valid but are not eligible for strict reuse.
    result={'kind':'full_schema2','data':identity(samples,model),
            'metrics_sha256':sha(output/'metrics.json')}
    (output/'diagnostics_identity.json').write_text(json.dumps(result,indent=2)+'\n')
    return True


def reuse(samples,model,output,wall_seconds,pilot):
    """Accept only exact file bytes + dtype/shape + geometry and mask identity."""
    begun=time.perf_counter()
    output,pilot=Path(output),Path(pilot)
    proofpath=pilot/'diagnostics_identity.json'
    if not proofpath.exists(): return False
    cached=json.loads(proofpath.read_text())
    loaded_sha=getattr(sys.modules.get('metrics'),'_METRICS_CODE_SHA256',None)
    producer=json.loads((pilot/'metrics.json').read_text())
    if not loaded_sha or producer.get('metrics_code_sha256')!=loaded_sha or cached.get('data',{}).get('estimator_code_sha256')!=loaded_sha:
        return False
    if cached.get('kind')!='full_schema2' or sha(pilot/'metrics.json')!=cached.get('metrics_sha256'):
        return False
    pilot_now=identity(pilot/'samples.npy',model)
    final=identity(samples,model)
    if cached['data']!=pilot_now or pilot_now!=final:
        (output/'cache_rejection.json').write_text(json.dumps({'cached':cached['data'],'current_pilot':pilot_now,'final':final},indent=2)+'\n')
        return False
    m=json.loads((pilot/'metrics.json').read_text())
    if m.get('schema_version')!=2 or m.get('diagnostic_mask_status')!='model_lp_verified': return False
    m['cached_source_metrics_wall_seconds']=m.get('metrics_wall_seconds')
    m['metrics_wall_seconds']=time.perf_counter()-begun
    m.update(samples=str(Path(samples).resolve()),wall_seconds=wall_seconds,
             samples_per_second=final['shape'][0]*final['shape'][1]/wall_seconds,
             diagnostics_reused_from=str((pilot/'metrics.json').resolve()),
             diagnostics_reuse_note='Fresh same-seed process timing; identical analysis-file bytes verified. Diagnostics are reused, not independent random samples.',
             estimated_time_to_quality_seconds=None,
             time_to_quality_note='No interpolated estimate; actual exact-budget time is recorded in quality.json.')
    (output/'metrics.json').write_text(json.dumps(m,indent=2,allow_nan=False)+'\n')
    if (pilot/'reaction_metrics.csv').exists(): shutil.copy2(pilot/'reaction_metrics.csv',output/'reaction_metrics.csv')
    (output/'diagnostics_identity_proof.json').write_text(json.dumps({'cached_metrics_sha256':cached['metrics_sha256'],
      'pilot_analysis_file':str((pilot/'samples.npy').resolve()),'final_analysis_file':str(Path(samples).resolve()),
      'exact_file_bytes_shape_dtype_geometry_mask_verified':True,'pilot_identity':pilot_now,'final_identity':final},indent=2)+'\n')
    return True


def rhat_precheck(samples,model,tolerance=1e-7,draws_limit=None):
    """Streaming Welford moments reproduce classical split Rhat without ESS.

One dense backmap per draw block avoids repeating many narrow GEMMs. All
reaction bounds are checked on actual retained samples (all current budgets
are even; for odd budgets the middle draw is excluded as in split Rhat).
"""
    started=time.perf_counter();samples,model=Path(samples),Path(model)
    x=np.load(samples,mmap_mode='r')
    if draws_limit is not None: x=x[:,:draws_limit]
    chains,draws,d=x.shape;half=draws//2
    with np.load(model/'rounding/polytope.npz') as p: T,shift=p['T'],p['shift']
    report=json.loads((model/'diagnostic_mask.json').read_text())
    if report['geometry_sha256']!=sha(model/'rounding/polytope.npz'): raise ValueError('Mask geometry mismatch')
    active=np.load(model/'active_reactions.npy')
    lb=np.loadtxt(model/'gem/l_bounds.csv',delimiter=',').reshape(-1)
    ub=np.loadtxt(model/'gem/u_bounds.csv',delimiter=',').reshape(-1)
    means=np.zeros((2*chains,len(shift)));m2=np.zeros_like(means)
    minimum=np.full(len(shift),np.inf);maximum=np.full(len(shift),-np.inf)
    invalid=exact_invalid=coordinate_invalid=checked=0;worst=0.
    for h,begin in [(0,0),(1,draws-half)]:
        nold=0
        for start in range(begin,begin+half,256):
            end=min(start+256,begin+half);count=end-start
            z=np.asarray(x[:,start:end],dtype=np.float64).reshape(chains*count,d)
            f=(z@T.T+shift).reshape(chains,count,len(shift))
            if not np.isfinite(f).all(): raise ValueError('Nonfinite samples/backmap')
            batch_mean=f.mean(axis=1)
            centered=f-batch_mean[:,None,:]
            batch_m2=np.sum(centered*centered,axis=1)
            sl=slice(h*chains,(h+1)*chains)
            delta=batch_mean-means[sl]
            m2[sl]+=batch_m2+delta*delta*(nold*count/(nold+count))
            means[sl]+=delta*(count/(nold+count));nold+=count
            minimum=np.minimum(minimum,f.min(axis=(0,1)));maximum=np.maximum(maximum,f.max(axis=(0,1)))
            violation=np.maximum(np.maximum(lb-f,f-ub),0)
            per_sample=violation.max(axis=-1)
            invalid+=int(np.count_nonzero(per_sample>tolerance));exact_invalid+=int(np.count_nonzero(per_sample>0))
            coordinate_invalid+=int(np.count_nonzero(violation>tolerance));checked+=chains*count
            worst=max(worst,float(violation.max()))
    active=active&(maximum>minimum)
    within=(m2/(half-1)).mean(axis=0)
    between=half*means.var(axis=0,ddof=1)
    with np.errstate(divide='ignore',invalid='ignore'):
        values=np.sqrt(((half-1)/half*within+between/half)/within)[active]
    values=values[~np.isnan(values)]
    median=float(np.median(values)) if len(values) else None
    return {'schema_version':2,'diagnostic_mask_status':'model_lp_verified','shape':list(x.shape),
      'stage':'masked_classical_split_rhat_precheck','split_rhat_median':median,
      'split_rhat_p95':float(np.quantile(values,.95)) if len(values) else None,
      'bulk_ess_median':None,'bulk_ess_computed':False,'diagnosed_reactions':int(len(values)),
      'sample_bound_violation_fraction':invalid/checked,'bounds_tolerance':tolerance,
      'coordinate_bound_violation_fraction':coordinate_invalid/(checked*len(shift)),
      'zero_tolerance_sample_bound_violation_fraction':exact_invalid/checked,
      'max_absolute_bound_violation':worst,'checked_samples':checked,
      'precheck_wall_seconds':time.perf_counter()-started}
