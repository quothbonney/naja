"""Compare batched versus flattened backmap BLAS calls on real iJO1366 geometry."""
import json
import os
from pathlib import Path
import statistics
import time
import numpy as np

root=Path(__file__).resolve().parents[1]
with np.load(root/'data/models/iJO1366/rounding/polytope.npz') as f:
    T=f['T'][:16].T
rng=np.random.default_rng(123)
x=rng.normal(size=(16,256,T.shape[0]))
methods={'batched_3d':lambda:x@T,
         'flattened_2d':lambda:(x.reshape(-1,x.shape[-1])@T).reshape(16,256,-1)}
np.testing.assert_allclose(methods['batched_3d'](),methods['flattened_2d'](),rtol=1e-12,atol=1e-12)
result={'hostname':os.uname().nodename,'block_shape':list(x.shape),'T_shape':list(T.shape),
        'openblas_threads':os.environ.get('OPENBLAS_NUM_THREADS'),'repetitions':100}
for name,call in methods.items():
    call(); values=[]
    for _ in range(100):
        start=time.perf_counter();call();values.append(time.perf_counter()-start)
    result[name+'_median_seconds']=statistics.median(values)
result['flat_speedup']=result['batched_3d_median_seconds']/result['flattened_2d_median_seconds']
print(json.dumps(result,indent=2))
# Diagnostic workload comparison uses 64 active and 64 LP-masked real reactions.
import sys
sys.path.insert(0,str(root))
from metrics import diagnostics
model=root/'data/models/iJO1366'
mask=np.load(model/'active_reactions.npy')
indices=np.r_[np.flatnonzero(mask)[:64],np.flatnonzero(~mask)[:64]]
with np.load(model/'rounding/polytope.npz') as f:
    T,shift=f['T'][indices],f['shift'][indices]
z=np.load(root/'results/naja/iJO1366/seed101/samples.npy',mmap_mode='r')[:,:1000]
flux=(np.asarray(z,dtype=np.float64).reshape(-1,z.shape[-1])@T.T+shift).reshape(z.shape[0],z.shape[1],-1)
variable=np.ptp(flux,axis=(0,1))>0
budgets=(100,250,500,1000)
# First warm the common ArviZ/Numba path.
diagnostics(flux[:,:100,variable][:,:,:1])
values={}
for optimized in (False,True):
    start=time.perf_counter()
    for n in budgets:
        selected=variable & (mask[indices] if optimized and n<1000 else True)
        rhat,ess=diagnostics(flux[:,:n,selected])
    values['optimized' if optimized else 'prior']=time.perf_counter()-start
print(json.dumps({'diagnostic_subset_reactions':len(indices),'model_masked':int((~mask[indices]).sum()),
                  'prefix_budgets':budgets,'diagnostic_seconds':values,
                  'diagnostic_speedup':values['prior']/values['optimized']},indent=2))
