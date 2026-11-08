"""Release parity, failure propagation, export precision and unmasked bounds audit."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

B=Path(__file__).resolve().parents[1]
R=B/'results/closeout/task1'
EXE=B/'build-closeout/naja'
REFERENCE=B/'build-closeout-reference/naja'

def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def run_case(label,model,*,binary=EXE,backmap=False,dtype='float32',backend=None,chains=16,samples=11000,thin=96):
    out=R/label;out.mkdir(parents=True,exist_ok=True)
    command=[str(binary),'sample','run','--model-dir',str(B/'data/models'/model),
        '--out-root',str(out/'native'),'--n-chains',str(chains),'--n-samples',str(samples),
        '--thinning',str(thin),'--tpb','128','--gpu','0','--start-policy','file',
        '--extra-constraints','ignore','--constraint-eps','0','--extra-constraint-eps','0',
        '--pair-prob','0','--bounds-policy','ignore','--barrier-rotate','--write-npy','--quiet']
    if binary==EXE:
        command+=['--seed','101','--sample-dtype',dtype]
        if backend:command+=['--barrier-rotate-backend',backend]
    if backmap:command+=['--backmap']
    marker=out/'execution.json'
    if not marker.exists():
        started=time.perf_counter()
        with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
            result=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=900)
        record=dict(command=command,returncode=result.returncode,wall_seconds=time.perf_counter()-started,
            binary_sha256=sha(binary),utc=datetime.now(timezone.utc).isoformat(),job=os.getenv('SLURM_JOB_ID'))
        save(marker,record)
    record=read(marker);assert record['returncode']==0,(label,record)
    profiles=list((out/'native').rglob('profile.json'));assert len(profiles)==1
    profile=read(profiles[0]);array=profiles[0].parent/'samples.npy'
    x=np.load(array,mmap_mode='r')
    with np.load(B/'data/models'/model/'rounding/polytope.npz') as p:
        dimension=p['T'].shape[0] if backmap else p['A'].shape[1]
    assert x.shape==(chains*samples,dimension),(label,x.shape,dimension)
    assert x.dtype==np.dtype(dtype),(label,x.dtype,dtype)
    value=dict(label=label,model=model,backmap=backmap,dtype=dtype,chains=chains,saved_draws_per_chain=samples,
        retained_draws_per_chain=10000 if samples==11000 else samples,warmup_saved_draws=1000 if samples==11000 else 0,
        array=str(array),shape=list(x.shape),profile=profile,execution=record)
    save(out/'verified_layout.json',value)
    return value

def parity(reference,candidate,cast=False):
    a=np.load(reference['array'],mmap_mode='r');b=np.load(candidate['array'],mmap_mode='r')
    assert a.shape==b.shape
    different=0;maximum=0.;retained_different=0
    first=reference['warmup_saved_draws']*reference['chains']
    for lo in range(0,len(a),4096):
        hi=min(lo+4096,len(a));x=np.asarray(a[lo:hi]);y=np.asarray(b[lo:hi])
        if cast:y=y.astype(x.dtype)
        assert x.dtype==y.dtype and np.isfinite(x).all() and np.isfinite(y).all()
        uint=np.uint32 if x.dtype.itemsize==4 else np.uint64
        mismatch=x.view(uint)!=y.view(uint)
        different+=int(mismatch.sum());maximum=max(maximum,float(np.abs(x.astype(np.float64)-y.astype(np.float64)).max()))
        retained_different+=int(mismatch[max(0,first-lo):].sum())
    return dict(reference=reference['label'],candidate=candidate['label'],counterfactual_cast=cast,
        bitwise_equal=different==0,all_saved_scalar_mismatches=different,retained_scalar_mismatches=retained_different,
        maximum_absolute_coordinate_difference=maximum,compared_saved_shape=list(a.shape),retained_draws_per_chain=reference['retained_draws_per_chain'])

class Bounds:
    def __init__(self):
        self.samples=0;self.coordinates=0;self.bad=0;self.bad_coordinates=0;self.maximum=0.
    def update(self,flux,lower,upper):
        assert np.isfinite(flux).all()
        violation=np.maximum(np.maximum(lower-flux,flux-upper),0.)
        invalid=violation>1e-7
        self.samples+=len(flux);self.coordinates+=flux.size
        self.bad+=int(invalid.any(axis=1).sum());self.bad_coordinates+=int(invalid.sum())
        self.maximum=max(self.maximum,float(violation.max()))
    def result(self):
        return dict(samples_checked=self.samples,reaction_values_checked=self.coordinates,
            tolerance=1e-7,sample_bound_violation_fraction=self.bad/self.samples if self.samples else None,
            coordinate_bound_violation_fraction=self.bad_coordinates/self.coordinates if self.coordinates else None,
            maximum_original_bound_violation=self.maximum,invalid_samples=self.bad)

def bounds(case):
    model=B/'data/models'/case['model']
    with np.load(model/'rounding/polytope.npz') as p:T,shift=p['T'],p['shift'].reshape(-1)
    lower=np.loadtxt(model/'gem/l_bounds.csv',delimiter=',').reshape(-1)
    upper=np.loadtxt(model/'gem/u_bounds.csv',delimiter=',').reshape(-1)
    x=np.load(case['array'],mmap_mode='r');start=case['warmup_saved_draws']*case['chains']
    actual,retained,casted,cast_retained=Bounds(),Bounds(),Bounds(),Bounds()
    for lo in range(0,len(x),2048):
        points=np.asarray(x[lo:lo+2048],dtype=np.float64)
        flux=points if case['backmap'] else points@T.T+shift
        actual.update(flux,lower,upper)
        skip=max(0,start-lo)
        if skip<len(points):retained.update(flux[skip:],lower,upper)
        if case['dtype']=='float64':
            q=points.astype(np.float32).astype(np.float64)
            qflux=q if case['backmap'] else q@T.T+shift
            casted.update(qflux,lower,upper)
            if skip<len(points):cast_retained.update(qflux[skip:],lower,upper)
    result=dict(label=case['label'],all_saved_including_warmup=actual.result(),retained=retained.result(),
        geometry_sha256=sha(model/'rounding/polytope.npz'),lower_bounds_sha256=sha(model/'gem/l_bounds.csv'),
        upper_bounds_sha256=sha(model/'gem/u_bounds.csv'),unmasked=True,no_filtering_or_clipping=True)
    if case['dtype']=='float64':result.update(counterfactual_float32_all_saved=casted.result(),counterfactual_float32_retained=cast_retained.result())
    save(R/case['label']/'original_bounds.json',result)
    return result

def negative_probes():
    rows=[]
    base=[str(EXE),'sample','run','--model-dir',str(B/'data/models/iJO1366'),
        '--n-chains','1','--n-samples','4','--thinning','1','--tpb','128','--seed','101','--write-npy','--quiet']
    cases=[('invalid_gpu',['--gpu','999999']),('invalid_barrier_backend',['--barrier-rotate-backend','invalid']),
        ('invalid_threads',['--tpb','17']),('unsupported_sample_columns',['--n-chains','65536','--n-samples','65536']),
        ('cuda_allocation_failure',['--n-samples','2147483647'])]
    for label,flags in cases:
        out=R/'negative'/label;out.mkdir(parents=True,exist_ok=True)
        command=base+['--out-root',str(out/'native')]+flags
        result=subprocess.run(command,capture_output=True,text=True,timeout=60)
        (out/'stdout.log').write_text(result.stdout);(out/'stderr.log').write_text(result.stderr)
        assert result.returncode!=0,(label,'incorrect success',result.stdout,result.stderr)
        assert not list((out/'native').rglob('samples.npy')),(label,'failed run wrote samples')
        rows.append(dict(label=label,returncode=result.returncode,command=command,stderr=result.stderr))
    save(R/'negative_probes.json',dict(status='passed',cases=rows))

def main():
    assert (R/'tests-passed.txt').exists()
    commit=subprocess.check_output(['git','-C',str(B/'closeout-release'),'rev-parse','HEAD'],text=True).strip()
    assert sha(EXE)==(R/'tests-binary-sha256.txt').read_text().split()[0]
    negative_probes()
    reference=run_case('iJO_reference_seed101','iJO1366',binary=REFERENCE)
    default=run_case('iJO_default_gpu_rotation','iJO1366')
    compatibility=run_case('iJO_compat_cpu_rotation','iJO1366',backend='cpu')
    matches=[parity(reference,default),parity(reference,compatibility)]
    save(R/'iJO_parity.json',dict(comparisons=matches,
        interpretation='Reference is public23873f0 plus the saved one-line constant-seed101 shim. Default GPU A×Q can differ numerically from CPU A×Q; mismatches are measured, never hidden.'))
    recon=[run_case('RECON1_reduced_f32','RECON1'),
        run_case('RECON1_reaction_f32','RECON1',backmap=True),
        run_case('RECON1_reduced_f64','RECON1',dtype='float64'),
        run_case('RECON1_reaction_f64','RECON1',dtype='float64',backmap=True)]
    recon_bounds=[bounds(case) for case in recon]
    casts=[parity(recon[0],recon[2],cast=True),parity(recon[1],recon[3],cast=True)]
    auto=run_case('host_prefault_auto64','iJO1366',dtype='float64',chains=64,samples=4,thin=1)
    effective=auto['profile']['config']['host_prefault_threads'];allowed=sorted(os.sched_getaffinity(0))
    assert 0<effective<=min(8,len(allowed))
    result=dict(status='complete',git_commit=commit,binary_sha256=sha(EXE),reference_binary_sha256=sha(REFERENCE),
        correctness_log=str(R/'gpu-correctness-tests.log'),iJO_parity=matches,
        RECON1_exports=recon_bounds,RECON1_counterfactual_cast_parity=casts,
        host_prefault=dict(chains=64,draws_per_chain=4,effective_threads=effective,allowed_cpu_ids=allowed,
            note='Short allocation/configuration smoke only; not a convergence result.'),
        exact_protocol=dict(seed=101,chains=16,warmup=1000,retained=10000,thinning=96,tpb=128),
        bound_scope='Every original reaction at every saved point, with warmup-inclusive and retained-only results separately, tolerance1e-7. No reaction mask or sample filter.',
        source_sha256=sha(Path(__file__)),updated_utc=datetime.now(timezone.utc).isoformat())
    save(R/'summary.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
