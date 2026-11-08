#!/usr/bin/env python3
"""Full global PolyRound rerounding of the exact inherited conditional system."""
import time
PROCESS_STARTED=time.perf_counter()
import hashlib
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import numpy as np
from scipy.linalg import svdvals
from PolyRound.api import PolyRoundApi
from PolyRound.mutable_classes.polytope import Polytope
from PolyRound.settings import PolyRoundSettings

B=Path(__file__).resolve().parents[1]
R=B/'results/conditioned_prototype/iJO1366'
parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,default=R/'models/shared')
parser.add_argument('--output',type=Path,default=R/'models/global_reround')
parser.add_argument('--initial-coordinate-scale',type=float,default=1e-6)
args=parser.parse_args()
source=args.source.resolve()
out=args.output.resolve()

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')

if (out/'global_reround_proof.json').exists():
    assert json.loads((out/'global_reround_proof.json').read_text())['status']=='passed'
    sys.exit(0)
out.mkdir(parents=True,exist_ok=True)
arrays=np.load(source/'rounding/polytope.npz')
A,b,start,T,shift=[arrays[k].copy() for k in ('A','b','start','T','shift')]
b=b.reshape(-1);shift=shift.reshape(-1)
assert np.min(b-A@start)>0
p=Polytope(A,b)
assert p.A.shape==A.shape
# This is an already reduced, strictly full-dimensional target. A second
# simplify/equality-elimination stage could change its numerical affine hull.
p.apply_shift(start)
assert args.initial_coordinate_scale>0
p.apply_transformation(np.eye(A.shape[1])*args.initial_coordinate_scale)
settings=PolyRoundSettings(backend='glpk',verbose=False,hp_flags={'FeasibilityTol':1e-9})
t=time.perf_counter()
p=PolyRoundApi.round_polytope(p,settings)
round_seconds=time.perf_counter()-t
P=p.transformation.values
q=p.shift.values.reshape(-1)
assert P.shape==(A.shape[1],A.shape[1])
singular=svdvals(P)
assert singular.min()>0
newA=p.A.values;newb=p.b.values.reshape(-1)
expectedA=A@P;expectedb=b-A@q
# Any positive row scaling is harmless; all saved halfspaces must match the
# exact original conditional rows under this one invertible coordinate change.
scales=np.einsum('ij,ij->i',newA,expectedA)/np.einsum('ij,ij->i',expectedA,expectedA)
assert np.all(scales>0)
Aerror=float(np.max(np.abs(newA-scales[:,None]*expectedA)/(1+np.abs(newA))))
berror=float(np.max(np.abs(newb-scales*expectedb)/(1+np.abs(newb))))
assert max(Aerror,berror)<1e-10
assert np.min(newb)>0
newT=T@P
newshift=T@q+shift
(out/'rounding').mkdir(exist_ok=True)
shutil.copytree(source/'gem',out/'gem',dirs_exist_ok=True)
np.savez(out/'rounding/polytope.npz',A=newA,b=newb,start=np.zeros(A.shape[1]),T=newT,shift=newshift)
np.savez(out/'rounding/coordinate_change_proof.npz',P=P,q=q,positive_row_scales=scales)
meta=json.loads((source/'meta.json').read_text())
meta.update(model_id=meta['model_id']+'_exact_shared_global_reround',n=len(shift),d=A.shape[1],m=len(newb),
    preprocessing_variant='full_global_PolyRound_after_uniform_coordinate_scale_same_exact_target',
    initial_coordinate_scale=args.initial_coordinate_scale,
    timings=dict(round_seconds=round_seconds,polyround_seconds=round_seconds,total_seconds=time.perf_counter()-PROCESS_STARTED),
    min_start_slack=float(newb.min()),hostname=platform.node(),command=sys.argv,
    numerical_reduction_policy='No simplify or equality elimination: strict full-dimensional inherited A/b is the declared target.',
    original_shared_geometry_sha256=sha(source/'rounding/polytope.npz'))
save(out/'meta.json',meta)
save(out/'rounding/manifest.json',dict(format_version=2,model_name=out.name,rounding_dim=A.shape[1],n_reactions=len(shift),n_constraints=len(newb)))
with (out/'validation.log').open('w') as log:
    subprocess.run([str(B/'env-cpu/bin/python'),str(B/'scripts/validate_preprocessed.py'),'--model',str(out),'--raw',meta['source']],check=True,stdout=log,stderr=subprocess.STDOUT)
sys.path.insert(0,str(B))
from conditional_model_api import ensure_cached_mask,install_cached_mask
common,report=ensure_cached_mask(source)
report=dict(report,model=out.name,geometry_sha256=sha(out/'rounding/polytope.npz'),
            certificate_geometry_sha256=report['geometry_sha256'],
            exact_target_transport='Reaction LP ranges preserved by this validated square invertible affine map; no second numerical reduction.')
install_cached_mask(out,common,report)
assert np.array_equal(np.load(out/'active_reactions.npy'),common)

proof=dict(status='passed',target='Exact inherited conditional A/b, not an independently reduced raw-model hull',
    dimension=int(A.shape[1]),rows=int(A.shape[0]),source_geometry_sha256=sha(source/'rounding/polytope.npz'),
    destination_geometry_sha256=sha(out/'rounding/polytope.npz'),
    invertible_P_min_singular=float(singular.min()),invertible_P_max_singular=float(singular.max()),
    positive_row_scale_min=float(scales.min()),positive_row_scale_max=float(scales.max()),
    transformed_A_relative_identity_error=Aerror,transformed_b_relative_identity_error=berror,
    common_conditional_mask_identical=True,global_round_algorithm_seconds=round_seconds,
    preparation_inclusive_seconds_before_external_validation=meta['timings']['total_seconds'],
    full_wrapper_inclusive_seconds_with_validation_and_mask=time.perf_counter()-PROCESS_STARTED,
    timing_scope='Global MVE round phase primary; complete preparation before external validation secondary; validation/LP-mask are external diagnostic costs.',
    source_script_sha256=sha(Path(__file__)),no_target_relaxation=True,no_second_affine_hull_reduction=True,
    initial_coordinate_scale=args.initial_coordinate_scale,
    dimension_gate_note='Default PolyRound dimension gate applied in uniformly rescaled coordinates. Original failed default attempt is retained; no halfspace or target changed.',
    resources={k:v for k,v in os.environ.items() if k in ('SLURM_CPUS_PER_TASK','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')})
save(out/'global_reround_proof.json',proof)
print(json.dumps(proof,indent=2),flush=True)
