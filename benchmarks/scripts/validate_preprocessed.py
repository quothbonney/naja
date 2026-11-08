#!/usr/bin/env python3
"""Check a rounded bundle against its unchanged original BiGG JSON model."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import cobra
import hopsy
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',required=True)
p.add_argument('--raw',required=True)
p.add_argument('--output')
p.add_argument('--tolerance',type=float,default=1e-7)
args=p.parse_args()
root=Path(args.model)
meta=json.loads((root/'meta.json').read_text())
cobra.Configuration().solver='glpk'
original=cobra.io.load_json_model(args.raw)
S=cobra.util.array.create_stoichiometric_matrix(original,array_type='lil').tocsr()
ids=[r.id for r in original.reactions]
lb=np.array([r.lower_bound for r in original.reactions])
ub=np.array([r.upper_bound for r in original.reactions])
data=np.load(root/'rounding/polytope.npz')
A,b,start,T,shift=[data[k] for k in ('A','b','start','T','shift')]
errors=[]
pruning=meta.get('zero_pruning') or {}
pruned=[ids.index(r) for r in pruning.get('zero_pruned_reactions',[])]
pruned_bound=0.
if pruned:
    if np.count_nonzero(T[pruned]) or np.count_nonzero(shift[pruned]):
        errors.append('pruned zero rows not restored as exact zeros')
    pruned_bound=float(max(0,np.max(lb[pruned]),np.max(-ub[pruned])))
    if pruned_bound>1e-9: errors.append('pruned zeros exceed declared 1e-9 original-bound precision')
if any(not np.isfinite(v).all() for v in (A,b,start,T,shift)):
    errors.append('nonfinite array')
if T.shape != (len(ids),A.shape[1]): errors.append('backmap shape')
if (root/'gem/reaction_ids.txt').read_text().splitlines()!=ids:
    errors.append('reaction order mismatch')
raw_sha=hashlib.sha256(Path(args.raw).read_bytes()).hexdigest()
if raw_sha!=meta['source_sha256']: errors.append('source SHA mismatch')
slack=b-A@start
if slack.min()<=0: errors.append('start not strictly interior')
center=T@start+shift
bound_viol=float(max(0,np.max(lb-center),np.max(center-ub)))
eq_shift=float(np.max(np.abs(S@shift)))
eq_transform=float(np.max(np.abs(S@T)))
if max(bound_viol,eq_shift,eq_transform)>args.tolerance:
    errors.append('original bounds or equality map residual exceeds tolerance')
problem=hopsy.Problem(A,b)
chains=[hopsy.MarkovChain(problem,proposal=hopsy.UniformCoordinateHitAndRunProposal,
                          starting_point=start.copy()) for _ in range(2)]
_,samples=hopsy.sample(chains,[hopsy.RandomNumberGenerator(7281+i) for i in range(2)],
                       n_samples=64,thinning=100,n_procs=1)
flux=samples.reshape(-1,A.shape[1])@T.T+shift
sample_bound=float(max(0,np.max(lb-flux),np.max(flux-ub)))
sample_eq=float(np.max(np.abs(S@flux.T)))
if max(sample_bound,sample_eq)>args.tolerance:
    errors.append('pilot original-space feasibility exceeds tolerance')
result=dict(status='passed' if not errors else 'failed',errors=errors,
            raw_sha256=raw_sha,model=str(root.resolve()),hostname=platform.node(),
            tolerance=args.tolerance,n=len(ids),d=A.shape[1],m=A.shape[0],
            min_start_slack=float(slack.min()),max_start_bound_violation=bound_viol,
            max_S_shift=eq_shift,max_S_T=eq_transform,pilot_chains=2,pilot_draws=64,
            pilot_thinning=100,max_pilot_bound_violation=sample_bound,
            max_pilot_equality_residual=sample_eq,preprocessing_backend=meta['solver'])
result.update(zero_pruned_reactions=len(pruned),max_pruned_bound_violation=pruned_bound,
              zero_embedding_exact=bool(not pruned or
                  (np.count_nonzero(T[pruned])==0 and np.count_nonzero(shift[pruned])==0)))
out=Path(args.output) if args.output else root/'validation.json'
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
raise SystemExit(bool(errors))
