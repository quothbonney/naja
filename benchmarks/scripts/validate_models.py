#!/usr/bin/env python3
"""Independent static checks against original BiGG stoichiometry and bounds."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix

B=Path(__file__).resolve().parents[1]
records=[]
for model in sorted((B/'data/models').iterdir()):
    if not (model/'meta.json').exists(): continue
    meta=json.loads((model/'meta.json').read_text())
    source=Path(meta['source'])
    raw=json.loads(source.read_text())
    ids=(model/'gem/reaction_ids.txt').read_text().splitlines()
    assert ids==[r['id'] for r in raw['reactions']]
    metabolites={m['id']:i for i,m in enumerate(raw['metabolites'])}
    row,col,val=[],[],[]
    for j,reaction in enumerate(raw['reactions']):
        for key,value in reaction['metabolites'].items():
            row.append(metabolites[key]);col.append(j);val.append(value)
    S=coo_matrix((val,(row,col)),shape=(len(metabolites),len(ids))).tocsr()
    with np.load(model/'rounding/polytope.npz') as p:
        A,b,T,shift,start=(p[k] for k in ['A','b','T','shift','start'])
    flux=T@start+shift
    lower=np.array([r['lower_bound'] for r in raw['reactions']])
    upper=np.array([r['upper_bound'] for r in raw['reactions']])
    record={'model':model.name,'n':len(ids),'d':A.shape[1],'m':A.shape[0],
            'source_sha_matches':hashlib.sha256(source.read_bytes()).hexdigest()==meta['source_sha256'],
            'finite_arrays':all(np.isfinite(a).all() for a in [A,b,T,shift,start]),
            'minimum_start_slack':float(np.min(b-A@start)),
            'max_start_bound_violation':float(np.maximum(np.maximum(lower-flux,flux-upper),0).max()),
            'max_stoichiometric_direction_residual':float(np.max(np.abs(S@T))),
            'max_stoichiometric_shift_residual':float(np.max(np.abs(S@shift))),
            'max_stoichiometric_start_residual':float(np.max(np.abs(S@flux)))}
    records.append(record)
(B/'results/setup/model-static-checks.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
