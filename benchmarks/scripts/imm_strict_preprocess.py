"""One process-local strict-settings PolyRound trial; canonical geometry is untouched."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np
import PolyRound.settings as settings

B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B))
R=B/'results/crossmodel_extension/iMM904_strict'
R.mkdir(parents=True,exist_ok=True)
original=settings.PolyRoundSettings

class StrictSettings(original):
    def __init__(self,*args,**kwargs):
        kwargs.update(thresh=1e-9,accepted_tol_violation=1.,hp_flags={'FeasibilityTol':1e-10})
        super().__init__(*args,**kwargs)

settings.PolyRoundSettings=StrictSettings
model=R/'model'
command=[str(B/'cpu_baseline.py'),'prep','--model',str(B/'data/raw/iMM904.json'),
    '--out',str(model),'--backend','glpk','--feasibility-tol','1e-10']
(R/'protocol.json').write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
    original_model='iMM904',thresh=1e-9,accepted_tol_violation=1.,feasibility_tolerance=1e-10,
    command=command,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    interpretation='Same biological input/bounds; new finite-precision preparation. No claim of exact equivalence to the previous cached A/b.'),indent=2)+'\n')
sys.argv=command
try:
    runpy.run_path(str(B/'cpu_baseline.py'),run_name='__main__')
    p=dict(np.load(model/'rounding/polytope.npz'))
    lo=np.loadtxt(model/'gem/l_bounds.csv',delimiter=',').reshape(-1)
    hi=np.loadtxt(model/'gem/u_bounds.csv',delimiter=',').reshape(-1)
    ids=(model/'gem/reaction_ids.txt').read_text().splitlines()
    start=p['T']@p['start']+p['shift'].reshape(-1)
    violation=np.maximum(np.maximum(lo-start,start-hi),0.)
    S=np.loadtxt(model/'gem/A_eq.csv',delimiter=',')
    report=dict(status='prepared',n=len(ids),d=p['A'].shape[1],m=p['A'].shape[0],
        old_geometry_d=288,geometry_sha256=hashlib.sha256((model/'rounding/polytope.npz').read_bytes()).hexdigest(),
        center_max_original_bound_violation=float(violation.max()),
        center_max_stoichiometric_residual=float(np.abs(S@start).max()),
        affine_stoichiometric_residual_max=float(np.abs(S@p['T']).max()),
        minimum_cached_start_slack=float(np.min(p['b'].reshape(-1)-p['A']@p['start'])))
    (R/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
except Exception as error:
    (R/'preparation.json').write_text(json.dumps(dict(status='failed',error=repr(error)),indent=2)+'\n')
    raise
