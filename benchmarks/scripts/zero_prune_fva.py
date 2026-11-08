#!/usr/bin/env python3
"""Full original-bounds FVA to certify zero reactions before PolyRound."""
import argparse,hashlib,json,platform,time
from pathlib import Path
import cobra
from cobra.flux_analysis import flux_variability_analysis
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',required=True);p.add_argument('--out',required=True)
p.add_argument('--workers',type=int,default=16)
a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
sha=hashlib.sha256(Path(a.model).read_bytes()).hexdigest()
if (out/'fva.json').exists():
    assert json.loads((out/'fva.json').read_text())['source_sha256']==sha
    print('Using complete cached FVA',flush=True);raise SystemExit(0)
cobra.Configuration().solver='glpk'
model=cobra.io.load_json_model(a.model);model.solver='glpk';model.objective=0
model.solver.configuration.tolerances.feasibility=1e-9
bounds=[r.bounds for r in model.reactions]
start=time.perf_counter()
table=flux_variability_analysis(model,reaction_list=list(model.reactions),
    fraction_of_optimum=0.,loopless=False,processes=a.workers)
wall=time.perf_counter()-start
assert bounds==[r.bounds for r in model.reactions]
finite=np.isfinite(table[['minimum','maximum']]).all(axis=1)
zero=finite & (table[['minimum','maximum']].abs().max(axis=1)<=1e-9)
table.to_csv(out/'ranges.csv')
metadata=dict(source=str(Path(a.model).resolve()),source_sha256=sha,solver='glpk',
    feasibility_tolerance=1e-9,objective_zero=True,fraction_of_optimum=0.,
    open_exchanges=False,loopless=False,bounds_modified=False,workers=a.workers,
    zero_cutoff=1e-9,zero_reactions=int(zero.sum()),n=len(table),
    nonfinite_reactions=int((~finite).sum()),wall_seconds=wall,hostname=platform.node())
tmp=out/'fva.json.tmp';tmp.write_text(json.dumps(metadata,indent=2)+'\n');tmp.replace(out/'fva.json')
print(json.dumps(metadata,indent=2),flush=True)
