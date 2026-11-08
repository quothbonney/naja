"""Four independent warm GLPK models; unresolved/capped candidates stay active."""
import argparse,json,os,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from cached_target_mask_bounded import build,sha

def worker(task):
    model,indices,cap=task
    return build(model,candidate_indices=indices,lp_time_ms=cap)

def build_parallel(model,workers=4,lp_time_ms=1000):
    model=Path(model).resolve();started=time.perf_counter();z=np.load(model/'rounding/polytope.npz')
    candidates=np.flatnonzero(np.max(np.abs(z['T']),axis=1)<=1e-6)
    tasks=[(model,candidates[i::workers].tolist(),lp_time_ms) for i in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as pool:results=list(pool.map(worker,tasks))
    active=np.logical_and.reduce([a for a,_ in results]);report=results[0][1]
    records=sorted([r for _,p in results for r in p['reactions']],key=lambda r:r['index'])
    assert len(records)==len(candidates) and len({r['index'] for r in records})==len(candidates)
    report.update(reactions=records,candidates=len(candidates),active_reactions=int(active.sum()),
        excluded_nearconstant_reactions=int((~active).sum()),wall_seconds=time.perf_counter()-started,
        workers=workers,worker_code_sha256=sha(Path(__file__).with_name('cached_target_mask_bounded.py')),
        orchestration_code_sha256=sha(Path(__file__)),shard_policy='Disjoint candidate indices; independent GLPK models with identical target/options. Only verified optimal narrow ranges excluded; caps/unresolved stay active.')
    return active,report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('model',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--lp-time-ms',type=int,default=1000);a=p.parse_args()
 active,report=build_parallel(a.model,a.workers,a.lp_time_ms);a.output.mkdir(parents=True,exist_ok=True);np.save(a.output/'active_reactions.npy',active);(a.output/'diagnostic_mask.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['candidates','excluded_nearconstant_reactions','wall_seconds','workers']}))
