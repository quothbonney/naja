"""Prepare one rank-stable exact-KO target for the three sampling pipelines."""
import argparse,json,time
from pathlib import Path
from conditional_model_api import compile_conditional_mask,prepare_global,write_json
p=argparse.ArgumentParser();p.add_argument('--condition-root',type=Path,required=True);a=p.parse_args();r=a.condition_root.resolve();status=r/'quality/preparation_status.json';status.parent.mkdir(exist_ok=True)
shared_ready=False
try:
 info=json.loads((r/'condition.json').read_text());assert info['status']=='ready'
 compile_conditional_mask(r)
 shared_ready=True
 write_json(status,dict(status='global_rounding_running',shared_mask_ready=True))
 proof=prepare_global(r/'models/shared',r/'models/global_reround')
 write_json(status,dict(status='complete',shared_mask_ready=True,same_cached_target=True,global_proof=str(r/'models/global_reround/global_reround_proof.json'),
   shared_geometry_preparation_seconds=info['preparation_seconds'],shared_interior_LP_calls=info.get('LP_calls'),
   shared_interior_LP_seconds=info.get('LP_solve_seconds'),mandatory_hull_projection=True,
   baseline_label='Inherited rounding after required orthonormal affine-hull projection',
   global_round_algorithm_seconds=proof['global_round_algorithm_seconds']))
except Exception as e:
 write_json(status,dict(status='failed',shared_mask_ready=shared_ready,error=repr(e)));raise
