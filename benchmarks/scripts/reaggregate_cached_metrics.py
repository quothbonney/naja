"""Exact full-budget remasking of the frozen producer's per-reaction diagnostics.

The verified metrics producer computes all non-exact-constant reaction diagnostics
at the full draw budget. Prefixes below the full budget cannot use this shortcut.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B))
import metrics

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def reaggregate(source, model, output):
    started=time.perf_counter()
    source,model,output=map(lambda path:Path(path).resolve(),(source,model,output))
    assert source!=output, 'New output directory required; source remains unchanged'
    producer=source/'metrics.json';table=source/'reaction_metrics.csv'
    original=json.loads(producer.read_text())
    expected=sha(B/'metrics.py')
    assert original['schema_version']==2 and original['metrics_code_sha256']==expected
    # This exact producer computes full-budget diagnostics for every non-exact
    # reaction; pin to its reviewed bytes rather than infer from schema alone.
    assert expected=='854e524cade5cecc6a9c5541cb15957e0336c3461434c2363d9ef2a2b701947c'
    certificate=json.loads((model/'diagnostic_mask.json').read_text())
    assert certificate['target_geometry_kind']=='exact_cached_polytope'
    geometry=sha(model/'rounding/polytope.npz')
    assert geometry==certificate['geometry_sha256']==original['diagnostic_mask_geometry_sha256']
    original_model=Path(original['model_dir']).resolve()
    assert original_model.is_dir(), 'Original model directory required to verify unmasked bound provenance'
    gem_hashes={}
    for name in ('l_bounds.csv','u_bounds.csv','reaction_ids.txt'):
        old,new=original_model/'gem'/name,model/'gem'/name
        assert sha(old)==sha(new), f'Original bound/reaction order changed: {name}'
        gem_hashes[name]=sha(new)
    mask=np.load(model/'active_reactions.npy')
    ids=(model/'gem/reaction_ids.txt').read_text().splitlines()
    assert mask.dtype==bool and mask.shape==(len(ids),)
    with table.open() as handle:
        reader=csv.DictReader(handle);fields=reader.fieldnames;rows=list(reader)
    assert [row['reaction'] for row in rows]==ids
    extent=np.array([float(row['empirical_range']) for row in rows])
    exact=extent==0
    rhat=np.array([float(row['split_rhat']) for row in rows])
    ess=np.array([float(row['bulk_ess']) for row in rows])
    assert not np.any(np.isnan(rhat[~exact])), 'Full per-reaction diagnostics missing'
    # Infinity means nonconvergence; it is retained by the frozen scalar reducer.
    active=mask & ~exact
    summary=metrics.scalar_summary(rhat[active],ess[active])
    draws=int(original['shape'][1])
    passed=bool(summary['split_rhat_median'] is not None and summary['bulk_ess_median'] is not None
        and summary['split_rhat_median']<=1.01 and summary['bulk_ess_median']>=100)
    result=dict(original)
    for key in ('first_quality_prefix_draws','estimated_time_to_quality_seconds','prefix_quality'):
        result.pop(key,None)
    result.update(summary,model_dir=str(model),constant_reactions_excluded=int((~active).sum()),
        diagnostic_mask_status='model_lp_verified',diagnostic_mask_geometry_sha256=geometry,
        diagnostic_mask_source_sha256=certificate['source_sha256'],
        diagnostic_mask_target_geometry_kind='exact_cached_polytope',
        diagnostic_mask_range_tolerance=certificate['range_tolerance'],
        solver_detected_nearconstant_reactions_excluded=int((~mask).sum()),
        exactly_constant_sampled_reactions=int(exact.sum()),
        solver_excluded_empirical_range_max=float(extent[~mask].max()) if np.any(~mask) else None,
        solver_excluded_empirical_range_over_1e_minus_7_count=int(np.sum(extent[~mask]>1e-7)),
        prefix_quality=[dict(draws_per_chain=draws,**summary,quality_met=passed)],
        first_quality_prefix_draws=draws if passed else None,
        estimated_time_to_quality_seconds=None,
        time_to_quality_note='Only full-budget diagnostics reaggregated. No inferred time or lower-budget prefix claim.',
        source_metrics_wall_seconds=original.get('metrics_wall_seconds'),
        metrics_wall_seconds=time.perf_counter()-started,
        reaggregation=dict(utc=datetime.now(timezone.utc).isoformat(),source_metrics=str(producer),
            source_metrics_sha256=sha(producer),source_reaction_csv=str(table),source_reaction_csv_sha256=sha(table),
            diagnostic_mask_sha256=sha(model/'diagnostic_mask.json'),active_reactions_sha256=sha(model/'active_reactions.npy'),
            unchanged_gem_sha256=gem_hashes,
            code_sha256=sha(Path(__file__)),samples_reused_unchanged=True,
            original_bound_checks_reused_unmasked=True,
            policy='Full-budget per-reaction classical split-Rhat/ArviZ bulk ESS reused exactly; only common model-derived aggregation mask changes. Original sample/coordinate bound statistics unchanged.'))
    output.mkdir(parents=True,exist_ok=True)
    destination=output/'metrics.json'
    temp=destination.with_suffix('.tmp')
    temp.write_text(json.dumps(metrics.json_safe(result),indent=2,allow_nan=False)+'\n');temp.replace(destination)
    for i,row in enumerate(rows):
        row['constant_excluded']=str(not active[i]);row['solver_nearconstant_excluded']=str(not mask[i])
    with (output/'reaction_metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='Directory containing frozen metrics.json and reaction_metrics.csv')
    parser.add_argument('--model',type=Path,required=True,help='Separate same-geometry model directory with exact-cached-target certificate')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();value=reaggregate(args.source,args.model,args.output)
    print(json.dumps({key:value[key] for key in ('split_rhat_median','bulk_ess_median','sample_bound_violation_fraction')}))

if __name__=='__main__':
    main()
