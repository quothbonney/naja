#!/usr/bin/env python3
"""Cache a model-derived nearconstant reaction mask using original-model LPs.

This changes diagnostic aggregation only, never the sampled target or bounds.
Candidate selection uses nearzero rows of the saved affine backmap. Exclusion
requires successful original-model min/max LPs with flux range <= tolerance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def build_mask(model_dir, range_tolerance=1e-7, candidate_tolerance=1e-6, force=False):
    import cobra
    model_dir = Path(model_dir).resolve()
    meta = json.loads((model_dir / 'meta.json').read_text())
    source = Path(meta['source'])
    source_hash = digest(source)
    if source_hash != meta['source_sha256']:
        raise ValueError('raw model hash disagrees with preprocessing metadata')
    geometry_hash = digest(model_dir / 'rounding/polytope.npz')
    report_path = model_dir / 'diagnostic_mask.json'
    if report_path.exists() and (model_dir / 'active_reactions.npy').exists() and not force:
        prior = json.loads(report_path.read_text())
        if (prior['source_sha256'] == source_hash and prior['geometry_sha256'] == geometry_hash
                and prior['range_tolerance'] == range_tolerance
                and prior['candidate_tolerance'] == candidate_tolerance):
            print(f'{model_dir.name}: validated cached diagnostic mask', flush=True)
            return prior
        raise ValueError('diagnostic mask cache provenance mismatch; explicitly rebuild with --force')
    start = time.perf_counter()
    cobra.Configuration().solver = 'glpk'
    model = cobra.io.load_json_model(str(source))
    model.solver = 'glpk'
    model.solver.configuration.tolerances.feasibility = 1e-9
    ids = (model_dir / 'gem/reaction_ids.txt').read_text().splitlines()
    with np.load(model_dir / 'rounding/polytope.npz') as arrays:
        row_max = np.max(np.abs(arrays['T']), axis=1)
    if len(ids) != len(row_max):
        raise ValueError('reaction IDs and backmap shapes differ')
    active = np.ones(len(ids), dtype=bool)
    records = []
    candidates = np.flatnonzero(row_max <= candidate_tolerance)
    for iteration, i in enumerate(candidates):
        reaction = model.reactions.get_by_id(ids[i])
        record = {'reaction': ids[i], 'index': int(i), 'T_row_max': float(row_max[i])}
        try:
            model.objective = reaction.flux_expression
            model.objective_direction = 'min'
            lower = model.slim_optimize(error_value=None)
            lower_status = str(model.solver.status)
            model.objective_direction = 'max'
            upper = model.slim_optimize(error_value=None)
            upper_status = str(model.solver.status)
            width = float(upper - lower)
            valid = (lower_status == 'optimal' and upper_status == 'optimal'
                     and np.isfinite(lower) and np.isfinite(upper) and width >= -range_tolerance)
            nearconstant = bool(valid and width <= range_tolerance)
            record.update(minimum=float(lower), maximum=float(upper), range=width,
                          minimum_status=lower_status, maximum_status=upper_status,
                          nearconstant_excluded=nearconstant)
            active[i] = not nearconstant
        except Exception as exc:
            record.update(nearconstant_excluded=False, error=str(exc))
        records.append(record)
        if (iteration + 1) % 100 == 0:
            print(f'{model_dir.name}: {iteration+1}/{len(candidates)} candidate reaction LPs', flush=True)
    result = dict(schema_version=1, model=model_dir.name,
                  generated_utc=datetime.now(timezone.utc).isoformat(),
                  source=str(source), source_sha256=source_hash, geometry_sha256=geometry_hash,
                  solver='GLPK', cobra_version=cobra.__version__,
                  solver_feasibility_tolerance=1e-9, range_tolerance=range_tolerance,
                  candidate_tolerance=candidate_tolerance,
                  policy='Candidates T_row_max<=candidate_tolerance; exclude only successful original-model min/max LP range<=range_tolerance. No objective/biomass constraint added. Diagnostic aggregation only; sampling target unchanged.',
                  n_reactions=len(ids), candidates=len(candidates),
                  active_reactions=int(active.sum()), excluded_nearconstant_reactions=int((~active).sum()),
                  wall_seconds=time.perf_counter()-start, reactions=records)
    temporary_array = model_dir / 'active_reactions.tmp.npy'
    np.save(temporary_array, active)
    temporary_array.replace(model_dir / 'active_reactions.npy')
    temporary_report = report_path.with_suffix('.json.tmp')
    temporary_report.write_text(json.dumps(result, indent=2) + '\n')
    temporary_report.replace(report_path)
    print(f"{model_dir.name}: excluded {(~active).sum()}/{len(ids)} LP-nearconstant reactions in {result['wall_seconds']:.2f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dirs', type=Path, nargs='+')
    parser.add_argument('--range-tolerance', type=float, default=1e-7)
    parser.add_argument('--candidate-tolerance', type=float, default=1e-6)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    for model_dir in args.model_dirs:
        build_mask(model_dir, args.range_tolerance, args.candidate_tolerance, args.force)

if __name__ == '__main__':
    main()
