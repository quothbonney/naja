"""Lookup only: reaggregate completed held-out16 data with its original-model mask."""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import numpy as np

B = Path(__file__).resolve().parents[1]
base = B/'results/fresh16_conservative/iJO1366'
x = json.loads((base/'summary.json').read_text())
model = B/'data/models/iJO1366'
mask = np.load(model/'active_reactions.npy')
ids = (model/'gem/reaction_ids.txt').read_text().splitlines()
rows = []
for row in x['rows']:
    source = Path(row['source_run'])
    tab = list(csv.DictReader((source/'reaction_metrics.csv').open()))
    assert [v['reaction'] for v in tab] == ids
    keep = np.array([flag and float(v['empirical_range']) > 0 for flag, v in zip(mask, tab)])
    rv = np.array([float(v['split_rhat']) for v in tab])
    ev = np.array([float(v['bulk_ess']) for v in tab])
    assert np.isfinite(rv[keep]).all() and np.isfinite(ev[keep]).all()
    rhat, ess = float(np.median(rv[keep])), float(np.median(ev[keep]))
    rows.append(dict(method=row['method'], seed=row['seed'], original_model_mask_excluded=int((~mask).sum()),
        rhat=rhat, ess=ess, pass_quality=bool(rhat <= 1.01 and ess >= 100 and row['sample_bound_violation_fraction'] == 0),
        compute_seconds=row['sampler_compute_seconds'], ess_per_compute_second=ess/row['sampler_compute_seconds'],
        source_csv=str(source/'reaction_metrics.csv')))
    print(row['method'], row['seed'], rhat, ess, rows[-1]['pass_quality'])
result = dict(scope='Read-only reaggregation of existing all-reaction full-budget statistics; no new sampling or estimator changes.',
    mask_sha256=hashlib.sha256((model/'active_reactions.npy').read_bytes()).hexdigest(),
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows)
paired = []
for seed in sorted({r['seed'] for r in rows}):
    group = {r['method']: r for r in rows if r['seed'] == seed}
    cpu, gpu = group['cpu'], group['gpu']
    paired.append(dict(seed=seed, both_pass=cpu['pass_quality'] and gpu['pass_quality'],
        compute_time_ratio=cpu['compute_seconds']/gpu['compute_seconds'],
        ess_rate_ratio=gpu['ess_per_compute_second']/cpu['ess_per_compute_second']))
result['paired'] = paired
result['median_paired_compute_time_ratio'] = statistics.median(r['compute_time_ratio'] for r in paired)
result['median_paired_ess_rate_ratio'] = statistics.median(r['ess_rate_ratio'] for r in paired)
print('Paired medians:', result['median_paired_compute_time_ratio'], result['median_paired_ess_rate_ratio'])
(B/'results/closeout/fresh16_original_mask_lookup.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
