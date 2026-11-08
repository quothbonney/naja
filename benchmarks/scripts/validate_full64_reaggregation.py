#!/usr/bin/env python3
"""Compare a full fresh512-chain478-mask assay to exact CSV reaggregation."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

B=Path(__file__).resolve().parents[1]
R=B/'results/high_throughput/iJO1366'
fresh=R/'cached_target_audit/gpu_coordinate/seed101001'
aggregate=R/'cached_target_aggregate/gpu_coordinate/seed101001'
f=json.loads((fresh/'metrics.json').read_text())
a=json.loads((aggregate/'metrics.json').read_text())
fields=['split_rhat_median','split_rhat_p95','bulk_ess_median','diagnosed_reactions',
    'sample_bound_violation_fraction','coordinate_bound_violation_fraction',
    'zero_tolerance_sample_bound_violation_fraction','zero_tolerance_coordinate_bound_violation_fraction',
    'max_absolute_bound_violation','constant_reactions_excluded',
    'solver_detected_nearconstant_reactions_excluded','exactly_constant_sampled_reactions']
checks={field:f[field]==a[field] for field in fields}
with (fresh/'reaction_metrics.csv').open() as handle:x=list(csv.DictReader(handle))
with (aggregate/'reaction_metrics.csv').open() as handle:y=list(csv.DictReader(handle))
checks['all_per_reaction_rows_exact']=x==y
assert all(checks.values()),checks
def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
record=dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),fresh_job='856',
    source=f['samples'],shape=f['shape'],fresh=str(fresh),aggregate=str(aggregate),checks=checks,
    source_sha256={str(path):sha(path) for path in [fresh/'metrics.json',fresh/'reaction_metrics.csv',
        aggregate/'metrics.json',aggregate/'reaction_metrics.csv',B/'metrics.py',B/'scripts/reaggregate_cached_metrics.py']},
    scope='Fresh full512x5000x581 reassay with conservative478 mask matches all2583 per-reaction rows and primary/feasibility summaries exactly. Frozen producer supplies all non-exact full-budget statistics independent of mask; prefixes are not reused.')
(R/'cached_target_audit/reaggregation_validation.json').write_text(json.dumps(record,indent=2)+'\n')
print('passed',len(x),'per-reaction rows')
