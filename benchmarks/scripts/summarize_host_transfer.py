#!/usr/bin/env python3
"""Keep allocation and first-touch costs in the host-transfer comparison."""
import argparse,csv,json,statistics
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('job_dir');a=p.parse_args();root=Path(a.job_dir)
with (root/'transfers.csv').open() as f:
    raw=list(csv.DictReader(line for line in f if not line.startswith('#')))
rows=[]
assert {int(r['bytes']) for r in raw}=={2**30,8*2**30,26*2**30},'Incomplete size cases'
for size in sorted({int(r['bytes']) for r in raw}):
    assert sum(r['variant']=='gpu' and r['phase']=='free' for r in raw if int(r['bytes'])==size)==1,'Incomplete GPU case'
    for variant in ('cold_pageable','prefault8_pageable','pinned'):
        events=[r for r in raw if int(r['bytes'])==size and r['variant']==variant]
        def seconds(phase):
            return sum(float(r['seconds']) for r in events if r['phase']==phase)
        warm=statistics.median(float(r['seconds']) for r in events if r['phase']=='copy_warm')
        first=seconds('alloc')+seconds('prefault8')+seconds('copy_first')
        rows.append(dict(gib=size/2**30,variant=variant,alloc_seconds=seconds('alloc'),
            prefault_seconds=seconds('prefault8'),first_copy_seconds=seconds('copy_first'),
            first_result_seconds=first,free_seconds=seconds('free'),
            first_lifecycle_seconds=first+seconds('free'),warm_copy_median_seconds=warm,
            warm_gib_per_second=size/2**30/warm,
            first_copy_minor_faults=sum(int(r['minor_faults']) for r in events if r['phase']=='copy_first')))
result=dict(rows=rows,n_warm_repeats=3,endpoint_checks_passed=True,
            note='First result includes host allocation and prefaulting. GPU allocation/fill are common setup and remain in raw CSV. Free cost is also reported. Timings are one GPU allocation, not Naja end-to-end measurements.')
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
with (root/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for r in rows:
    print(f"{r['gib']:2.0f} GiB {r['variant']:20s} alloc={r['alloc_seconds']:.3f} prefault={r['prefault_seconds']:.3f} firstcopy={r['first_copy_seconds']:.3f} totalfirst={r['first_result_seconds']:.3f} warm={r['warm_copy_median_seconds']:.3f}s")
