#!/usr/bin/env python3
"""Conservative cached-target diagnostics on saved full64/GPU512 arrays; no resampling."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

B = Path(__file__).resolve().parents[1]
R = B / 'results/high_throughput/iJO1366'
A = R / 'cached_target_audit'
A.mkdir(exist_ok=True)
model = A / 'model'
model.mkdir(exist_ok=True)
source = B / 'data/models/iJO1366'
for name in ('gem', 'rounding'):
    target = model / name
    if not target.exists():
        target.symlink_to(source / name, target_is_directory=True)
for name in ('active_reactions.npy', 'diagnostic_mask.json'):
    target = model / name
    expected = B / 'results/knockout_panel/base_cached_mask' / name
    if target.exists():
        assert target.read_bytes() == expected.read_bytes()
    else:
        shutil.copy2(expected, target)
shutil.copy2(source / 'meta.json', model / 'meta.json')
manifest_path = A / 'jobs.json'
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
summary = json.loads((R / 'summary.json').read_text())
for method in ('cpu', 'gpu_coordinate'):
    for row in summary['groups'][method]['rows']:
        seed = row['seed']
        if any(item['method'] == method and item['seed'] == seed for item in manifest):
            continue
        root = Path(row['run_dir'])
        output = A / method / f'seed{seed}'
        output.mkdir(parents=True, exist_ok=True)
        command = ['sbatch', '--parsable', f'--chdir={B.parent}',
                   str(B / 'scripts/full64_cached_target_metrics.sbatch'),
                   str(root / 'samples.npy'), str(model), str(output),
                   str(row['draws_per_chain'])]
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        manifest.append(dict(method=method, seed=seed, job=job, command=command,
            source_run=str(root), output=str(output),
            submitted_utc=datetime.now(timezone.utc).isoformat(),
            note='Original frozen estimator; full-budget only; conservative478 cached-target mask. No resampling.'))
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(method, seed, job, flush=True)
