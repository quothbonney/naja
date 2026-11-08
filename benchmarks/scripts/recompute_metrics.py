#!/usr/bin/env python3
"""Schedule outdated completed-run metrics for schema-2 model-mask recomputation."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def stale_metric(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return value.get('schema_version', 0) < 2 or value.get('diagnostic_mask_status') != 'model_lp_verified'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--execute', type=Path)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.execute:
        sys.path.insert(0, str(ROOT))
        from metrics import evaluate
        entry = json.loads(args.execute.read_text())[args.index]
        run_path, model = Path(entry['run']), Path(entry['model'])
        with (run_path.parent / 'metrics.recompute.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            run = json.loads(run_path.read_text())
            metric = run_path.parent / 'metrics.json'
            if run.get('status') != 'complete' or not stale_metric(metric):
                print('Skipped changed/completed metrics:', run_path)
                return
            prior = json.loads(metric.read_text())
            version = prior.get('schema_version', 0)
            suffix = f'schema{version}' + ('.provisional' if version >= 2 else '')
            for name in ('metrics.json', 'reaction_metrics.csv'):
                source = run_path.parent / name
                target = source.with_name(f'{source.stem}.{suffix}{source.suffix}')
                if source.exists() and not target.exists():
                    shutil.copy2(source, target)
            evaluate(run_path.parent / 'samples.npy', model, run_path.parent)
        print('Recomputed schema2:', run_path)
        return
    entries = []
    for path in sorted((ROOT / 'results').rglob('run.json')):
        try:
            run = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if run.get('status') != 'complete' or not stale_metric(path.parent / 'metrics.json'):
            continue
        model = Path(run.get('model_dir', run.get('model', '')))
        if not (model / 'diagnostic_mask.json').exists():
            continue
        entries.append({'run': str(path), 'model': str(model)})
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    manifest = ROOT / 'audit' / f'metrics-recompute-{stamp}.json'
    manifest.write_text(json.dumps(entries, indent=2) + '\n')
    print(f'{len(entries)} outdated completed metrics; manifest={manifest}')
    if args.submit and entries:
        result = subprocess.run(['sbatch', '--parsable', f'--array=0-{len(entries)-1}%{args.concurrency}',
                                 '--output=logs/metrics-recompute-%A_%a.log',
                                 'scripts/recompute_metrics.sbatch', str(manifest)],
                                cwd=ROOT, check=True, capture_output=True, text=True)
        print('Submitted Slurm job:', result.stdout.strip())

if __name__ == '__main__':
    main()
