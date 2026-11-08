"""Scheduler-backed end-of-campaign cleanup, independent of the Codex/login session."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

B = Path(__file__).resolve().parents[1]
prefixes = ('bench-', 'quality-', 'metrics-', 'diagnostic-mask', 'naja-polyround',
            'Recon3D-validation', 'hopsy-tuned-default', 'recon1-')
own_id = os.environ.get('SLURM_JOB_ID')
result = subprocess.run(['squeue', '-h', '-r', '-u', str(os.getuid()), '-o', '%i|%j|%Z'],
                        capture_output=True, text=True, check=True)
cancelled = []
for line in result.stdout.splitlines():
    job, name, directory = line.split('|', 2)
    if job == own_id or not name.startswith(prefixes) or Path(directory) not in (B, B.parent):
        continue
    action = subprocess.run(['scancel', job], capture_output=True, text=True)
    cancelled.append(dict(job=job, name=name, returncode=action.returncode,
                          stderr=action.stderr.strip()))
record = dict(utc=datetime.now(timezone.utc).isoformat(), cleanup_job=own_id,
    reason='Original 12-hour campaign limit; preserve time for final report.', actions=cancelled)
out = B / 'results/slurm-deadline-cleanup.json'
out.write_text(json.dumps(record, indent=2) + '\n')
subprocess.run([str(B / 'env-cpu/bin/python'), str(B / 'scripts/new_results_report.py')], check=False)
print(json.dumps(record, indent=2))
