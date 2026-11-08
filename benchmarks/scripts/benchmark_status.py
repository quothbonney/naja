"""Read-only recovery snapshot; does not submit, release, cancel or rerun work."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

B = Path(__file__).resolve().parents[1]
snapshot = dict(utc=datetime.now(timezone.utc).isoformat(), workspace=str(B.parent), controllers=[], sources={})
pid_files = list((B / 'audit').rglob('*.pid')) + list((B / 'results').rglob('*.pid'))
pid_files += [B / 'results/refresh.pid', B / 'results/campaign.pid']
seen = set()
for path in pid_files:
    try:
        pid = int(path.read_text().strip())
        if pid in seen:
            continue
        seen.add(pid)
        proc = Path(f'/proc/{pid}/cmdline')
        command = proc.read_bytes().replace(b'\0', b' ').decode(errors='replace') if proc.exists() else None
        snapshot['controllers'].append(dict(pid_file=str(path.relative_to(B)), pid=pid,
                                            alive=command is not None, command=command))
    except (OSError, ValueError):
        pass
for name, relative in {
    'refresh':'results/refresh-status.json',
    'full64_sequence':'audit/high-throughput/full64-state.json',
    'held_jobs':'audit/high-throughput/full64-held-jobs.json',
    'selected_cpu_budget':'audit/high-throughput/cpu-selected-budget.json',
    'fresh16':'results/fresh16_validation/iJO1366/summary.json',
    'high_throughput':'results/high_throughput/iJO1366/summary.json',
    'conditioned_quality':'results/conditioned_prototype/quality_summary.json',
    'dimension_audit':'results/conditioned_prototype/dimensionality_warning.json',
    'knockout_catalog':'results/knockout_panel/catalog_summary.json',
    'slurm_cleanup':'results/slurm-deadline-cleanup.json',
    'continuation_window':'audit/continuation-20261108/window.json',
    'full64_audited':'results/high_throughput/iJO1366/comparison.json',
    'fresh16_heldout':'results/fresh16_conservative/iJO1366/summary.json',
    'conditioned_panel_audited':'results/knockout_panel/overnight_audit/summary.json',
    'conditioned_off_rotate_whiten':'results/conditioned_extensions/off_rotate_whiten_20261108/summary.json',
    'conditioned_200k':'results/conditioned_extensions/rotate_200k_20261108/summary.json',
    'conditioned_mixture':'results/conditioned_extensions/whiten_pair_mixture_20261108/summary.json',
    'conditioned_reports':'results/conditioned_extensions/reports_status.json',
    'crossmodel_cached':'results/crossmodel_extension/cached_assay_summary.json',
}.items():
    path = B / relative
    try:
        data = json.loads(path.read_text())
        snapshot['sources'][name] = dict(path=relative, exists=True,
            status=data.get('status', data.get('stage', 'see source')),
            keys=list(data.keys()))
    except (OSError, json.JSONDecodeError):
        snapshot['sources'][name] = dict(path=relative, exists=False)
try:
    jobs = subprocess.run(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%M|%C|%R'],
                          capture_output=True, text=True, timeout=15, check=True)
    snapshot['slurm'] = jobs.stdout.splitlines()
except (OSError, subprocess.SubprocessError) as error:
    snapshot['slurm_error'] = str(error)
out = B / 'results/RESUME_STATUS.json'
tmp = out.with_suffix('.tmp')
tmp.write_text(json.dumps(snapshot, indent=2)+'\n')
tmp.replace(out)
print(snapshot['utc'])
for row in snapshot['controllers']:
    print(f"controller pid={row['pid']} alive={row['alive']} {row['pid_file']}")
print('\n'.join(snapshot.get('slurm', [])))
print(f'Full snapshot: {out}')
