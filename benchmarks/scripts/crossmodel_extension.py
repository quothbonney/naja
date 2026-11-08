"""Bounded follow-up CPU budgets, preserving all original benchmark evidence."""
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

B = Path(__file__).resolve().parents[1]
R = B / 'results/crossmodel_extension'
SEEDS = (101, 202, 303)
R.mkdir(parents=True, exist_ok=True)

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def one(model, seed, draws, reference):
    label = f'extension_20261108_worker_thin2000_n{draws}'
    out = B / 'results/native_cpu' / model / label / f'seed{seed}'
    command = [str(B / 'env-cpu-native/bin/python'), str(B / 'scripts/native_physical_pilot.py'),
        '--worker-timing', '--model', model, '--label', label, '--seed', str(seed),
        '--draws', str(draws), '--thinning', '2000']
    if model == 'iAF1260':
        command += ['--reuse-diagnostics-from', str(reference)]
    record = R / f'{model}_n{draws}_seed{seed}.json'
    if record.exists() and read(record).get('status') == 'complete':
        return read(record)
    if not (out / 'metrics.json').exists():
        assert not out.exists(), f'Interrupted attempt requires explicit review: {out}'
        write(R / 'progress.json', dict(utc=datetime.now(timezone.utc).isoformat(),
            model=model, seed=seed, draws=draws, command=command, job=os.getenv('SLURM_JOB_ID')))
        with (R / f'{model}_n{draws}_seed{seed}.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=2400)
    run, m, timing = [read(out / name) for name in
        ('run.json', 'metrics.json', 'sampling_math_timing.json')]
    assert m['schema_version'] == 2 and m['diagnostic_mask_status'] == 'model_lp_verified'
    assert timing['status'] == 'complete' and timing['worker_count'] == 16
    samples = np.load(out / 'samples.npy', mmap_mode='r')
    prior = np.load(reference / 'samples.npy', mmap_mode='r')
    n = min(prior.shape[1], samples.shape[1])
    identical = all(np.array_equal(samples[:, lo:min(lo+256, n)], prior[:, lo:min(lo+256, n)])
                    for lo in range(0, n, 256))
    assert identical, 'Instrumentation/reference trajectory parity failed'
    affinity = read(out / 'physical_affinity.json')
    assert affinity['selected_physical_core_count'] == 16
    passed = bool(m['split_rhat_median'] <= 1.01 and m['bulk_ess_median'] >= 100
                  and m['sample_bound_violation_fraction'] == 0 and m['bounds_tolerance'] == 1e-7)
    value = dict(status='complete', model=model, seed=seed, draws=draws,
        chains=16, warmup=1000, thinning=2000, physical_cores=16, reserved_logical_cpus=32,
        quality_and_bounds_met=passed, split_rhat_median=m['split_rhat_median'],
        bulk_ess_median=m['bulk_ess_median'],
        sample_bound_violation_fraction=m['sample_bound_violation_fraction'],
        maximum_bound_violation=m['max_absolute_bound_violation'],
        bounds_tolerance=m['bounds_tolerance'],
        worker_draw_critical_path_seconds=timing['worker_draw_critical_path_seconds'],
        sampler_phase_inclusive_seconds=timing['hopsy_sample_phase_inclusive_seconds'],
        secondary_wrapper_process_seconds=timing['instrumented_wrapper_process_seconds'],
        bulk_ess_per_worker_draw_second=m['bulk_ess_median']/timing['worker_draw_critical_path_seconds'],
        transitions_including_warmup=16*(draws+1000)*2000,
        reference=str(reference), reference_compared_prefix_draws=n,
        reference_prefix_bitwise_identical=identical,
        full_array_bitwise_identical=identical and prior.shape == samples.shape,
        run_directory=str(out), geometry_sha256=sha(B/'data/models'/model/'rounding/polytope.npz'),
        metrics_sha256=sha(B/'metrics.py'), worker_instrumentation_sha256=sha(B/'scripts/hopsy_worker_timing.py'),
        command=command, job_id=os.getenv('SLURM_JOB_ID'))
    write(record, value)
    summarize()
    return value

def summarize():
    groups = []
    all_rows = []
    for model, budgets in [('iML1515', (12000, 15000)), ('iAF1260', (10000,))]:
        gpu_rows = read(B/'results/precision_quality'/model/'summary.json')['rows'] if model == 'iML1515' else [
            read(B/'results/larger_quality'/model/'gpu'/f'seed{s}'/'quality.json')['measured'] for s in SEEDS]
        if model == 'iAF1260':
            for row in gpu_rows:
                row['sampling_phase_seconds'] = read(Path(row['run_dir'])/'run.json')['profile']['timing']['gpu_sampling_s']
        for draws in budgets:
            paths = [R/f'{model}_n{draws}_seed{s}.json' for s in SEEDS]
            rows = [read(path) for path in paths if path.exists()]
            if not rows:
                continue
            all_rows.extend(rows)
            complete = len(rows) == 3 and all(row['quality_and_bounds_met'] for row in rows)
            group = dict(model=model, draws=draws, rows=rows, passing_seeds=sum(row['quality_and_bounds_met'] for row in rows),
                tested_seeds=len(rows), complete_three_seed_group=complete)
            if complete:
                cpu = statistics.median(row['worker_draw_critical_path_seconds'] for row in rows)
                gpu = statistics.median(row['sampling_phase_seconds'] for row in gpu_rows)
                group.update(cpu_worker_seconds_median=cpu, gpu_sampler_seconds_median=gpu,
                    cpu_over_gpu_time_ratio=cpu/gpu,
                    paired_gpu_over_cpu_ess_rate=statistics.median(
                        (gr['bulk_ess_median']/gr['sampling_phase_seconds'])/cr['bulk_ess_per_worker_draw_second']
                        for cr, gr in zip(rows, gpu_rows)))
            groups.append(group)
    value = dict(schema_version=1, updated_utc=datetime.now(timezone.utc).isoformat(), groups=groups,
        scope='Fixed predeclared fresh budget extensions; original10k failures retained. CPU maximum accumulated worker C++draw time includes bindings/statecopy/descheduling, excludes IPC/output. GPU sampler includes allocation/slack/RNG. All original bounds checked at1e-7.',
        timer_proof='Full old trajectory parity for iAF1260; same-seed historical10k prefix parity for iML1515. Frozen globally validated instrumentation is unchanged.')
    write(R/'summary.json', value)
    lines = ['# Cross-model CPU budget extensions', '', value['scope'], '',
        '| Model | Retained draws | Passing / tested seeds | CPU worker s | GPU sampler s | CPU/GPU time | GPU/CPU ESS/s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for group in groups:
        values = [f"{group[key]:.4f}" if key in group else '—' for key in
                  ('cpu_worker_seconds_median','gpu_sampler_seconds_median','cpu_over_gpu_time_ratio','paired_gpu_over_cpu_ess_rate')]
        lines.append('| ' + ' | '.join([group['model'],str(group['draws']),f"{group['passing_seeds']}/{group['tested_seeds']}",*values]) + ' |')
    lines += ['', 'Every shown time is an actual execution at the stated budget. Failed and partial groups have no speedup estimate. '
        'These are finite tested configurations, not globally optimal CPU/GPU settings. See summary.json and per-seed JSON for commands, parity and geometry hashes.', '']
    (R/'README.md').write_text('\n'.join(lines))
    if all_rows:
        with (R/'table.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)

def main():
    cap = read(B/'results/iml_time_to_target/summary.json')['cpu_thin2000_fixed_cap']['rows']
    references = {row['seed']: Path(row['run_dir']) for row in cap}
    results = [one('iML1515', seed, 12000, references[seed]) for seed in SEEDS]
    if not all(row['quality_and_bounds_met'] for row in results):
        for seed in SEEDS:
            one('iML1515', seed, 15000, references[seed])
    for seed in SEEDS:
        one('iAF1260', seed, 10000, B/'results/larger_quality/iAF1260/cpu2000'/f'seed{seed}'/'final')
    summarize()

if __name__ == '__main__':
    main()
