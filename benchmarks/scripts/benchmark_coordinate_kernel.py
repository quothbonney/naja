"""Paired sampler-phase comparison, preserving full outputs and exact commands."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

B = Path(__file__).resolve().parents[1]
O = B / 'results/coordinate_kernel/paired'
O.mkdir(parents=True, exist_ok=True)
proof = json.loads((B / 'results/coordinate_kernel/verification.json').read_text())
assert proof['status'] == 'passed' and proof['gpu_correctness_tests_passed'] == 5
cases = [
    ('reference', B / 'build-h100-prefault/naja', B / 'results/prefault-build-verification/verification.json'),
    ('coordinate', B / 'build-h100-coordinate/naja', B / 'results/coordinate_kernel/verification.json'),
]

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

rows = []
for chains in [16, 512, 1024]:
    hashes = {}
    for name, exe, verification in cases:
        out = O / f'chains{chains}' / name
        command = [sys.executable, str(B / 'naja_runner.py'), '--exe', str(exe),
            '--model', str(B / 'data/models/iJO1366'), '--out', str(out), '--seed', '101001',
            '--chains', str(chains), '--draws', '5000', '--warmup', '1000', '--tpb', '128',
            '--thinning', '500', '--barrier', 'rotate', '--pair-prob', '0',
            '--host-prefault-threads', '8', '--wide-indices-proof', str(verification), '--resume']
        with (O / f'chains{chains}-{name}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
        run = json.loads((out / 'run.json').read_text())
        assert run['status'] == 'complete'
        hashes[name] = sha(out / 'samples.npy')
        rows.append(dict(chains=chains, build=name, seed=101001, retained_draws_per_chain=5000,
            warmup_per_chain=1000, thinning=500, tpb=128, host_prefault_threads=8,
            sampler_phase_seconds=run['profile']['timing']['gpu_sampling_s'],
            retained_draws_per_sampler_second=chains*5000/run['profile']['timing']['gpu_sampling_s'],
            total_transitions=chains*6000*500,
            process_wall_including_normalization_seconds=run['wall_including_normalization_seconds'],
            samples_sha256=hashes[name], binary_sha256=run['exe_sha256'], run_dir=str(out), command=command))
        (O / 'progress.json').write_text(json.dumps(rows, indent=2) + '\n')
    assert hashes['reference'] == hashes['coordinate'], f'Paired trajectory mismatch at{chains} chains'

summary = dict(scope='Single-seed paired kernel specialization experiment, output-excluded sampler-phase lead; not a new CPU speedup or convergence claim.',
    job_id=os.environ.get('SLURM_JOB_ID'), rows=rows,
    pairs=[dict(chains=n, arrays_bitwise_equal=True,
        sampling_phase_speedup=next(r['sampler_phase_seconds'] for r in rows if r['chains']==n and r['build']=='reference')/
                               next(r['sampler_phase_seconds'] for r in rows if r['chains']==n and r['build']=='coordinate'))
           for n in [16,512,1024]])
(O / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
