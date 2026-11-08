"""Same-allocation, order-balanced implementation timing; clocks are diagnostics only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

B = Path(__file__).resolve().parents[1]
O = B / 'results/coordinate_kernel/clock_audit'
O.mkdir(parents=True, exist_ok=False)
builds = {
    'reference': (B / 'build-h100-prefault/naja', B / 'results/prefault-build-verification/verification.json'),
    'coordinate': (B / 'build-h100-coordinate/naja', B / 'results/coordinate_kernel/verification.json'),
}
with (O / 'hardware.xml').open('w') as f:
    subprocess.run(['nvidia-smi', '-q', '-x'], stdout=f, stderr=subprocess.STDOUT, check=True)
query = ['nvidia-smi', '--query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,power.limit,pstate',
         '--format=csv', '--loop-ms=200']
rows = []
clock_stream = (O / 'clock_diagnostics.csv').open('w')
poller = subprocess.Popen(query, stdout=clock_stream, stderr=subprocess.STDOUT)
try:
    for index, seed in enumerate([101001, 202001, 303001]):
        order = ['reference', 'coordinate'] if index % 2 == 0 else ['coordinate', 'reference']
        hashes = {}
        for name in order:
            exe, proof = builds[name]
            out = O / f'seed{seed}' / name
            command = [sys.executable, str(B / 'naja_runner.py'), '--exe', str(exe),
                '--model', str(B / 'data/models/iJO1366'), '--out', str(out), '--seed', str(seed),
                '--chains', '512', '--draws', '5000', '--warmup', '1000', '--tpb', '128',
                '--thinning', '500', '--barrier', 'rotate', '--pair-prob', '0',
                '--host-prefault-threads', '8', '--wide-indices-proof', str(proof)]
            begin = datetime.now(timezone.utc).isoformat()
            with (O / f'{seed}-{name}.log').open('w') as f:
                subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=300)
            run = json.loads((out / 'run.json').read_text())
            with (out / 'samples.npy').open('rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            hashes[name] = digest
            rows.append(dict(seed=seed, build=name, begin_utc=begin,
                end_utc=datetime.now(timezone.utc).isoformat(),
                sampler_phase_seconds=run['profile']['timing']['gpu_sampling_s'],
                sample_sha256=digest, binary_sha256=run['exe_sha256'], run_dir=str(out)))
            (O / 'progress.json').write_text(json.dumps(rows, indent=2) + '\n')
        assert hashes['reference'] == hashes['coordinate'], f'Paired output mismatch at seed{seed}'
    summary = dict(scope='Fresh same-GPU, alternating build-order audit. Clock/power telemetry is diagnostic only and is not the Nsight utilization figure.',
        rows=rows, all_paired_outputs_bitwise_identical=True, clock_query=query,
        pairs=[dict(seed=seed, sampling_speedup=
            next(r['sampler_phase_seconds'] for r in rows if r['seed']==seed and r['build']=='reference') /
            next(r['sampler_phase_seconds'] for r in rows if r['seed']==seed and r['build']=='coordinate'))
            for seed in [101001,202001,303001]])
    (O / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
finally:
    poller.terminate()
    poller.wait(timeout=5)
    clock_stream.close()
