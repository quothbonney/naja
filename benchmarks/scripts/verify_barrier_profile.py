"""Verify unchanged CPU mode before interpreting the isolated barrier experiment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

B = Path(__file__).resolve().parents[1]
O = B / 'results/barrier_profile/verification'
O.mkdir(parents=True, exist_ok=True)
candidate = B / 'build-h100-barrier-profile/naja'
reference = B / 'build-h100-coordinate/naja'
env = os.environ.copy()
env.pop('NAJA_BARRIER_GPU_ROTATE', None)
env.pop('NAJA_BARRIER_VALIDATE_ROTATION', None)
env['NAJA_BARRIER_TRACE'] = '1'
with (O / 'gpu-correctness.log').open('w') as log:
    subprocess.run([str(B / 'scripts/enter.sh'), 'ctest', '--test-dir',
        str(candidate.parent), '-V', '-R', '^test_gpu_'], stdout=log,
        stderr=subprocess.STDOUT, env=env, check=True)
cases = []
for model, base, draws, warmup, thinning, chains in [
    ('BOX', 'synthetic', 128, 16, 2, 4),
    ('iJO1366', 'models', 256, 100, 96, 16),
]:
    arrays = {}
    for label, exe in [('reference', reference), ('cpu_instrumented', candidate)]:
        out = O / model / label
        subprocess.run([sys.executable, str(B / 'naja_runner.py'), '--exe', str(exe),
            '--model', str(B / 'data' / base / model), '--out', str(out), '--seed', '101',
            '--chains', str(chains), '--draws', str(draws), '--warmup', str(warmup),
            '--thinning', str(thinning), '--tpb', '128'], env=env, check=True)
        arrays[label] = np.load(out / 'samples.npy')
    equal = bool(np.array_equal(arrays['reference'], arrays['cpu_instrumented']))
    cases.append(dict(model=model, arrays_bitwise_equal=equal))
    assert equal, cases[-1]
with candidate.open('rb') as f:
    digest = hashlib.file_digest(f, 'sha256').hexdigest()
proof = dict(status='passed', binary_sha256=digest, gpu_analytic_tests_passed=5,
    cases=cases, scope='Instrumented CPU rotation mode preserves prior trajectories; GPU rotation algebra/feasibility are separate checks.')
(O / 'verification.json').write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps(proof, indent=2))
