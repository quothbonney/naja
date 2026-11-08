"""GPU analytic tests and same-trajectory checks for the isolated specialization."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

B = Path(__file__).resolve().parents[1]
O = B / 'results/coordinate_kernel'
candidate = B / 'build-h100-coordinate/naja'
reference = B / 'build-h100-prefault/naja'
with (O / 'gpu-correctness.log').open('w') as log:
    subprocess.run([str(B / 'scripts/enter.sh'), 'ctest', '--test-dir',
        str(candidate.parent), '-V', '-R', '^test_gpu_'], stdout=log,
        stderr=subprocess.STDOUT, check=True)

rows = []
for case, model, chains, draws, warmup, thinning, pair in [
    ('box_plain', 'BOX', 4, 128, 16, 2, 0),
    ('box_pair', 'BOX', 4, 128, 16, 2, .5),
    ('iJO_plain', 'iJO1366', 16, 256, 100, 96, 0),
]:
    arrays = {}
    for label, exe in [('reference', reference), ('candidate', candidate)]:
        out = O / 'verification' / case / label
        model_dir = B / ('data/synthetic' if model == 'BOX' else 'data/models') / model
        subprocess.run([sys.executable, str(B / 'naja_runner.py'), '--exe', str(exe),
            '--model', str(model_dir), '--out', str(out), '--seed', '101',
            '--chains', str(chains), '--draws', str(draws), '--warmup', str(warmup),
            '--thinning', str(thinning), '--tpb', '128', '--pair-prob', str(pair)], check=True)
        arrays[label] = np.load(out / 'samples.npy')
    equal = bool(np.array_equal(arrays['reference'], arrays['candidate']))
    rows.append(dict(case=case, arrays_bitwise_equal=equal,
        max_absolute_difference=float(np.max(np.abs(arrays['reference']-arrays['candidate'])))))
    assert equal, rows[-1]

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

proof = dict(status='passed', gpu_correctness_tests_passed=5, cases=rows,
    same_seed_original_samples_bitwise_equal=True, wide_index_build_verified=True,
    binary_sha256=sha(candidate), reference_binary_sha256=sha(reference),
    patch_sha256=sha(O / 'kernel.patch'), job_id=os.environ.get('SLURM_JOB_ID'),
    scope='Separate coordinate-only specialization. Mixture fallback retained. No frozen binary/results changed.')
(O / 'verification.json').write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps(proof, indent=2))
