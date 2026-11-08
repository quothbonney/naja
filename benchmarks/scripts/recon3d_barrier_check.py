"""Small deterministic feasibility audit, never a convergence result."""
import json
from pathlib import Path
import numpy as np

B = Path(__file__).resolve().parents[1]
out = B / 'results/exploratory_recon3d_barrier/seed101'
run = json.loads((out / 'run.json').read_text())
assert run['status'] == 'complete'
model = B / 'data/models/Recon3D_nullspace_only'
with np.load(model / 'rounding/polytope.npz') as data:
    A, b, T, shift = (data[k] for k in ('A', 'b', 'T', 'shift'))
samples = np.load(out / 'samples.npy', mmap_mode='r')
flat = samples.reshape(-1, samples.shape[-1])
indices = np.unique(np.linspace(0, len(flat)-1, min(1024, len(flat)), dtype=int))
z = np.asarray(flat[indices], dtype=np.float64)
flux = z @ T.T + shift
lo = np.loadtxt(model / 'gem/l_bounds.csv', delimiter=',')
hi = np.loadtxt(model / 'gem/u_bounds.csv', delimiter=',')
v = np.maximum(np.maximum(lo-flux, flux-hi), 0)
result = dict(scope='Equality-only, no MVE; whitened GPU feasibility/throughput probe, not convergence or canonical comparison.',
    n=int(T.shape[0]), d=int(A.shape[1]), m=int(A.shape[0]), checked_samples=len(indices),
    selection='1024 evenly spaced flattened retained samples', tolerance=1e-7,
    max_original_bound_violation=float(v.max()),
    original_bound_violation_fraction=float(np.mean(np.any(v > 1e-7, axis=1))),
    max_reduced_violation=float(np.maximum(z @ A.T-b, 0).max()),
    seed=run['seed'], chains=run['chains'], retained_draws_per_chain=run['draws'],
    discarded_warmup_draws_per_chain=run['warmup'],
    binary_sha256=run['exe_sha256'], sampling_wall_with_normalization=run['wall_including_normalization_seconds'],
    profile=run['profile'], bundle_validation=str(model / 'validation.json'))
(out / 'probe_summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
