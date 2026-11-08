#!/usr/bin/env python3
"""Check the localized seed override on a synthetic cube on the allocated GPU."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np

root = Path(__file__).resolve().parents[1]
model = root / 'data/synthetic/BOX'
(model / 'rounding').mkdir(parents=True, exist_ok=True)
(model / 'gem').mkdir(exist_ok=True)
d = 3
np.savez(model / 'rounding/polytope.npz', A=np.vstack([np.eye(d), -np.eye(d)]),
         b=np.ones(2*d), start=np.zeros(d), T=np.eye(d), shift=np.zeros(d))
(model / 'gem/reaction_ids.txt').write_text('x0\nx1\nx2\n')
np.savetxt(model / 'gem/l_bounds.csv', -np.ones(d))
np.savetxt(model / 'gem/u_bounds.csv', np.ones(d))
arrays = []
for name, seed in [('same_a',101), ('same_b',101), ('different',102)]:
    out = root / 'results/seed-check' / name
    subprocess.run([sys.executable, str(root/'naja_runner.py'), '--exe', str(root/'build-h100/naja'),
                    '--model', str(model), '--out', str(out), '--seed', str(seed),
                    '--chains', '4', '--draws', '128', '--warmup', '16',
                    '--thinning', '2', '--resume'], check=True)
    arrays.append(np.load(out/'samples.npy'))
result = {'same_seed_bitwise_equal': bool(np.array_equal(arrays[0],arrays[1])),
          'different_seed_differs': bool(not np.array_equal(arrays[0],arrays[2])),
          'shape': list(arrays[0].shape),
          'finite': bool(all(np.isfinite(a).all() for a in arrays)),
          'within_cube': bool(all(np.max(np.abs(a)) <= 1+1e-6 for a in arrays))}
(root/'results/seed-check/verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if not all(result[k] for k in ['same_seed_bitwise_equal','different_seed_differs','finite','within_cube']):
    raise SystemExit(1)
