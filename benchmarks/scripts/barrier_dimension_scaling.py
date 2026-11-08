"""Nine cached-model barrier phase costs; timing repeats, not quality trials."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import barrier_profile_runs as br

B = Path(__file__).resolve().parents[1]
O = B/'results/barrier_dimension_scaling'
MODELS = ['e_coli_core', 'iAB_RBC_283', 'iJR904', 'iAF1260', 'iJO1366',
          'iML1515', 'iMM904', 'RECON1', 'Recon3D_nullspace_only']
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--plot', action='store_true')
a = p.parse_args()
O.mkdir(exist_ok=True)
if not a.plot:
    rows = []
    for model in MODELS:
        for repeat in range(1, 4):
            br.O = O/f'repeat{repeat}'
            br.O.mkdir(exist_ok=True)
            for mode in ('cpu', 'gpu'):
                try:
                    record = br.collect(model, 'whiten' if model.startswith('Recon3D') else 'rotate',
                                        1e-6 if model == 'RECON1' else 1e-7, mode, 'unprofiled')
                    stages = {s['name']: s['seconds'] for s in record['stages']}
                    metadata = br.read(B/'data/models'/model/'meta.json')
                    rows.append(dict(model=model, repeat=repeat, mode=mode, status='complete',
                        d=record['profile']['config']['reduced_dim'], m=record['profile']['config']['constraints'],
                        barrier_seconds=record['profile']['timing']['barrier_rotation_s'],
                        hessian_and_eigensolve_waited_seconds=stages['barrier.hessian_syrk']+stages['barrier.eigensolve_and_wait'],
                        rotation_seconds=stages.get('barrier.cpu_rotate_A', stages.get('barrier.gpu_rotate_A')),
                        polyround_historical_mve_seconds=metadata.get('timings', {}).get('round_seconds'),
                        declared_tolerance=record['feasibility']['declared_tolerance'],
                        short_samples_declared_bounds_passed=record['feasibility']['declared_bounds_passed'],
                        max_original_bound_violation=record['feasibility']['max_original_bound_violation'],
                        source=str(Path(record['run_dir'])/'result.json')))
                except Exception as error:
                    rows.append(dict(model=model, repeat=repeat, mode=mode, status='failed', error=repr(error)))
                br.dump(O/'summary.json', dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                    binary_sha256=br.EXPECTED_SHA, rows=rows,
                    scope='Three independent process timing repeats, fixed seed101, 128 total short samples. No convergence claim. PolyRound MVE is historical one-time preprocessing; Naja operates on cached geometry after that preprocessing, except the explicitly unrounded Recon3D exploratory target. These costs do not establish interchangeable rounding quality. Stage times include host waits/lazy initialization, not pure kernel counters.'))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (O/'source.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields);w.writeheader();w.writerows(rows)
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = br.read(O/'summary.json')['rows']
    fig, ax = plt.subplots(figsize=(6.3, 3.6), layout='constrained')
    definitions = [('cpu', 'barrier_seconds', 'Original barrier phase', '#777777', 'o'),
                   ('gpu', 'barrier_seconds', 'GPU-rotation barrier phase', '#2365a3', 'o'),
                   ('gpu', 'hessian_and_eigensolve_waited_seconds', 'Hessian + eigensolve stages', '#40a480', 's'),
                   ('gpu', 'polyround_historical_mve_seconds', 'Historical PolyRound MVE', '#be6e25', '^')]
    for mode, key, label, color, marker in definitions:
        points = []
        for model in MODELS:
            group = [r for r in rows if r.get('status') == 'complete' and r['model'] == model and r['mode'] == mode and r.get(key) and r[key] > 0]
            if group:
                values = [r[key] for r in group]
                points.append((group[0]['d'], statistics.median(values), min(values), max(values)))
        points.sort()
        ax.errorbar([p[0] for p in points], [p[1] for p in points],
                    yerr=[[p[1]-p[2] for p in points], [p[3]-p[1] for p in points]],
                    fmt=marker, ms=4, color=color, label=label, capsize=2)
    ax.set(xscale='log', yscale='log', xlabel='Reduced dimension', ylabel='Phase wall time (s)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.15, which='major');ax.legend(frameon=False, fontsize=8)
    for suffix in ('svg', 'pdf', 'png'):
        fig.savefig(O/f'barrier_dimension_scaling.{suffix}', dpi=180)
