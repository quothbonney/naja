"""Compare archived serial, refactored serial and spawn workers on actual data."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import metrics

META = {'metrics_wall_seconds', 'metrics_code_sha256', 'metrics_workers',
        'metrics_allocated_cpus', 'metrics_parallel_worker_blas_threads',
        'metrics_parallel_code_verified', 'metrics_auto_parallel_min_work'}


def check_equal(left, right, prefix=''):
    if isinstance(left, dict):
        a = {k: v for k, v in left.items() if k not in META}
        b = {k: v for k, v in right.items() if k not in META}
        assert a.keys() == b.keys(), (prefix, a.keys() ^ b.keys())
        for key in a:
            check_equal(a[key], b[key], prefix + '/' + key)
    elif isinstance(left, list):
        assert len(left) == len(right)
        for i, (a, b) in enumerate(zip(left, right)):
            check_equal(a, b, prefix + '/' + str(i))
    elif isinstance(left, (int, float)) and not isinstance(left, bool):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=prefix)
    else:
        assert left == right, (prefix, left, right)


def projected_model():
    src = ROOT / 'data/models/iJO1366'
    out = ROOT / 'audit/metrics-parallel-fixtures/iJO1366_128rxn'
    (out / 'rounding').mkdir(parents=True, exist_ok=True)
    (out / 'gem').mkdir(exist_ok=True)
    active = np.load(src / 'active_reactions.npy')
    indices = np.r_[np.flatnonzero(active)[:64], np.flatnonzero(~active)[:64]]
    with np.load(src / 'rounding/polytope.npz') as arrays:
        np.savez(out / 'rounding/polytope.npz', T=arrays['T'][indices], shift=arrays['shift'][indices])
    for name in ('l_bounds.csv', 'u_bounds.csv'):
        values = np.loadtxt(src / 'gem' / name, delimiter=',')
        np.savetxt(out / 'gem' / name, values[indices], delimiter=',')
    ids = (src / 'gem/reaction_ids.txt').read_text().splitlines()
    (out / 'gem/reaction_ids.txt').write_text('\n'.join(ids[i] for i in indices) + '\n')
    np.save(out / 'active_reactions.npy', active[indices])
    report = json.loads((src / 'diagnostic_mask.json').read_text())
    report['geometry_sha256'] = hashlib.sha256((out / 'rounding/polytope.npz').read_bytes()).hexdigest()
    report['projection_audit_only'] = True
    report['original_reaction_indices'] = indices.tolist()
    (out / 'diagnostic_mask.json').write_text(json.dumps(report, indent=2) + '\n')
    return out


def main():
    archive = ROOT / 'audit/metrics.serial-before-parallel.py'
    spec = importlib.util.spec_from_file_location('serial_reference', archive)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    cases = [('core', ROOT / 'data/models/e_coli_core',
              ROOT / 'results/quality/e_coli_core/naja128/seed101/budget500/samples.npy'),
             ('iJO_128rxn_50k', projected_model(),
              ROOT / 'results/quality/iJO1366/naja128/seed101/final-budget50000/samples.npy')]
    evidence = []
    for name, model, samples in cases:
        results, seconds = {}, {}
        for label, function, kwargs in [('archived_serial', reference.evaluate, {}),
                                         ('serial', metrics.evaluate, {'workers': 1}),
                                         ('parallel4', metrics.evaluate, {'workers': 4})]:
            target = ROOT / 'audit/metrics-parallel-results' / name / label
            start = time.perf_counter()
            results[label] = function(samples, model, target, prefixes=[100, 500, 2000], **kwargs)
            seconds[label] = time.perf_counter() - start
        check_equal(results['archived_serial'], results['serial'])
        check_equal(results['serial'], results['parallel4'])
        assert results['parallel4']['metrics_workers'] == 4
        base = ROOT / 'audit/metrics-parallel-results' / name
        csv_equal = ((base / 'archived_serial/reaction_metrics.csv').read_bytes() ==
                     (base / 'serial/reaction_metrics.csv').read_bytes() ==
                     (base / 'parallel4/reaction_metrics.csv').read_bytes())
        assert csv_equal, name
        row = {'case': name, 'shape': results['serial']['shape'],
               'reactions': results['serial']['n_reactions'], 'all_scientific_fields_allclose_1e_minus12': True,
               'per_reaction_csv_bitwise_equal': csv_equal, 'seconds': seconds,
               'parallel_speedup': seconds['serial'] / seconds['parallel4']}
        evidence.append(row)
        print(json.dumps(row), flush=True)
    assert hashlib.sha256((ROOT / 'metrics.py').read_bytes()).hexdigest() == metrics._METRICS_CODE_SHA256
    proof = {'generated_utc': datetime.now(timezone.utc).isoformat(),
             'metrics_code_sha256': hashlib.sha256((ROOT / 'metrics.py').read_bytes()).hexdigest(),
             'serial_reference_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'cases': evidence}
    target = ROOT / 'audit/metrics-parallel-verification.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(proof, indent=2) + '\n')
    temporary.replace(target)

if __name__ == '__main__':
    main()
