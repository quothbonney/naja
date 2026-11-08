#!/usr/bin/env python3
"""Check diagnostics, chain layout, affine backmap, and independent bounds counts."""
import json
import hashlib
from pathlib import Path
import tempfile

import numpy as np

from metrics import as_chains, diagnostics, evaluate


def main():
    rng = np.random.default_rng(12345)
    iid = rng.normal(size=(4, 2000, 3))
    rhat, ess = diagnostics(iid)
    assert np.all(rhat < 1.01), rhat
    assert np.median(ess) > 4000, ess
    separated = iid + np.arange(4)[:, None, None] * 3
    bad_rhat, bad_ess = diagnostics(separated)
    assert np.all(bad_rhat > 2), bad_rhat
    assert np.median(bad_ess) < 20, bad_ess
    native = iid.transpose(1, 0, 2).reshape(-1, 3)
    np.testing.assert_array_equal(as_chains(native, 4, 'interleaved'), iid)
    root = Path(__file__).resolve().parent / 'tmp'
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        tmp = Path(tmp)
        model = tmp / 'MODEL'
        (model / 'rounding').mkdir(parents=True)
        (model / 'gem').mkdir()
        np.savez(model / 'rounding/polytope.npz', T=np.array([[2., 0., 0.], [0., 0., 0.], [0., 0., 1e-9]]), shift=[1., 7., 0.])
        (model / 'gem/reaction_ids.txt').write_text('variable\nconstant\nsolver_blocked_numeric_residue\n')
        np.savetxt(model / 'gem/l_bounds.csv', [-1, 7, -1e-7], delimiter=',')
        np.savetxt(model / 'gem/u_bounds.csv', [3, 7, 1e-7], delimiter=',')
        np.save(model / 'active_reactions.npy', np.array([True, False, False]))
        with (model / 'rounding/polytope.npz').open('rb') as f:
            sha = hashlib.file_digest(f, 'sha256').hexdigest()
        (model / 'diagnostic_mask.json').write_text(json.dumps({
            'geometry_sha256': sha, 'source_sha256': 'synthetic-test', 'range_tolerance': 1e-7}))
        samples = tmp / 'samples.npy'
        np.save(samples, iid)
        result = evaluate(samples, model, tmp / 'results', prefixes=[100, 500], wall_seconds=20)
        expected = np.mean(np.abs(iid[:, :, 0]) > 1)
        assert result['sample_bound_violation_fraction'] == expected
        np.testing.assert_allclose(result['coordinate_bound_violation_fraction'], expected / 3, rtol=1e-15)
        assert result['constant_reactions_excluded'] == 2
        assert result['legacy_empirical_constant_count'] == 1
        assert result['solver_detected_nearconstant_reactions_excluded'] == 2
        assert result['diagnostic_mask_status'] == 'model_lp_verified'
        assert result['schema_version'] == 2
        assert result['n_reactions'] == 3
        assert result['split_rhat_median'] < 1.01
        json.loads((tmp / 'results/metrics.json').read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    print('metrics checks passed: IID vs separated chains, interleaving, affine backmap, bounds, model-based constant mask despite numerical backmap residue')

if __name__ == '__main__':
    main()
