#!/usr/bin/env python3
"""Common reaction-space diagnostics for reduced Naja/hopsy samples.

Rows in 2-D interleaved Naja files are draw-major, then chain. All methods
are evaluated by the same ArviZ bulk ESS and classical split-Rhat protocol.
"""
import argparse
import csv
import hashlib
import json
import os
import uuid
import time
from pathlib import Path

import arviz as az
import numpy as np
import xarray as xr

PREFIX_GRID = (100, 250, 500, 1000, 2000, 5000, 10000)


def split_rhat(x):
    """Classical (unranked) split-Rhat; x has chain, draw, variable axes."""
    half = x.shape[1] // 2
    if half < 2:
        return np.full(x.shape[-1], np.nan)
    split = np.concatenate((x[:, :half], x[:, -half:]), axis=0)
    within = split.var(axis=1, ddof=1).mean(axis=0)
    between = half * split.mean(axis=1).var(axis=0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sqrt(((half - 1) / half * within + between / half) / within)


def bulk_ess(x):
    dataset = xr.Dataset({'flux': (('chain', 'draw', 'reaction'), x)})
    return az.ess(dataset, method='bulk')['flux'].values


def diagnostics(x):
    return split_rhat(x), bulk_ess(x)


def as_chains(samples, n_chains, layout):
    if samples.ndim == 3:
        if n_chains is not None and samples.shape[0] != n_chains:
            raise ValueError('n-chains does not match first sample dimension')
        return samples
    if samples.ndim != 2 or n_chains is None:
        raise ValueError('2-D input requires --n-chains and explicit --layout')
    if layout not in ('interleaved', 'chain-major'):
        raise ValueError('2-D input requires --layout interleaved or chain-major')
    if samples.shape[0] % n_chains:
        raise ValueError('sample row count is not divisible by chain count')
    draws = samples.shape[0] // n_chains
    if layout == 'interleaved':
        return samples.reshape(draws, n_chains, samples.shape[1]).transpose(1, 0, 2)
    return samples.reshape(n_chains, draws, samples.shape[1])


def scalar_summary(rhat, ess):
    # Infinity signals nonconvergence and must not be discarded.
    rhat = rhat[~np.isnan(rhat)]
    ess = ess[np.isfinite(ess)]
    return {
        'split_rhat_median': float(np.median(rhat)) if len(rhat) else None,
        'split_rhat_p95': float(np.quantile(rhat, .95)) if len(rhat) else None,
        'bulk_ess_median': float(np.median(ess)) if len(ess) else None,
        'diagnosed_reactions': int(len(rhat)),
    }


def json_safe(obj):
    # Strict JSON: nonfinite diagnostic values become null, never invalid tokens.
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        return None
    return obj


def evaluate(samples_path, model_dir, output, *, n_chains=None, layout=None,
             wall_seconds=None, chunk_size=16, bounds_tolerance=1e-7,
             prefixes=PREFIX_GRID):
    started = time.perf_counter()
    run_path = Path(samples_path).parent / 'run.json'
    if run_path.exists():
        run = json.loads(run_path.read_text())
        if run.get('status') != 'complete':
            raise ValueError('run is incomplete; refusing diagnostics on unwritten memmap rows')
        if wall_seconds is None:
            wall_seconds = run.get('wall_including_normalization_seconds')
            if wall_seconds is None:
                raw_wall = run.get('process_wall_seconds', run.get('wall_seconds', run.get('timings', {}).get('wall_seconds')))
                wall_seconds = raw_wall + run.get('normalization_seconds', 0.) if raw_wall else None
    samples = as_chains(np.load(samples_path, mmap_mode='r'), n_chains, layout)
    chains, draws, dimension = samples.shape
    if chains < 2 or draws < 4:
        raise ValueError('diagnostics require at least 2 chains and 4 draws')
    model_dir = Path(model_dir)
    with np.load(model_dir / 'rounding' / 'polytope.npz') as poly:
        T, shift = poly['T'], poly['shift'].reshape(-1)
    if T.ndim != 2 or T.shape[1] != dimension or T.shape[0] != len(shift):
        raise ValueError('sample dimension and affine backmap shapes do not match')
    lower = np.loadtxt(model_dir / 'gem/l_bounds.csv', delimiter=',').reshape(-1)
    upper = np.loadtxt(model_dir / 'gem/u_bounds.csv', delimiter=',').reshape(-1)
    ids = (model_dir / 'gem/reaction_ids.txt').read_text().splitlines()
    reactions = len(ids)
    if T.shape[0] != reactions or len(lower) != reactions or len(upper) != reactions:
        raise ValueError('GEM bounds/IDs and backmap reaction counts do not match')
    if np.any(lower > upper):
        raise ValueError('inconsistent reaction bounds')
    mask_path = model_dir / 'active_reactions.npy'
    mask_report_path = model_dir / 'diagnostic_mask.json'
    model_active = np.ones(reactions, dtype=bool)
    mask_report = None
    if mask_path.exists() and mask_report_path.exists():
        mask_report = json.loads(mask_report_path.read_text())
        with (model_dir / 'rounding/polytope.npz').open('rb') as f:
            geometry_hash = hashlib.file_digest(f, 'sha256').hexdigest()
        if mask_report['geometry_sha256'] != geometry_hash:
            raise ValueError('model diagnostic mask geometry hash mismatch')
        model_active = np.load(mask_path)
        if model_active.dtype != bool or model_active.shape != (reactions,):
            raise ValueError('invalid model diagnostic mask shape/dtype')
    budgets = sorted(set([int(n) for n in prefixes if 4 <= n <= draws] + [draws]))
    rhats = {n: np.full(reactions, np.nan) for n in budgets}
    esses = {n: np.full(reactions, np.nan) for n in budgets}
    constant = np.zeros(reactions, dtype=bool)
    exactly_constant = np.zeros(reactions, dtype=bool)
    empirical_ranges = np.zeros(reactions)
    any_invalid = np.zeros((chains, draws), dtype=bool)
    any_exact_invalid = np.zeros((chains, draws), dtype=bool)
    violations = np.zeros(reactions, dtype=np.int64)
    exact_violations = np.zeros(reactions, dtype=np.int64)
    worst = np.zeros(reactions)
    for lo in range(0, reactions, chunk_size):
        hi = min(reactions, lo + chunk_size)
        flux = np.empty((chains, draws, hi - lo), dtype=np.float64)
        for draw_lo in range(0, draws, 256):
            draw_hi = min(draws, draw_lo + 256)
            block = np.asarray(samples[:, draw_lo:draw_hi], dtype=np.float64)
            mapped = block.reshape(-1, dimension) @ T[lo:hi].T
            flux[:, draw_lo:draw_hi] = mapped.reshape(chains, draw_hi - draw_lo, hi - lo) + shift[lo:hi]
        if not np.isfinite(flux).all():
            raise ValueError('nonfinite samples/backmapped flux detected')
        extent = np.ptp(flux, axis=(0, 1))
        scale = np.maximum(1., np.max(np.abs(flux), axis=(0, 1)))
        empirical_ranges[lo:hi] = extent
        constant[lo:hi] = extent <= 1e-12 * scale
        exactly_constant[lo:hi] = extent == 0.
        for n in budgets:
            active = ~exactly_constant[lo:hi]
            if n < draws:
                active &= model_active[lo:hi]  # full-budget pass alone retains the legacy all-variable summary
            if np.any(active):
                rhat, ess = diagnostics(flux[:, :n, active])
                idx = np.arange(lo, hi)[active]
                rhats[n][idx], esses[n][idx] = rhat, ess
        violation = np.maximum(np.maximum(lower[lo:hi] - flux, flux - upper[lo:hi]), 0.)
        invalid, exact_invalid = violation > bounds_tolerance, violation > 0.
        any_invalid |= invalid.any(axis=-1)
        any_exact_invalid |= exact_invalid.any(axis=-1)
        violations[lo:hi] = invalid.sum(axis=(0, 1))
        exact_violations[lo:hi] = exact_invalid.sum(axis=(0, 1))
        worst[lo:hi] = violation.max(axis=(0, 1))
    primary_active = model_active & ~exactly_constant
    excluded_model_ranges = empirical_ranges[~model_active]
    summaries = []
    first_quality = None
    for n in budgets:
        item = {'draws_per_chain': n, **scalar_summary(rhats[n][primary_active], esses[n][primary_active])}
        median_rhat, median_ess = item['split_rhat_median'], item['bulk_ess_median']
        item['quality_met'] = bool(median_rhat is not None and median_ess is not None
                                   and median_rhat <= 1.01 and median_ess >= 100)
        if item['quality_met'] and first_quality is None:
            first_quality = n
        summaries.append(item)
    result = {
        'schema_version': 2, 'model': model_dir.name,
        'samples': str(Path(samples_path).resolve()), 'model_dir': str(model_dir.resolve()),
        'shape': list(samples.shape), 'sample_space': 'reduced_affine_backmapped_to_reactions',
        'n_reactions': reactions, 'constant_reactions_excluded': int((~primary_active).sum()),
        'constant_definition': 'model-derived solver-detected nearconstant mask plus exactly constant sampled reactions',
        'diagnostic_mask_status': 'model_lp_verified' if mask_report else 'model_mask_missing_provisional',
        'diagnostic_mask_geometry_sha256': mask_report['geometry_sha256'] if mask_report else None,
        'diagnostic_mask_source_sha256': mask_report['source_sha256'] if mask_report else None,
        'diagnostic_mask_range_tolerance': mask_report['range_tolerance'] if mask_report else None,
        'solver_detected_nearconstant_reactions_excluded': int((~model_active).sum()),
        'exactly_constant_sampled_reactions': int(exactly_constant.sum()),
        'solver_excluded_empirical_range_max': float(excluded_model_ranges.max()) if len(excluded_model_ranges) else None,
        'solver_excluded_empirical_range_over_1e_minus_7_count': int((excluded_model_ranges > 1e-7).sum()),
        'solver_excluded_empirical_range_note': 'Measured ranges are checked and retained even for LP-nearconstant reactions; values above 1e-7 expose affine-backmap numerical residue. Exclusion changes diagnostics only, never bound checks.',
        'legacy_empirical_numerically_variable_summary': scalar_summary(rhats[draws][~constant], esses[draws][~constant]),
        'legacy_empirical_constant_count': int(constant.sum()),
        'legacy_empirical_constant_definition': 'full-run max-min <= 1e-12 * max(1, max(abs(flux)))',
        'empirical_absolute_range_gt_1e_minus_7_summary': scalar_summary(rhats[draws][empirical_ranges > 1e-7], esses[draws][empirical_ranges > 1e-7]),
        'diagnostic_definition': 'classical split-Rhat; ArviZ pooled rank-normalized bulk ESS',
        'arviz_version': az.__version__, 'numpy_version': np.__version__,
        **scalar_summary(rhats[draws][primary_active], esses[draws][primary_active]),
        'bounds_tolerance': bounds_tolerance,
        'sample_bound_violation_fraction': float(any_invalid.mean()),
        'coordinate_bound_violation_fraction': float(violations.sum() / (chains * draws * reactions)),
        'zero_tolerance_sample_bound_violation_fraction': float(any_exact_invalid.mean()),
        'zero_tolerance_coordinate_bound_violation_fraction': float(exact_violations.sum() / (chains * draws * reactions)),
        'max_absolute_bound_violation': float(worst.max()),
        'wall_seconds': wall_seconds,
        'samples_per_second': chains * draws / wall_seconds if wall_seconds else None,
        'prefix_quality': summaries,
        'first_quality_prefix_draws': first_quality,
        'estimated_time_to_quality_seconds': (wall_seconds * first_quality / draws
                                             if wall_seconds and first_quality else None),
        'time_to_quality_note': 'Linear interpolation from total wall time; NOT measured time-to-quality. Includes fixed overhead proportionally. Independent budget runs required for measured comparison.',
        'metrics_wall_seconds': time.perf_counter() - started,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    temporary_token = f'{os.getpid()}.{uuid.uuid4().hex}'
    temporary_json = output / f'metrics.{temporary_token}.json.tmp'
    temporary_json.write_text(json.dumps(json_safe(result), indent=2) + '\n')
    temporary_json.replace(output / 'metrics.json')
    temporary_csv = output / f'reaction_metrics.{temporary_token}.csv.tmp'
    with temporary_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['reaction', 'constant_excluded', 'solver_nearconstant_excluded', 'empirical_range', 'split_rhat', 'bulk_ess',
                         'bound_violation_fraction', 'zero_tolerance_bound_violation_fraction',
                         'max_absolute_bound_violation'])
        for j, reaction in enumerate(ids):
            writer.writerow([reaction, bool(not primary_active[j]), bool(not model_active[j]), empirical_ranges[j], rhats[draws][j], esses[draws][j],
                             violations[j] / (chains * draws), exact_violations[j] / (chains * draws), worst[j]])
    temporary_csv.replace(output / 'reaction_metrics.csv')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', required=True, type=Path)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--n-chains', type=int)
    parser.add_argument('--layout', choices=['interleaved', 'chain-major'])
    parser.add_argument('--wall-seconds', type=float)
    parser.add_argument('--chunk-size', type=int, default=16)
    parser.add_argument('--bounds-tolerance', type=float, default=1e-7)
    parser.add_argument('--prefixes', type=int, nargs='*', default=PREFIX_GRID)
    args = parser.parse_args()
    if args.chunk_size < 1 or args.bounds_tolerance < 0:
        parser.error('chunk-size must be positive and bounds-tolerance nonnegative')
    result = evaluate(args.samples, args.model_dir, args.output or args.samples.parent,
                      n_chains=args.n_chains, layout=args.layout, wall_seconds=args.wall_seconds,
                      chunk_size=args.chunk_size, bounds_tolerance=args.bounds_tolerance,
                      prefixes=args.prefixes)
    print(json.dumps(json_safe({k: result[k] for k in ['split_rhat_median', 'split_rhat_p95',
          'bulk_ess_median', 'sample_bound_violation_fraction', 'first_quality_prefix_draws']})))

if __name__ == '__main__':
    main()
