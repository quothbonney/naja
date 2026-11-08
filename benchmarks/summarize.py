#!/usr/bin/env python3
"""Collect completed run diagnostics and paired equal-sample CPU/Naja speedups."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics

FROZEN_NAJA_SHA256 = "c7baf4f0ed282b936ac0dbac4b896bfbfc040eb06da6f84aff6605fddf4b0fc6"


def collect(root):
    rows = []
    for path in sorted(root.rglob('run.json')):
        try:
            run = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue  # another job may be replacing this status file
        if run.get('method') not in ('naja', 'hopsy_chrr', 'cobra_chrr', 'dingo'):
            continue
        metric_path = path.parent / 'metrics.json'
        try:
            metrics = json.loads(metric_path.read_text()) if metric_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            metrics = {}
        timing = run.get('timings', {})
        internal_wall = run.get('wall_seconds', timing.get('wall_seconds'))
        raw_wall = run.get('process_wall_seconds', internal_wall)
        timing_scope = ('naja_external_process_plus_normalization' if run['method'] == 'naja' else
                        'cpu_full_process' if run.get('process_wall_seconds') is not None else
                        'cpu_startup_excluded_provisional')
        normalization = run.get('normalization_seconds', 0.)
        wall = run.get('wall_including_normalization_seconds')
        if wall is None:
            wall = raw_wall + normalization if raw_wall is not None else None
        model_dir = Path(run.get('model_dir', run.get('model', 'unknown')))
        model = model_dir.name
        try:
            model_meta = json.loads((model_dir / 'meta.json').read_text())
        except (OSError, json.JSONDecodeError):
            model_meta = {}
        try:
            validation = json.loads((model_dir / 'validation.json').read_text())
        except (OSError, json.JSONDecodeError):
            validation = {}
        family = path.relative_to(root).parts[0]
        build_variant = run.get('build_variant', 'wheel' if run['method'] == 'hopsy_chrr' else None)
        canonical_build = ((family == 'naja' and run.get('exe_sha256') == FROZEN_NAJA_SHA256) or
                           (family == 'hopsy' and build_variant in ('wheel', None)))
        shape = metrics.get('shape', run.get('shape', []))
        row = dict(model=model, method=run['method'], seed=run.get('seed'),
                   experiment_family=family, canonical_build_eligible=canonical_build,
                   build_variant=build_variant, naja_binary_sha256=run.get('exe_sha256'),
                   hopsy_build=run.get('hopsy_build'), hopsy_cxxflags=run.get('hopsy_cxxflags'),
                   environment_signature=run.get('environment_signature'),
                   threads_per_block=run.get('profile', {}).get('config', {}).get('tpb_ss'),
                   pair_probability=run.get('pair_probability', 0.0),
                   n_reactions=model_meta.get('n'), n_constraints=model_meta.get('m'),
                   preprocessing_hostname=model_meta.get('hostname'),
                   preprocessing_polyround_seconds=model_meta.get('timings', {}).get('polyround_seconds'),
                   preprocessing_validation_tolerance=validation.get('tolerance'),
                   preprocessing_validation_status=validation.get('status'),
                   preprocessing_strict_1e_minus_7_passed=validation.get('strict_1e_minus_7_passed'),
                   preprocessing_precision_caveat=validation.get('precision_caveat'),
                   preprocessing_max_start_bound_violation=validation.get('max_start_bound_violation'),
                   preprocessing_max_pilot_bound_violation=validation.get('max_pilot_bound_violation'),
                   thinning_label=run.get('thinning_label', run.get('requested_thinning')),
                   actual_thinning=run.get('thinning', run.get('profile', {}).get('config', {}).get('thinning')),
                   workers=run.get('workers'), barrier=run.get('barrier'),
                   status=run.get('status'), chains=run.get('chains'), draws=run.get('draws'),
                   warmup=run.get('warmup'), reduced_dimension=shape[2] if len(shape) == 3 else None,
                   native_wall_seconds=internal_wall, process_wall_seconds=run.get('process_wall_seconds'),
                   timing_scope=timing_scope, normalization_seconds=normalization,
                   matched_output_wall_seconds=wall,
                   retained_samples_per_second=(run.get('chains', 0) * run.get('draws', 0) / wall) if wall else None,
                   split_rhat_median=metrics.get('split_rhat_median'),
                   split_rhat_p95=metrics.get('split_rhat_p95'),
                   bulk_ess_median=metrics.get('bulk_ess_median'),
                   bound_violation_fraction=metrics.get('sample_bound_violation_fraction'),
                   coordinate_bound_violation_fraction=metrics.get('coordinate_bound_violation_fraction'),
                   zero_tolerance_sample_bound_violation_fraction=metrics.get('zero_tolerance_sample_bound_violation_fraction'),
                   zero_tolerance_coordinate_bound_violation_fraction=metrics.get('zero_tolerance_coordinate_bound_violation_fraction'),
                   max_bound_violation=metrics.get('max_absolute_bound_violation'),
                   first_quality_prefix_draws=metrics.get('first_quality_prefix_draws'),
                   metrics_present=bool(metrics), metrics_schema_version=metrics.get('schema_version'),
                   diagnostic_mask_status=metrics.get('diagnostic_mask_status'),
                   diagnostic_mask_geometry_sha256=metrics.get('diagnostic_mask_geometry_sha256'),
                   solver_excluded_empirical_range_max=metrics.get('solver_excluded_empirical_range_max'),
                   solver_excluded_empirical_range_over_1e_minus_7_count=metrics.get('solver_excluded_empirical_range_over_1e_minus_7_count'),
                   legacy_empirical_median_rhat=metrics.get('legacy_empirical_numerically_variable_summary', {}).get('split_rhat_median'),
                   legacy_empirical_median_ess=metrics.get('legacy_empirical_numerically_variable_summary', {}).get('bulk_ess_median'),
                   run_dir=str(path.parent))
        row.update(run.get('profile', {}).get('timing', {}))
        rows.append(row)
    identities = defaultdict(list)
    for row in rows:
        configuration = {k: row.get(k) for k in ('method', 'thinning_label', 'actual_thinning',
                          'workers', 'barrier', 'threads_per_block', 'pair_probability',
                          'build_variant', 'naja_binary_sha256', 'hopsy_build', 'hopsy_cxxflags')}
        row['sampler_configuration'] = json.dumps(configuration, sort_keys=True)
        row['duplicate_canonical_seed_rejected'] = False
        if row['experiment_family'] in ('naja', 'hopsy'):
            key = (row['model'], row['seed'], row['chains'], row['draws'], row['warmup'], row['sampler_configuration'])
            identities[key].append(row)
    for duplicates in identities.values():
        if len(duplicates) > 1:
            for row in duplicates:
                row['duplicate_canonical_seed_rejected'] = True
    return rows


def paired_speedups(rows):
    groups = defaultdict(list)
    for row in rows:
        if (not row.get('canonical_build_eligible')
                or row.get('experiment_family') not in ('naja', 'hopsy')
                or row.get('duplicate_canonical_seed_rejected')
                or row['status'] != 'complete' or not row['metrics_present']
                or row['diagnostic_mask_status'] != 'model_lp_verified'
                or (row.get('metrics_schema_version') or 0) < 2):
            continue
        key = tuple(row[k] for k in ('model', 'seed', 'chains', 'draws', 'warmup', 'reduced_dimension', 'diagnostic_mask_geometry_sha256'))
        groups[key].append(row)
    paired = []
    for key, runs in groups.items():
        cpus = [r for r in runs if r['method'] != 'naja' and r['matched_output_wall_seconds']
                and r['timing_scope'] == 'cpu_full_process']
        gpus = [r for r in runs if r['method'] == 'naja' and r['matched_output_wall_seconds']]
        if not cpus or not gpus:
            continue
        cpu = min(cpus, key=lambda r: r['matched_output_wall_seconds'])
        for gpu in gpus:
            paired.append(dict(model=key[0], seed=key[1], chains=key[2], draws=key[3],
                               fastest_cpu_method=cpu['method'],
                               fastest_cpu_thinning=cpu['thinning_label'],
                               fastest_cpu_workers=cpu['workers'],
                               cpu_seconds=cpu['matched_output_wall_seconds'],
                               naja_seconds=gpu['matched_output_wall_seconds'],
                               naja_native_cli_seconds=gpu['native_wall_seconds'],
                               naja_adapter_seconds=gpu['normalization_seconds'],
                               naja_barrier=gpu['barrier'], naja_thinning=gpu['actual_thinning'],
                               equal_sample_speedup=cpu['matched_output_wall_seconds'] / gpu['matched_output_wall_seconds'],
                               cpu_median_rhat=cpu['split_rhat_median'], naja_median_rhat=gpu['split_rhat_median'],
                               cpu_median_ess=cpu['bulk_ess_median'], naja_median_ess=gpu['bulk_ess_median']))
    return paired


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as f:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parent / 'results')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    out = args.output or args.results / 'summary'
    out.mkdir(parents=True, exist_ok=True)
    rows = collect(args.results)
    paired = paired_speedups(rows)
    write_csv(out / 'runs.csv', rows)
    write_csv(out / 'paired_equal_sample_speedups.csv', paired)
    by_model = defaultdict(list)
    for pair in paired:
        by_model[pair['model']].append(pair['equal_sample_speedup'])
    summaries = [dict(model=k, paired_runs=len(v), median_speedup=statistics.median(v),
                      min_speedup=min(v), max_speedup=max(v)) for k, v in sorted(by_model.items())]
    write_csv(out / 'speedup_summary.csv', summaries)
    (out / 'protocol_notes.txt').write_text(
        'Equal-sample ratios pair seed, model, chain count, retained draws, warmup and reduced dimension.\n'
        'CPU choice is fastest completed baseline per pair, with actual thinning recorded.\n'
        'Wall includes load, initialization, warmup, sampling, and native writes; Naja standardized-output conversion is added.\n'
        'Preprocessing and external diagnostics excluded for both methods. Different output precision is disclosed in run metadata.\n'
        'These are equal-sample comparisons, not equal-quality speedups. Inspect Rhat/ESS and bounds for both methods.\n'
        'Prefix time-to-quality is an interpolation estimate, not a measured stopping time.\n'
        'Missing/failed/time-capped runs remain in runs.csv; only completed runs with diagnostics enter ratios.\n')
    print(json.dumps({'runs': len(rows), 'paired_comparisons': len(paired), 'output': str(out)}))

if __name__ == '__main__':
    main()
