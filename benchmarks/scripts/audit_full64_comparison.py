#!/usr/bin/env python3
"""Audit the saved 512-chain full64-CPU/H100 comparison without modifying runs."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

B = Path(__file__).resolve().parents[1]
R = B / 'results/high_throughput/iJO1366'
SEEDS = (101001, 202001, 303001)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def median(values):
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-array-hashes', action='store_true',
                        help='Read and hash all six retained arrays (~100 GB).')
    args = parser.parse_args()
    source = read(R / 'summary.json')
    checks, rows, identities, pending, cpu_chain_seeds = [], [], [], [], []
    verification_path = R / 'cached_target_audit/array_hash_verification.json'
    prior_hash_checks = read(verification_path)['checks'] if verification_path.exists() else []
    reaggregation_validation_path=R/'cached_target_audit/reaggregation_validation.json'
    reaggregation_validation=read(reaggregation_validation_path) if reaggregation_validation_path.exists() else {}
    sources = {}
    for method in ('cpu', 'gpu_coordinate'):
        originals = source['groups'][method]['rows']
        assert sorted(row['seed'] for row in originals) == list(SEEDS)
        for original in originals:
            seed = original['seed']
            root = Path(original['run_dir'])
            run = read(root / 'run.json')
            legacy = read(root / 'metrics.json')
            identity = read(root / 'diagnostics_identity.json')
            identities.append(identity['data'])
            assert run['status'] == 'complete'
            assert run['seed'] == seed and run['chains'] == 512
            assert run['draws'] == (12000 if method == 'cpu' else 5000)
            assert run['warmup'] == 1000
            assert legacy['metrics_code_sha256'] == sha(B / 'metrics.py')
            assert identity['metrics_sha256'] == sha(root / 'metrics.json')
            assert identity['data']['estimator_code_sha256'] == sha(B / 'metrics.py')
            assert identity['data']['shape'] == [512, run['draws'], 581]
            proof_path = root / 'diagnostics_identity_proof.json'
            if proof_path.exists():
                reuse = read(proof_path)
                assert reuse['exact_file_bytes_shape_dtype_geometry_mask_verified'] is True
                assert reuse['pilot_identity'] == reuse['final_identity']
                checks.append(dict(method=method, seed=seed, check='guarded_diagnostics_reuse',
                                   status='passed', source=str(proof_path)))
            if method == 'cpu':
                assert run['chain_seeds'] == list(range(seed, seed+512))
                cpu_chain_seeds.extend(run['chain_seeds'])
                affinity = read(root / 'physical_affinity.json')
                selected = set(affinity['selected_logical_cpus'])
                allowed = set(affinity['allowed_logical_cpus'])
                cores = {(entry['socket'], entry['core']) for entry in affinity['topology']
                         if entry['cpu'] in selected}
                assert len(selected) == len(cores) == 64 and len(allowed) == 128
                assert all(set(map(int, entry['siblings'].split(','))) <= allowed
                           for entry in affinity['topology'] if entry['cpu'] in selected)
                timing = read(root / 'sampling_math_timing.json')
                assert timing['proof']['instrumentation_module_sha256'] == sha(B / 'scripts/hopsy_worker_timing.py')
                logs = [read(path) for path in (root / 'worker_math_logs').glob('*.json')]
                assert sorted(item['chain_idx'] for item in logs) == list(range(512))
                assert all(item['n_samples'] == 13000 and item['thinning'] == 2000 for item in logs)
                totals = defaultdict(float)
                for item in logs:
                    totals[item['pid']] += item['draw_call_wall_seconds']
                assert len(totals) == timing['worker_count'] == 64
                assert abs(max(totals.values()) - timing['worker_draw_critical_path_seconds']) < 1e-8
                compute = timing['worker_draw_critical_path_seconds']
                inclusive = run['timings']['sample_seconds']
                process = timing['instrumented_wrapper_process_seconds']
                assert compute == original['worker_draw_critical_path_seconds']
                checks.append(dict(method=method, seed=seed, check='64physical_128reserved_and_worker_log_sum',
                                   status='passed', worker_logs=len(logs), workers=len(totals)))
            else:
                assert run['profile']['config']['thinning'] == 500
                assert run['profile']['config']['tpb_ss'] == 128
                assert run['host_prefault_threads'] == 8 and run['barrier'] == 'rotate'
                compute = run['profile']['timing']['gpu_sampling_s']
                inclusive = compute
                process = run['wall_including_normalization_seconds']
                assert compute == original['sampler_phase_seconds']
            if args.verify_array_hashes:
                actual_hash = sha(root / 'samples.npy')
                assert actual_hash == identity['data']['samples_sha256']
                checks.append(dict(method=method, seed=seed, check='fresh_retained_file_sha256',
                                   status='passed', samples_sha256=actual_hash))
            else:
                prior = next((item for item in prior_hash_checks
                              if item['method']==method and item['seed']==seed), None)
                if prior:
                    assert prior['samples_sha256'] == identity['data']['samples_sha256']
                    checks.append(prior)
            assay_path = R / 'cached_target_audit' / method / f'seed{seed}' / 'metrics.json'
            aggregate_path = R / 'cached_target_aggregate' / method / f'seed{seed}' / 'metrics.json'
            if reaggregation_validation.get('status')=='passed' and aggregate_path.exists():
                assay_path=aggregate_path
            assays = [('legacy_raw_model_mask878', legacy, root / 'metrics.json')]
            if assay_path.exists():
                strict = read(assay_path)
                assert strict['shape'] == [512, run['draws'], 581]
                assert strict['solver_detected_nearconstant_reactions_excluded'] == 478
                assert strict['metrics_code_sha256'] == legacy['metrics_code_sha256']
                assert strict['diagnostic_mask_geometry_sha256'] == legacy['diagnostic_mask_geometry_sha256']
                assays.append(('cached_target_mask478', strict, assay_path))
            else:
                pending.append(dict(method=method, seed=seed, expected=str(assay_path)))
            for assay, metric, path in assays:
                passed = (metric['split_rhat_median'] <= 1.01 and metric['bulk_ess_median'] >= 100
                          and metric['sample_bound_violation_fraction'] == 0
                          and metric['bounds_tolerance'] == 1e-7)
                transitions = 512 * (run['draws'] + 1000) * (2000 if method == 'cpu' else 500)
                rows.append(dict(assay=assay, method=method, seed=seed,
                    chains=512, retained_draws_per_chain=run['draws'], warmup_per_chain=1000,
                    thinning=2000 if method == 'cpu' else 500, transitions_including_warmup=transitions,
                    sampler_compute_seconds=compute, inclusive_sampler_phase_seconds=inclusive,
                    process_seconds_secondary=process, median_split_rhat=metric['split_rhat_median'],
                    p95_split_rhat=metric['split_rhat_p95'], median_bulk_ess=metric['bulk_ess_median'],
                    median_ess_per_compute_second=metric['bulk_ess_median']/compute,
                    median_ess_per_transition=metric['bulk_ess_median']/transitions,
                    sample_bound_violation_fraction=metric['sample_bound_violation_fraction'],
                    maximum_absolute_bound_violation=metric['max_absolute_bound_violation'],
                    diagnosed_reactions=metric['diagnosed_reactions'],
                    exactly_constant_sampled_reactions=metric['exactly_constant_sampled_reactions'],
                    quality_and_bounds_pass=passed, source_run=str(root), source_metrics=str(path)))
                sources[str(path)] = sha(path)
            for name in ('run.json', 'diagnostics_identity.json', 'sampling_math_timing.json', 'physical_affinity.json'):
                path = root / name
                if path.exists():
                    sources[str(path)] = sha(path)
    for field in ('estimator_code_sha256', 'geometry_sha256', 'mask_sha256', 'mask_policy_sha256',
                  'lower_bounds_sha256', 'upper_bounds_sha256', 'reaction_ids_sha256'):
        assert len({identity[field] for identity in identities}) == 1
    assert len(cpu_chain_seeds) == len(set(cpu_chain_seeds)) == 1536
    checks.append(dict(check='common_geometry_bounds_ids_legacy_mask_estimator', status='passed',
                       geometry_sha256=identities[0]['geometry_sha256']))
    groups = {}
    paired = {}
    for assay in ('legacy_raw_model_mask878', 'cached_target_mask478'):
        ar = [row for row in rows if row['assay'] == assay]
        if len(ar) != 6:
            continue
        for method in ('cpu', 'gpu_coordinate'):
            mr = [row for row in ar if row['method'] == method]
            groups[f'{assay}/{method}'] = dict(passing_seeds=sum(row['quality_and_bounds_pass'] for row in mr),
                tested_seeds=len(mr), sampler_compute_seconds_median=median(row['sampler_compute_seconds'] for row in mr),
                median_reaction_ess_median=median(row['median_bulk_ess'] for row in mr),
                median_reaction_ess_per_compute_second_median=median(row['median_ess_per_compute_second'] for row in mr))
        by_seed = {seed: {row['method']: row for row in ar if row['seed'] == seed} for seed in SEEDS}
        pairs = []
        for seed, methods in by_seed.items():
            c,g = methods['cpu'],methods['gpu_coordinate']
            pairs.append(dict(seed=seed, both_pass=c['quality_and_bounds_pass'] and g['quality_and_bounds_pass'],
                compute_time_ratio=c['sampler_compute_seconds']/g['sampler_compute_seconds'],
                effective_sampling_rate_ratio=g['median_ess_per_compute_second']/c['median_ess_per_compute_second'],
                inclusive_sampler_phase_time_ratio=c['inclusive_sampler_phase_seconds']/g['inclusive_sampler_phase_seconds'],
                process_time_ratio_secondary=c['process_seconds_secondary']/g['process_seconds_secondary']))
        cgroup,ggrp=groups[f'{assay}/cpu'],groups[f'{assay}/gpu_coordinate']
        paired[assay] = dict(rows=pairs, all_six_runs_pass=all(p['both_pass'] for p in pairs),
            compute_time_ratio_of_medians=cgroup['sampler_compute_seconds_median']/ggrp['sampler_compute_seconds_median'],
            paired_compute_time_ratio_median=median(p['compute_time_ratio'] for p in pairs),
            paired_effective_sampling_rate_ratio_median=median(p['effective_sampling_rate_ratio'] for p in pairs),
            note='Fixed512-chain budgets meet the same quality threshold; retained counts and achieved pooledESS differ. Budgets selected under historical878 assay, not re-minimized under478. Not globally minimal time-to-quality across chain counts/configurations.')
    fresh = read(B / 'results/fresh16_validation/iJO1366/summary.json')
    fresh_summary = dict(fixed_cpu_pass=len(fresh['fixed_groups']['cpu']['passing_fixed_seeds']),
        fixed_gpu_pass=len(fresh['fixed_groups']['gpu']['passing_fixed_seeds']), required=3,
        extension_comparisons=fresh['paired_seed_comparisons'],
        note='Fixed budgets do not validate legacy2.73x across fresh seeds; extensions remain separately labeled.')
    conservative16_path=B/'results/fresh16_conservative/iJO1366/summary.json'
    conservative16=read(conservative16_path) if conservative16_path.exists() else {}
    scaling_context={}
    if conservative16.get('all_six_pass') and 'cached_target_mask478/cpu' in groups:
        c16=conservative16['groups']['cpu']['ess_per_compute_second_median']
        g16=conservative16['groups']['gpu']['ess_per_compute_second_median']
        c512=groups['cached_target_mask478/cpu']['median_reaction_ess_per_compute_second_median']
        g512=groups['cached_target_mask478/gpu_coordinate']['median_reaction_ess_per_compute_second_median']
        scaling_context=dict(cpu16_ess_rate=c16,gpu16_ess_rate=g16,cpu512_ess_rate=c512,gpu512_ess_rate=g512,
            cpu_rate_increase=c512/c16,gpu_rate_increase=g512/g16,source=str(conservative16_path),
            caveat='Different selected configurations: CPU16vs64physical workers; GPU retained budgets, thinning, TPB, binary and prefault differ. Not an isolated chain-count or globally minimal time-to-target experiment.')
    report = dict(schema_version=1, generated_utc=datetime.now(timezone.utc).isoformat(),
        status='pending_cached_target_diagnostics' if pending else 'complete',
        criterion=dict(median_classical_split_rhat_max=1.01, median_pooled_rank_normalized_bulk_ess_min=100,
                       bound_tolerance=1e-7, allowed_sample_bound_violation_fraction=0),
        scopes=dict(cpu='64physical workers,128reserved logical CPUs. Max sum of timed MarkovChain.draw calls over workerPIDs; binding/state-copy/descheduling included, startup/IPC/output excluded.',
                    gpu='OneH100,512chains,rotate,TPB128,prefault8,coordinate-specialized binary. Native sampler phase includes allocation/slackGEMM/RNG init; download/write excluded.',
                    excluded='Preprocessing, diagnostics, queue waits, compilation and pilot-selection work are separate.'),
        rows=rows, groups=groups, paired=paired, pending=pending, audit_checks=checks,
        fresh16_validation=fresh_summary, source_sha256=sources,
        completed_conservative16_scaling_context=scaling_context,
        historical_mask_scope='878mask used original-model LPs.478mask only excludes certified narrow reactions on exact cachedA/b; unresolved remain active. Frozen original metrics preserved.',
        reaggregation_validation=reaggregation_validation,
        actual_array_hashes_rechecked=sum(c['check']=='fresh_retained_file_sha256' for c in checks)==6)
    if args.verify_array_hashes:
        verification_path.write_text(json.dumps(dict(generated_utc=report['generated_utc'], status='passed',
            checks=[c for c in checks if c['check']=='fresh_retained_file_sha256']), indent=2)+'\n')
    (R / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    with (R / 'comparison.csv').open('w') as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Audited iJO1366: one H100 versus 64 physical CPU cores', '',
        f"Updated {report['generated_utc']}. Audit status: **{report['status']}**.", '',
        '512 chains per method; CPU seeds 101001, 202001, 303001 have disjoint cross-replicate chain-seed ranges; '
        'Naja uses those distinct base seeds with per-chain CURAND subsequences. '
        'CPU: 12,000 retained + 1,000 warmup draws/chain, thinning 2,000. '
        'Naja: 5,000 retained + 1,000 warmup, thinning 500, rotation, TPB 128, prefault 8. '
        'Budgets were selected from first-seed pilot evidence, followed by distinct seeds 202001/303001. '
        'Both methods target the same cached 581-D polytope and original reaction bounds.', '',
        'Lead CPU time is the maximum accumulated C++ draw-call interval over 64 workers. '
        'It includes binding/state-copy and descheduling, and excludes pool/IPC/output. '
        'Lead GPU time is the native sampler phase, including allocation/slack/RNG setup and excluding download/write. '
        'Neither is labeled pure arithmetic-only. Diagnostics and preprocessing are outside both timers.', '']
    for assay in ('cached_target_mask478','legacy_raw_model_mask878'):
        ar = [row for row in rows if row['assay']==assay]
        lines += [f'## {assay}', '']
        if len(ar)!=6:
            lines += ['Conservative exact-cached-target reassay is still pending. No strict headline is assigned yet.', '']
            continue
        p=paired[assay]
        lines += [f"All six quality/bounds checks pass: **{p['all_six_runs_pass']}**. "
                  f"Median paired GPU/CPU effective-sampling rate: **{p['paired_effective_sampling_rate_ratio_median']:.3f}×**; "
                  f"CPU/GPU compute ratio of medians: **{p['compute_time_ratio_of_medians']:.3f}×**.", '',
                  '| Method | Seed | Retained/chain | Compute s | Median split-Rhat | Median bulk ESS | ESS/compute s | Bounds fraction |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in ar:
            lines.append(f"| {row['method']} | {row['seed']} | {row['retained_draws_per_chain']:,} | "
                         f"{row['sampler_compute_seconds']:.4f} | {row['median_split_rhat']:.6f} | "
                         f"{row['median_bulk_ess']:.1f} | {row['median_ess_per_compute_second']:.2f} | "
                         f"{row['sample_bound_violation_fraction']:.2g} |")
        lines += ['', 'The legacy878-mask row is a historical assay using original-model LP certificates. '
            'The conservative478-mask reassay certifies exclusions directly on cachedA/b; unresolved reactions stay active. '
            'Exactly constant sampled reactions are excluded by the unchanged frozen estimator. '
            'All-reaction bound checks are unmasked in both assays.', '']
    lines += ['## Interpretation and verification', '',
        '- This is effective throughput at 512 chains and a shared quality threshold, not globally minimal latency across chain counts. '
        'Achieved median pooledESS differs (~89kCPU/~109kGPU under the conservative assay); ESS/s accounts for that difference. '
        'Budgets were selected under the historical878 assay and have not been re-minimized under478. '
        'CPU transitions total 13.312 billion versus GPU 1.536 billion, including warmup. '
        'Algorithm/basis/thinning efficiency contributes to the result; it is not a same-transition kernel acceleration claim.',
        '- The native64-core CPU is the fastest verified core configuration in this bulk study. '
        'The finite thinning/basis search does not establish a globally optimal CPU sampler.',
        '- All64cores and sibling reservations,512chain worker logs, accumulated timers, '
        'estimator/geometry/bounds/mask identities and guarded diagnostics reuse are audited by the source script.',
        '- CPU seed101001 reuses48-core diagnostics only after whole retained-file byte identity. '
        'CPU seeds202001/303001 have independent full diagnostics. GPU first-seed diagnostics similarly require full-file identity. '
        'Different replicate seeds are independent; same-seed retimes are not additional independent observations.',
        '- Original16-chain fixed configurations passCPU2/3 andGPU1/3 fresh seeds. '
        'Separately passing bounded extensions do not rescue fixed-budget generalization or a universal2.73× claim.', '',
        'The conservative478 assay reaggregates full-budget per-reaction values produced by the unchanged '
        'frozen estimator. A fresh512-chain rerun matches all2,583 per-reaction rows and aggregate/feasibility '
        'summaries exactly. Only full-budget endpoints are reused; lower-budget prefix statistics are not. '
        '[Fresh exact-dataset validation](cached_target_audit/reaggregation_validation.json).', '',
        '[Machine-readable audit and all sources](comparison.json) · [Per-seed CSV](comparison.csv).', '',
        'Regenerate from workspace root:', '', '```bash',
        'benchmarks/env-cpu/bin/python benchmarks/scripts/audit_full64_comparison.py', '```', '',
        'Optional full retained-file SHA256 recheck:', '', '```bash',
        'benchmarks/env-cpu/bin/python benchmarks/scripts/audit_full64_comparison.py --verify-array-hashes', '```', '']
    if scaling_context:
        lines += ['## Why the bulk configuration has a larger advantage', '',
            f"The separately frozen 16-chain validation passes all three new seeds per method and yields "
            f"{c16:.2f} CPU versus {g16:.2f} GPU median reaction ESS/s. The 512-chain configurations yield "
            f"{c512:.2f} versus {g512:.2f}: CPU effective throughput rises {c512/c16:.2f}× while GPU throughput "
            f"rises {g512/g16:.2f}×. CPU workers increase from 16 to 64 physical cores. These are comparisons "
            'of independently selected configurations at a common quality threshold, not isolated chain-count '
            'scaling: retained budgets, GPU thinning, TPB, binary and prefault settings also differ. '
            '[New fixed 16-chain validation](../../fresh16_conservative/iJO1366/VALIDATION.md).', '',
            'Separate Nsight throughput profiling measures mean SM-active of 11.71% at 16 chains and 97.22% '
            'at 512 chains. That protocol uses automatic thinning 96, TPB128 and 10,000 retained draws; '
            'it qualitatively supports the occupancy explanation but is not a convergence experiment or '
            'a quantitative multiplier for these quality-matched measurements. '
            '[Profiling methods and counters](../../nsight_timeline/README.md).', '']
    (R / 'AUDITED_COMPARISON.md').write_text('\n'.join(lines))
    print(report['status'],R/'AUDITED_COMPARISON.md')


if __name__=='__main__':main()
