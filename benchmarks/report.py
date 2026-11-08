#!/usr/bin/env python3
"""Render a reproducible nine-model benchmark status/table from saved artifacts."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from summarize import collect, paired_speedups, write_csv

MODELS = ('e_coli_core', 'iAB_RBC_283', 'iJR904', 'iAF1260', 'iJO1366',
          'iML1515', 'iMM904', 'RECON1', 'Recon3D')
SEEDS = (101, 202, 303)
METHODS = ('naja', 'hopsy default', 'hopsy jadebeck')


def fmt(value):
    return '—' if value is None else f'{value:.4g}'


def median(rows, field):
    vals = [r[field] for r in rows if r.get(field) is not None]
    return statistics.median(vals) if vals else None


def variant(row):
    if row['method'] == 'naja':
        config = (row.get('barrier'), row.get('threads_per_block'), row.get('pair_probability'), row.get('thinning_label'))
        if config == ('rotate', 128, 0.0, 0):
            return 'naja'
        return f"naja (barrier={config[0]},TPB={config[1]},pair={config[2]},thin={config[3]})"
    label = f"hopsy {row['thinning_label']}"
    return label if row.get('workers') in (16, None) else label + f" ({row['workers']} worker)"


def supplementary_sections(root):
    lines = ['', '## Measured time to quality', '',
             'Criterion: masked median classical split-Rhat ≤ 1.01 and median pooled bulk ESS ≥ 100. These are separately executed exact-budget sampling times, including startup, 1,000 retained-draw warmup, writes and Naja adapter. Prefixes choose the budget; their interpolated times are not used. Search/diagnostic costs are excluded. The tested budget grid and CPU variants are finite, so these are not globally optimal stopping times. Bounds are checked separately from the convergence criterion.', '',
             '[Measured quality records and per-seed comparisons](quality/summary.json). Incomplete seed groups remain explicitly incomplete. [Figure SVG](figures/measured_quality_time.svg) · [plotted source CSV](figures/measured_quality_source.csv).', '',
             '![Measured passing-budget times with observed seed ranges](figures/measured_quality_time.png)', '',
             '| Model | Variant | Passing measured seeds | Median selected draws/chain | Median process wall s | Max sample bound violation fraction (1e-7) |',
             '|---|---|---:|---:|---:|---:|']
    quality_path = root / 'results/quality/summary.json'
    if quality_path.exists():
        quality = json.loads(quality_path.read_text())
        for group in quality.get('groups', []):
            bounds = []
            for row in quality.get('measured_rows', []):
                if row['model'] == group['model'] and row['variant'] == group['variant']:
                    try:
                        metric = json.loads((Path(row['run_dir']) / 'metrics.json').read_text())
                        bounds.append(metric['sample_bound_violation_fraction'])
                    except (OSError, json.JSONDecodeError, KeyError):
                        pass
            lines.append('| ' + ' | '.join([group['model'], group['variant'],
                f"{group['measured_seed_count']}/3", fmt(group['median_budget']),
                fmt(group['median_measured_seconds']), fmt(max(bounds) if bounds else None)]) + ' |')
    else:
        lines.append('| Pending | | | | | |')
    lines += ['', '## Empirical uniformity agreement', '',
              '[Full marginal and correlation comparison](uniformity/summary.json). e_coli_core only: one million retained samples per method, two independently seeded hopsy references, Naja rotate, Naja whiten plus pair probability 0.5, and a separately verified whiten/Jacobi-pair/Dikin mixture (pair 0.5, Dikin 0.15, coordinate 0.35; 11 runtime directions). Common model-derived reaction mask. This tests empirical agreement on e_coli_core, not a general uniformity proof. [Mixture execution evidence](uniformity/naja_whiten_dikin_pair/mixture_verification.json).', '',
              '| Comparison against hopsy reference 1 | Median W1 / reference std | p95 W1 / reference std | Correlation Frobenius distance |',
              '|---|---:|---:|---:|']
    uniformity_path = root / 'results/uniformity/summary.json'
    if uniformity_path.exists():
        uniformity = json.loads(uniformity_path.read_text())
        for name, value in uniformity['comparisons'].items():
            norm = value['wasserstein1_divided_by_pooled_reference_std']
            lines.append('| ' + ' | '.join([name, fmt(norm['median']), fmt(norm['p95']),
                                           fmt(value['correlation_frobenius_distance'])]) + ' |')
        lines += ['', f"Active reactions: {uniformity['active_reaction_count']}. The hopsy2401 row is the independent-reference floor. This artifact uses bound tolerance 1e-6; the fixed-budget/quality tables above use 1e-7, so their violation fractions have different definitions."]
    else:
        lines.append('| Pending | | | |')
    lines += ['', '## H100 chain and thread-block scaling', '',
              '[Full scaling timings and sampled telemetry](scaling/iJO1366/summary.json). iJO1366, seed 101, rotate, automatic thinning, 10,000 retained plus 1,000 warmup draws per chain; one factor changes at a time. These exploratory runs are excluded from the fixed-budget model table. [Figure SVG](figures/h100_chain_scaling.svg) · [plotted source CSV](figures/chain_scaling_source.csv).', '',
              '![H100 chain scaling: GPU phase versus whole-run retained throughput](figures/h100_chain_scaling.png)', '',
              '| Case | Chains | TPB | Retained samples / sampling s | Native process s | Adapter s | Sampled peak GPU MiB |',
              '|---|---:|---:|---:|---:|---:|---:|']
    scaling_path = root / 'results/scaling/iJO1366/summary.json'
    if scaling_path.exists():
        scaling = json.loads(scaling_path.read_text())
        for row in scaling['rows']:
            lines.append('| ' + ' | '.join([row['case'], str(row['chains']), str(row['tpb']),
                fmt(row['retained_samples_per_sampling_second']), fmt(row['native_process_wall_seconds']),
                fmt(row['normalization_seconds']), fmt(row['sampled_peak_gpu_memory_mib'])]) + ' |')
        lines += ['', 'Telemetry is sampled once per second and can miss memory peaks. Feasibility checks cover only 1,024 deterministic reduced-space samples per case at tolerance 1e-6; this section is throughput scaling, not an equal-quality comparison.']
    else:
        lines.append('| Pending | | | | | | |')
    lines += ['', '## CPU source-build optimization pilot (one seed)', '',
              '[Three-way pilot CSV](native_cpu/pilot_table.csv) · [provenance and summary](native_cpu/pilot_summary.json). This is a separate iJO1366 seed101 build experiment: 16 chains, 10,000 retained plus 1,000 warmup draws, thinning 2,000. It is excluded from the frozen wheel-based canonical table. Native three-seed passing-budget measurements are reported only when completed; this pilot is not a three-seed fastest-CPU claim.', '',
              '| Build | hopsy build ID | Flags | Process s | Sampling s | Median Rhat | Median bulk ESS | Bound violation fraction (1e-7) |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    native_path = root / 'results/native_cpu/pilot_summary.json'
    if native_path.exists():
        native = json.loads(native_path.read_text())
        for row in native['rows']:
            lines.append('| ' + ' | '.join([row['variant'], row['build'], row['flags'],
                fmt(row['process_seconds']), fmt(row['sample_seconds']), fmt(row['split_rhat_median']),
                fmt(row['bulk_ess_median']), fmt(row['bound_violation_fraction'])]) + ' |')
        lines += ['', f"The controlled source_generic/native comparison uses the same official SDIST (build 9b787d9), GCC 11.4.0 and runtime; only native ISA/tuning flags differ. Observed native speedup: {native['native_process_speedup_vs_source_generic']:.3f}× process, {native['native_sampling_speedup_vs_source_generic']:.3f}× sampling. The installed wheel reports build dadab7b, so wheel/native differences alone do not isolate compiler flags. Analytic cube checks passed. Rhat is near the cutoff and changes slightly with floating-point trajectories; these results do not establish improved mixing."]
    else:
        lines.append('| Pending | | | | | | | |')
    lines += ['', '## Experimental wide-index 512-chain probe', '',
              '[Safe-build scaling summary](scaling-extension-safe/iJO1366/summary.json) · [build/correctness proof](safe-build-verification/wide-index-proof.json). This uses a separate cast-only wide-index binary, SHA256 `da263d6060416d96da366392210b9468422890bda8200718da452941436a8f18`; the canonical binary remains `c7baf4…`. Results are not pooled into the frozen canonical table or the original scaling curve.', '',
              '| Case | Chains | TPB | Retained samples / sampling s | Native process s | Adapter s | Sampled peak GPU MiB |',
              '|---|---:|---:|---:|---:|---:|---:|']
    safe_path = root / 'results/scaling-extension-safe/iJO1366/summary.json'
    if safe_path.exists():
        safe = json.loads(safe_path.read_text())
        for row in safe.get('rows', []):
            lines.append('| ' + ' | '.join([row['case'], str(row['chains']), str(row['tpb']),
                fmt(row['retained_samples_per_sampling_second']), fmt(row['native_process_wall_seconds']),
                fmt(row['normalization_seconds']), fmt(row['sampled_peak_gpu_memory_mib'])]) + ' |')
        lines += ['', 'At 512 chains TPB128 was faster than TPB512; the small-chain tuning result does not generalize automatically. These are single-pass throughput probes with full saved-array finite checks and only 1,024 sampled feasibility checks, not full-output convergence diagnostics. One-second memory peaks are sampled, not exact.']
    else:
        lines.append('| Pending | | | | | | |')
    lines += ['', 'The original frozen-binary 512-chain attempts were invalid: signed 32-bit element-size products overflowed above INT_MAX, and CUDA/cuBLAS errors occurred despite process exit status zero. Those records remain [explicitly invalid](scaling-extension/iJO1366/summary.json) and supply no throughput evidence. The separate cast-only fix passed all five GPU tests and a same-seed BOX comparison bitwise. Runner guards now reject backend errors, unsupported INT_MAX products, nonfinite output and wrong dimensions; see [guard tests](runner-guards-test.log). The wide-index patch is scoped to the reduced-output path and does not claim all other large-array paths were widened.', '']
    lines += ['', '## Paired reduced-export precision experiment', '']
    precision_path = root / 'results/precision/RECON1/seed303/summary.json'
    if precision_path.exists():
        p = json.loads(precision_path.read_text())
        f64 = p['representations']['float64_export']
        quant = p['representations']['paired_float32_quantization']
        lines += ['[Same-trajectory RECON1 seed303 evidence](precision/RECON1/seed303/summary.json) · [float64 build proof](f64-build-verification/verification.json). One unchanged double-precision trajectory is backmapped as T@Z64+shift and T@float64(float32(Z64))+shift, using float64 backmaps. The quantized trajectory reproduces the historical frozen float32 output bitwise. This isolates export quantization, rather than comparing different stochastic trajectories.', '',
                  f"Maximum varying/nonexcluded reaction-bound error falls from {quant['max_nonexcluded_reaction_bound_violation']:.8g} to {f64['max_nonexcluded_reaction_bound_violation']:.8g}; the float64 full-output maximum is {f64['max_absolute_reaction_bound_violation']:.8g}, confined to LP-excluded nearconstant reactions. At tolerance 1e-7, the unfiltered float64 any-reaction fraction remains {f64['sample_bound_violation_fraction']['1e-07']:.8g}. Float64 therefore resolves the observed large seed303 export errors, but does not establish strict 1e-7 compliance, eliminate geometry residue, or validate every model/seed. No bounds filtering or clipping was applied. The experimental binary is `{p['new_binary_sha256']}`; canonical timings/results retain the frozen float32 build.", '']
    else:
        lines.append('Pending paired precision evidence.')
    lines += ['', '## Opt-in host first-touch experiment', '']
    prefault_path = root / 'results/host_prefault/iJO1366/summary.json'
    if prefault_path.exists():
        p = json.loads(prefault_path.read_text())
        off, on = p['medians']['0'], p['medians']['8']
        lines += ['[Three paired repetitions](host_prefault/iJO1366/summary.json) · [build proof](prefault-build-verification/verification.json). iJO1366, seed101, 512 chains, TPB128, 10,000 retained plus 1,000 warmup draws/chain; one H100 and eight allocated CPUs. Same experimental binary, fresh processes, alternating off/on, on/off, off/on order. All retained arrays are SHA256-identical. This is a throughput/transfer test, excluded from canonical quality comparisons.', '',
                  f"Median download phase including allocation, first-touch threads, joins, tail-byte touch and cudaMemcpy: {off['download_including_prefault_seconds']:.4g} → {on['download_including_prefault_seconds']:.4g} s. Native process: {off['native_process_seconds']:.4g} → {on['native_process_seconds']:.4g} s; process plus normalization: {off['end_to_end_seconds']:.4g} → {on['end_to_end_seconds']:.4g} s. Median paired speedups are {p['median_paired_download_speedup']:.3f}× download, {p['median_paired_native_process_speedup']:.3f}× native process and {p['median_paired_end_to_end_speedup']:.3f}× including normalization. The binary is `{p['binary_sha256']}`; first-touch defaults off. This single model/seed result does not establish general speedup, convergence, or exact memory peaks.", '']
    else:
        lines.append('Pending paired first-touch evidence.')
    lines += ['', '## CPU physical-core placement control', '',
              'The earlier 16-worker native CPU measurements reserved 16 logical CPUs, which can place sibling hardware threads on only eight physical cores. This separate placement control reserves 32 logical CPUs and pins 16 workers to one hardware thread on each of 16 physical cores. Same native build and iJO1366 thinning 2,000; 16 chains, 10,000 retained plus 1,000 discarded warmup draws/chain. It is excluded from the frozen canonical table and listed independently from packed-worker quality groups. The earlier 1.65× iJO1366 quality-search comparison uses the packed-worker CPU group; the physical-core comparison below supersedes it for a best-tested CPU claim.', '',
              '| Seed | Process s | Masked median Rhat | Median bulk ESS | Bound fraction (1e-7) | Passing | Evidence |',
              '|---|---:|---:|---:|---:|---|---|']
    physical_rows = []
    for seed in SEEDS:
        directory = root / f'results/native_cpu/iJO1366/physical16_thin2000/seed{seed}'
        try:
            run = json.loads((directory / 'run.json').read_text())
            metric = json.loads((directory / 'metrics.json').read_text())
            affinity = json.loads((directory / 'physical_affinity.json').read_text())
        except (OSError, json.JSONDecodeError):
            lines.append(f'| {seed} | — | — | — | — | pending | |')
            continue
        verified = (run.get('status') == 'complete' and metric.get('schema_version', 0) >= 2
                    and metric.get('diagnostic_mask_status') == 'model_lp_verified'
                    and affinity.get('selected_physical_core_count') == 16)
        passing = verified and metric['split_rhat_median'] <= 1.01 and metric['bulk_ess_median'] >= 100
        seconds = run.get('process_wall_seconds')
        if passing and seconds is not None:
            physical_rows.append((seed, seconds))
        lines.append('| ' + ' | '.join([str(seed), fmt(seconds), f"{metric['split_rhat_median']:.7g}",
            fmt(metric.get('bulk_ess_median')), fmt(metric.get('sample_bound_violation_fraction')),
            'yes' if passing else 'no/unverified',
            f'[run/affinity](native_cpu/iJO1366/physical16_thin2000/seed{seed}/physical_affinity.json)']) + ' |')
    lines += ['', f'Completed verified passing physical-core seeds: {len(physical_rows)}/3. Three-seed ratios require all three seeds to pass. Full-process CPU timing includes imports, startup, warmup, sampling and writes. Affinity/resource metadata is retained per seed; node load remains shared.', '']
    if len(physical_rows) == 3:
        quality_path = root / 'results/quality/summary.json'
        if quality_path.exists():
            quality = json.loads(quality_path.read_text())
            gpu = {r['seed']: r['measured_process_wall_seconds'] for r in quality.get('measured_rows', [])
                   if r['model'] == 'iJO1366' and r['variant'] == 'naja512'}
            if all(seed in gpu for seed, _ in physical_rows):
                ratios = [seconds / gpu[seed] for seed, seconds in physical_rows]
                lines += [f"Against these three physical-core CPU seeds, median CPU process wall is {statistics.median(seconds for _, seconds in physical_rows):.4g} s; Naja TPB512 median inclusive wall is {statistics.median(gpu[seed] for seed, _ in physical_rows):.4g} s. Median paired CPU/Naja ratio: {statistics.median(ratios):.3f}×. GPU uses 50,000 retained draws/chain versus CPU 10,000; both meet the declared median criterion. This is a tested selected-budget comparison, not a global optimum.", '']
    thinning_path = root / 'results/gpu_thinning/iJO1366/summary.json'
    if thinning_path.exists():
        sweep = json.loads(thinning_path.read_text())
        lines += ['', '## GPU thinning: newly measured selected budgets', '',
                  '[Sweep, pilot costs and preserved attempts](gpu_thinning/iJO1366/summary.json). iJO1366, frozen c7baf4 binary, 16 chains, TPB512, seeds 101/202/303. Every final is a fresh exact-budget process with fresh schema2 verified-mask diagnostics. Wall includes initialization, 1,000 discarded retained warmup draws/chain, sampling, writes and normalization. Pilot, prefix-search and diagnostic costs are excluded and recorded separately; changing thinning changes warmup move count.', '',
                  '| Thinning | Passing final seeds | Retained draws/chain | Median inclusive s | Median Rhat | Median bulk ESS | Max bound fraction (1e-7) |',
                  '|---:|---:|---:|---:|---:|---:|---:|']
        cpu_path = root / 'results/native_cpu/iJO1366/physical16_thin2000/summary.json'
        cpu = json.loads(cpu_path.read_text()) if cpu_path.exists() else {}
        cpu_times = {r['seed']: r['process_seconds'] for r in cpu.get('rows', []) if r.get('quality_met')}
        for group in sweep['groups']:
            passing = [r for r in group['rows'] if r.get('status') == 'complete' and r.get('quality_met')]
            if not passing:
                lines.append(f"| {group['thinning']} | 0/3 (pilot cap) | — | — | — | — | — |")
                continue
            lines.append('| ' + ' | '.join([str(group['thinning']), f'{len(passing)}/3',
                fmt(statistics.median(r['draws_per_chain'] for r in passing)),
                fmt(statistics.median(r['wall_seconds'] for r in passing)),
                f"{statistics.median(r['split_rhat_median'] for r in passing):.7g}",
                fmt(statistics.median(r['bulk_ess_median'] for r in passing)),
                fmt(max(r['sample_bound_violation_fraction'] for r in passing))]) + ' |')
            if group['thinning'] == 1000 and len(passing) == 3 and all(r['seed'] in cpu_times for r in passing):
                paired = statistics.median(cpu_times[r['seed']] / r['wall_seconds'] for r in passing)
                ratio = statistics.median(cpu_times.values()) / statistics.median(r['wall_seconds'] for r in passing)
                lines += ['', f'Best completed tested iJO1366 GPU configuration: thinning 1,000, TPB512, 2,000 retained draws/chain, median inclusive wall {group["measured_process_seconds_median"]:.6g} s versus physical-core native CPU 25.8015 s (10,000 retained draws/chain, thinning 2,000). Ratio of medians {ratio:.4f}×; median paired CPU/GPU {paired:.4f}×. CPU uses 16 physical cores with 32 logical CPUs reserved, not all 128 logical CPUs. The earlier automatic-thinning 96 / 50,000-draw GPU result 21.5712 s is historical and superseded for best-tested quality timing. This is a selected-budget comparison, not equal sample count or a global optimum. Near-cutoff Rhat and tuning on the same three seeds motivate fresh nonoverlapping seeds for the separate 512-chain study underway; no result from that study is inferred here.', '']
        lines += ['[Best-use-case interpretation and throughput limits](BEST_USE_CASE.md).', '']
    followups = sorted((root / 'results/native_cpu').glob('*/physical16_*_jade/summary.json'))
    if followups:
        lines += ['', '## Additional physical-core CPU controls', '',
                  'Matched wheel/native follow-ups reserve 32 logical CPUs and pin 16 workers to 16 physical cores. Only independently validated saved group summaries enter this table; configuration/build groups remain separate. These exact selected-budget executions retain discarded warmup, import/startup and write costs. They do not alter the frozen canonical table.', '',
                  '| Model | Build/configuration | Passing seeds | Median process s | Artifact |',
                  '|---|---|---:|---:|---|']
        for path in followups:
            value = json.loads(path.read_text())
            rows = value.get('rows', [])
            passing = [r for r in rows if r.get('quality_met') and r.get('bitwise_same_as_canonical')]
            lines.append('| ' + ' | '.join([value.get('model', path.parents[1].name),
                value.get('variant', path.parent.name), f'{len(passing)}/3',
                fmt(statistics.median(r['process_seconds'] for r in passing)) if passing else '—',
                f'[validated group]({path.relative_to(root / "results")})']) + ' |')
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / 'results'
    results.mkdir(exist_ok=True)
    rows = collect(results)
    canonical = [r for r in rows if r.get('canonical_build_eligible')
                 and r.get('experiment_family') in ('naja', 'hopsy')
                 and r['model'] in MODELS and r['seed'] in SEEDS
                 and r['chains'] == 16 and r['draws'] == 10000 and r['warmup'] == 1000]
    grouped = defaultdict(list)
    for row in canonical:
        grouped[(row['model'], variant(row))].append(row)
    lines = ['# Naja H100 benchmark status', '',
             f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}. Values below come from saved run/metrics files; absent results remain absent.", '',
             'Hardware/software: sampling runs use the Slurm node with Intel Xeon Platinum 8462Y+ CPUs (128 logical CPUs available) and one NVIDIA H100 80 GB HBM3 per GPU run, driver 580.126.20. Build toolkit: CUDA nvcc 12.9.86 (the driver reports CUDA 13.0 compatibility). CPU baseline: hopsy 1.7.0, PolyRound 0.5.0, COBRApy 0.32.1. Naja base SHA 23873f01d3268a79bbf84499a66adce5e88cc6cb plus saved localized patch; benchmark binary SHA256 c7baf4f0ed282b936ac0dbac4b896bfbfc040eb06da6f84aff6605fddf4b0fc6. [Captured environment](setup/h100-environment.txt).', '',
             '## Matched protocol', '',
             'One shared PolyRound-preprocessed polytope per model; 16 chains, 1,000 discarded retained-draw warmup, then 10,000 retained draws per chain. Planned seeds: 101, 202, 303. Raw BiGG bounds and medium are preserved; no objective/growth constraint is added. Preprocessing uses PolyRound/GLPK, independent of the expired Gurobi license.', '',
             'Naja uses one H100, barrier rotation, automatic geometry-based thinning, zero global/delta relaxation, and no filtering before these diagnostics. hopsy uses uniform CHRR with 16 multiprocessing workers (not 128 active workers), plus explicitly labeled one-worker optimization runs where present; either API default thinning 1 or the recorded Jadebeck guideline ceil(d²/6). All samples are backmapped by the same affine transform for common reaction-space diagnostics.', '',
             'Wall time includes model load, initialization, discarded warmup, sampling and output writing. Naja standardized-output conversion is also included conservatively in the primary wall comparison, but it is harness overhead (a second output copy), not required Naja work. Native CLI wall and adapter time are displayed separately. CPU full-process times include Python/NumPy/hopsy imports. Older CPU records lacking process_wall_seconds are explicitly marked startup-excluded and remain provisional; they are not pooled with full-process times or used in speedup ratios. Preprocessing and external diagnostics are reported separately/excluded equally. Naja native output is float32; hopsy output is float64. Timing scopes and phase times remain in each run.json.', '',
             'Classical split-Rhat and ArviZ pooled rank-normalized bulk ESS are computed per reaction, excluding model-derived solver-detected nearconstant reactions (original-model GLPK min/max flux range ≤ 1e-7, feasibility tolerance 1e-9) plus exactly constant sampled reactions. The same cached model mask is used for both methods; legacy method-dependent 1e-12 numerical-range summaries remain in metrics.json for audit. Only schema-2 metrics with a verified model mask populate primary quality cells. Table cells are medians across completed seeds of each per-run statistic; Rhat p95 is the median across seeds of the within-run reaction p95. Bounds report the fraction of samples with any reaction-bound violation exceeding 1e-7, before filtering; exact zero-tolerance counts are preserved in metrics.json.', '',
             '## Nine-model table', '',
             '| Model | n / d / m | PolyRound s | Method | Actual thinning | Seeds complete / diagnosed | Matched wall s / timing scope | Naja native CLI s | Naja adapter s | Retained samples/s | Median Rhat | Rhat p95 | Median bulk ESS | Bound violation fraction | Status |',
             '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for model in MODELS:
        meta_path = root / 'data/models' / model / 'meta.json'
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        validation_path = root / 'data/models' / model / 'validation.json'
        validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
        dims = ' / '.join(str(meta.get(k, '—')) for k in ('n', 'd', 'm'))
        prep = meta.get('timings', {}).get('polyround_seconds')
        additional = sorted({variant(r) for r in canonical if r['model'] == model} - set(METHODS))
        for method in (*METHODS, *additional):
            all_runs = grouped[(model, method)]
            rejected = [r for r in all_runs if r.get('duplicate_canonical_seed_rejected')]
            completed = [r for r in all_runs if r['status'] == 'complete' and not r.get('duplicate_canonical_seed_rejected')]
            diagnosed = [r for r in completed if r['metrics_present']
                         and r['diagnostic_mask_status'] == 'model_lp_verified'
                         and (r['metrics_schema_version'] or 0) >= 2]
            thinning = ','.join(str(v) for v in sorted(set(r['actual_thinning'] for r in completed if r['actual_thinning'] is not None))) or '—'
            states = defaultdict(int)
            for r in all_runs:
                states[r['status']] += 1
            state = ', '.join(f'{k}:{v}' for k, v in sorted(states.items())) or ('not run' if meta else 'preprocessing pending/missing')
            if rejected:
                state += '; DUPLICATE CONFIG/SEED REJECTED: ' + ','.join(str(v) for v in sorted(set(r['seed'] for r in rejected)))
            scopes = sorted(set(r['timing_scope'] for r in completed))
            full_timing = [r for r in completed if r['timing_scope'] != 'cpu_startup_excluded_provisional']
            timing_rows = full_timing or completed
            timing_note = '; '.join(scopes)
            if full_timing and len(full_timing) < len(completed):
                timing_note += f'; wall uses {len(full_timing)} full-process runs only'
            display_model = model + ' (precision-limited)' if validation.get('strict_1e_minus_7_passed') is False else model
            cells = [display_model, dims, fmt(prep), method, thinning,
                     f"{len(set(r['seed'] for r in completed))}/3; {len(set(r['seed'] for r in diagnosed))} diagnosed",
                     fmt(median(timing_rows, 'matched_output_wall_seconds')) + ('; ' + timing_note if timing_note else ''),
                     fmt(median(completed, 'native_wall_seconds')) if method.startswith('naja') else '—',
                     fmt(median(completed, 'normalization_seconds')) if method.startswith('naja') else '—',
                     fmt(median(timing_rows, 'retained_samples_per_second')),
                     fmt(median(diagnosed, 'split_rhat_median')),
                     fmt(median(diagnosed, 'split_rhat_p95')),
                     fmt(median(diagnosed, 'bulk_ess_median')),
                     fmt(median(diagnosed, 'bound_violation_fraction')), state]
            lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'RECON1 is precision-limited: its unchanged GLPK/presolve geometry passed a separately declared preprocessing verification gate of 1e-6; the strict 1e-7 check failed (center bound residual 1.1405e-7, pilot residual 1.5282e-7). This is not a relaxation of the sampling target: original bounds and arrays are unchanged. All common sample metrics still use 1e-7 and exact zero tolerance over every reaction. [Structured preprocessing attempts](setup/RECON1_preprocessing_attempts.json); strict failure evidence remains beside the model validation. CSV fields preserve the verification tolerance and strict-pass flag.', '']
    recon_runs = [r for r in canonical if r['model'] == 'RECON1' and r['method'] == 'naja' and r['status'] == 'complete']
    recon_maxima = ', '.join(f"seed {r['seed']}: {fmt(r['max_bound_violation'])}" for r in sorted(recon_runs, key=lambda r: r['seed']) if r['max_bound_violation'] is not None)
    if recon_maxima:
        lines += ['', 'RECON1 output-bound warning: the preprocessing center/pilot tolerance does not guarantee later sample accuracy. Current float32 reduced-output Naja runs have maximum absolute reaction-bound violations ' + recon_maxima + ', substantially larger than the 1e-6 preparation gate. The approximately 0.59 any-reaction sample-violation fraction is largely associated with nearly blocked reactions, while the coordinate fraction is much smaller; the larger errors also affect varying reactions. The same-trajectory float64/export-quantization test below establishes export quantization as the cause of the large seed303 varying-reaction errors; small LP-excluded residuals remain in full precision. All unfiltered fractions/maxima remain in CSV/metrics; passing Rhat/ESS would not imply bound compliance.', '']
    lines += ['', 'Preprocessing hardware differs: e_coli_core was preprocessed once on the controller (Xeon Platinum 8562Y+); the other prepared models ran on the Slurm node (Xeon Platinum 8462Y+). These are individual recorded preparation times, not a controlled cross-model CPU scaling comparison. Per-model meta.json and all_runs.csv preserve preprocessing hostnames. All benchmark sampling uses the Slurm H100 node.', '']
    pairs = paired_speedups(canonical)
    write_csv(results / 'paired_equal_sample_speedups.csv', pairs)
    write_csv(results / 'all_runs.csv', rows)
    lines += ['', '## Equal-sample speedups', '',
              'Each row pairs the same model, seed, chain count, retained budget, warmup and reduced dimension. CPU is the fastest completed baseline with full-process timing for that seed; provisional CPU timings that exclude Python/package startup are excluded from speedup pairs. Both methods’ quality is shown; a speedup here is not an equal-quality claim. Missing CPU variants can change which baseline is fastest.', '',
              '| Model | Seed | Fastest CPU / thinning | CPU wall s | Naja inclusive / native / adapter s | CPU/Naja | CPU Rhat / ESS | Naja Rhat / ESS |',
              '|---|---:|---|---:|---:|---:|---:|---:|']
    for pair in pairs:
        lines.append('| ' + ' | '.join([pair['model'], str(pair['seed']),
            f"{pair['fastest_cpu_method']} / {pair['fastest_cpu_thinning']} / workers={pair['fastest_cpu_workers']}", fmt(pair['cpu_seconds']),
            f"{fmt(pair['naja_seconds'])} / {fmt(pair['naja_native_cli_seconds'])} / {fmt(pair['naja_adapter_seconds'])}", fmt(pair['equal_sample_speedup']),
            f"{fmt(pair['cpu_median_rhat'])} / {fmt(pair['cpu_median_ess'])}",
            f"{fmt(pair['naja_median_rhat'])} / {fmt(pair['naja_median_ess'])}"]) + ' |')
    if not pairs:
        lines.append('| No completed diagnosed matched pairs yet | | | | | | | |')
    lines += ['', '## Naja native phase times (seconds)', '',
              'Median per phase across completed canonical seeds. These internal profiler times exclude process startup and standardized-output conversion. Bins overlap: on the reduced rotated path, backtransform_s includes download_from_gpu_s. Do not sum phase columns or equate their sum with externally measured wall time.', '',
              '| Model | Barrier | Load | Barrier | Upload | Sample | Download | Backtransform (overlapping) | Write | Internal total |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    phase_keys = ('load_polytope_s', 'barrier_rotation_s', 'upload_to_gpu_s', 'gpu_sampling_s',
                  'download_from_gpu_s', 'backtransform_s', 'write_output_s', 'total_s')
    for model in MODELS:
        gpus = [r for r in canonical if r['model'] == model and r['method'] == 'naja' and r['status'] == 'complete']
        for barrier in sorted(set(r.get('barrier') or 'off' for r in gpus)):
            selected = [r for r in gpus if (r.get('barrier') or 'off') == barrier]
            lines.append('| ' + ' | '.join([model, barrier] + [fmt(median(selected, k)) for k in phase_keys]) + ' |')
    lines += ['', '## Model-derived diagnostic masks', '',
              'Candidate selection uses max(abs(T row)) ≤ 1e-6, followed by original-model min/max GLPK LPs. Only successful solver-detected ranges ≤ 1e-7 are excluded; unresolved or wider ranges remain active. This does not modify sampling targets or any bound checks. Empirical backmapped ranges above the LP tolerance expose affine-transform numerical residue and remain visible here.', '',
              '| Model | LP-nearconstant excluded / total | LP mask seconds | Maximum measured range among excluded reactions |',
              '|---|---:|---:|---:|']
    for model in MODELS:
        mask_path = root / 'data/models' / model / 'diagnostic_mask.json'
        if not mask_path.exists():
            lines.append(f'| {model} | pending | — | — |')
            continue
        mask = json.loads(mask_path.read_text())
        selected = [r for r in canonical if r['model'] == model and r['diagnostic_mask_status'] == 'model_lp_verified']
        ranges = [r['solver_excluded_empirical_range_max'] for r in selected if r['solver_excluded_empirical_range_max'] is not None]
        lines.append('| ' + ' | '.join([model, f"{mask['excluded_nearconstant_reactions']} / {mask['n_reactions']}",
                                        fmt(mask['wall_seconds']), fmt(max(ranges) if ranges else None)]) + ' |')
    failures = [r for r in rows if r['status'] in ('failed', 'timeout', 'cancelled')]
    lines += ['', '## Failures and incomplete work', '']
    lines += [f"- {r['model']} / {r['method']} / seed {r['seed']}: {r['status']}; `{Path(r['run_dir']).relative_to(root)}`" for r in failures]
    if not failures:
        lines.append('No failure/timeout run records found. Jobs terminated before writing run.json are not automatically counted as completed or failed; inspect saved Slurm logs.')
    lines += supplementary_sections(root)
    lines += ['', '## Limits on interpretation', '',
              '- Prefix budgets (100, 250, 500, 1,000, 2,000, 5,000, 10,000 retained draws per chain) identify when median Rhat ≤ 1.01 and median bulk ESS ≥ 100 in the saved trajectory. Any prefix wall-time estimate is linear interpolation, not a measured time-to-quality experiment.',
              '- Shared initial points with independent RNGs do not constitute overdispersed initialization. Rhat alone does not prove uniformity; analytic GPU tests and separate long-reference comparisons are required.',
              '- This prototype uses the proposed public BiGG panel. New results do not independently verify historical paper runs; source provenance is saved in `audit/data_inventory.md`.',
              '- MATLAB was not available in the compute-node PATH; hopsy is the current CPU baseline. No Dingo/COBRA benchmark is silently implied.',
              '- The production knockout corpus rebuild is outside this prototype scope. Conditioned, ablation, uniformity and scaling experiments have separate artifacts; absence from this model table does not imply their results.',
              '- Only results/naja/** and results/hopsy/** enter the fixed-budget table and its speedup pairs. Scaling, quality-budget, uniformity, seed-reproducibility and pilot families are excluded even when their seed/shape matches. Duplicate canonical configurations for the same seed are explicitly rejected. Exploratory runs are excluded from this table and retained in all_runs.csv.', '',
              'GPU correctness: all five analytic/device tests passed with production `--use_fast_math`; see `logs/correctness-fastmath-255.log`. The benchmark binary SHA and localized source patch are preserved under results/.', '',
              '[Independent final review](../audit/final_review.md).', '',
              'Source records: `data/raw/manifest.json`, `data/models/*/meta.json`, `results/**/run.json`, `results/**/metrics.json`, `logs/`, and environment locks. Run `python summarize.py` and `python report.py` to regenerate.', '']
    output = results / 'BENCHMARK_TABLE.md'
    output.write_text('\n'.join(lines))
    print(output)

if __name__ == '__main__':
    main()
