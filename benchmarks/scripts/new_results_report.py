"""Refresh only the report opened at the user's utilization-plot request."""
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import statistics

B = Path(__file__).resolve().parents[1]
R = B / 'results'
from sampling_benchmarks import generate
generate()  # Separate phase/source tables; never rewrites the historical report.


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def number(x):
    return 'pending' if x is None else f'{x:.5g}'


lines = ['# Naja: results since the utilization-plot request', '',
    f"Updated {datetime.now(timezone.utc).isoformat(timespec='seconds')}.", '',
    'New profiling, sampling analyses and experiments from this phase of the campaign. '
    '[Earlier campaign report](BENCHMARK_TABLE.md) is retained separately. '
    'Lead metrics are sampler time to a shared quality target and effective samples per sampling second; '
    'load, correction, transfer, writing and total process time remain separate. '
    'Pure GPU kernel timing and inclusive CPU sampling-phase proxies are explicitly distinguished.', '']
lines += ['[Cross-model sampling-phase source table](SAMPLING_BENCHMARKS.md) · '
          '[CSV](SAMPLING_BENCHMARKS.csv) · [Full source/provenance JSON](SAMPLING_BENCHMARKS.json). '
          'This provisional phase table includes CPU pool overhead; isolated worker draw-call timing is '
          'reported separately as it becomes available.', '']

continuation = read(B / 'audit/continuation-20261108/window.json')
if continuation:
    lines += ['The overnight campaign completed its deadline. A user-authorized three-hour '
        'continuation is auditing completed results and extending selected comparisons. '
        'Original-model LP masks and conservative cached-target masks are reported as separate '
        'diagnostic assays; saved samples and frozen earlier metrics are preserved. '
        '[Continuation window](../audit/continuation-20261108/window.json).', '']

worker_groups = read(R / 'WORKER_MATH_BENCHMARKS.json')
if worker_groups and worker_groups.get('gpu_sampler_phase_comparisons'):
    lines += ['## Sampling efficiency with CPU worker timing', '',
        'Three seeds per completed group. CPU timing accumulates C++ draw calls per worker '
        'and reports the critical path, excluding pool startup, IPC and output-array assembly. '
        'Binding/state-copy and descheduling remain included. GPU timing is the native sampler phase, '
        'including allocation, initial slack GEMM and RNG initialization, excluding download/writes. '
        'These explicit scopes supersede inclusive CPU pool timing for the compute comparison.', '',
        '| Model | CPU worker draw critical path (s) | GPU sampler (s) | CPU/GPU time | GPU/CPU ESS rate |',
        '|---|---:|---:|---:|---:|']
    for row in worker_groups['gpu_sampler_phase_comparisons']:
        lines.append('| ' + ' | '.join([row['model']] + [number(row[key]) for key in
            ['cpu_worker_draw_critical_path_seconds_median', 'gpu_sampler_phase_seconds_median',
             'ratio_cpu_worker_draw_over_gpu_sampler_phase', 'median_paired_gpu_over_cpu_ess_rate']]) + ' |')
    lines += ['', '[Per-seed worker timing, byte-parity and scope](WORKER_MATH_BENCHMARKS.json) · '
        '[CSV](WORKER_MATH_BENCHMARKS.csv). Selected budgets and transition counts remain separately reported. '
        'These original cross-model rows use original-model LP reaction masks; independent '
        'cached-target follow-ups and faster CPU budgets are reported separately below.', '']
    for study in worker_groups.get('budget_cap_studies', []):
        cap = study.get('cpu_thin2000_fixed_cap', {})
        if study.get('model') == 'iML1515' and cap.get('tested_count') == 3:
            rhats = [r['split_rhat_median'] for r in cap['rows']]
            lines += [f"iML1515's guideline-thinning CPU and float64 Naja groups both pass 3/3 seeds. "
                f"The separately tested CPU thinning-2,000 configuration passes {cap['passing_count']}/3 "
                f"at its {cap['draws']:,}-draw cap; median Rhats range from {min(rhats):.7g} "
                f"to {max(rhats):.7g}. A larger budget could pass, so the 8.51× comparison above "
                'is against the tested guideline configuration, not a globally optimized CPU baseline.', '']

crossmodel_cached = read(R / 'crossmodel_extension/cached_assay_summary.json')
if crossmodel_cached and crossmodel_cached.get('comparisons'):
    lines += ['### Updated cross-model conservative-assay comparisons', '',
        'New CPU budget extensions and worker timers, with cached-target reaction masks. '
        'Both groups pass all three seeds in each reported comparison. These fixed configurations '
        'are distinct from the guideline and original-model-mask rows above; common-mask budget '
        'refinement remains a separate experiment until its fresh retimes complete.', '',
        '| Model | CPU configuration | CPU compute median (s) | GPU sampler median (s) | CPU/GPU time | Paired GPU/CPU ESS rate |',
        '|---|---|---:|---:|---:|---:|']
    for row in crossmodel_cached['comparisons']:
        lines.append('| ' + ' | '.join([row['model'], row['cpu_method']] + [number(row[k]) for k in
            ('cpu_compute_median_seconds', 'gpu_compute_median_seconds', 'cpu_over_gpu_time_ratio',
             'paired_gpu_over_cpu_ess_rate')]) + ' |')
    lines += ['', '[Updated model table](crossmodel_extension/CACHED_TARGET_RESULTS.md) · '
        '[All groups, masks and source files](crossmodel_extension/cached_assay_summary.json).', '']

phase = read(R / 'gpu_thinning/iJO1366/sampling_phase_comparison.json')
if phase:
    lines += ['## Secondary inclusive-phase context and transition budgets', '',
        '16 chains; three seeds. Both methods pass median split-Rhat ≤1.01 and median bulk ESS ≥100. '
        'Naja retains 2,000 draws at thinning 1,000; native hopsy retains 10,000 at thinning 2,000. '
        'Both discard 1,000 retained warmup draws per chain. CPU uses 16 physical cores. '
        'The CPU hopsy.sample phase includes worker startup, serialization and IPC; '
        'the Naja sampler phase includes buffer allocation, initial slack and RNG initialization. '
        'Both exclude output download/file writing; these are phase proxies, not isolated CPU/GPU math counters.', '',
        '| Metric | CPU | Naja | Advantage |', '|---|---:|---:|---:|',
        f"| Median sampling phase (s) | {number(phase['cpu_sampling_phase_median'])} | {number(phase.get('gpu_sampler_phase_median', phase.get('gpu_kernel_median')))} | {number(phase['ratio_of_sampling_phase_medians'])}× time ratio |",
        f"| Median pooled reaction ESS / sampling second | {number(phase['cpu_ess_rate_median'])} | {number(phase['gpu_ess_rate_median'])} | {number(phase['median_paired_ess_rate_ratio'])}× median paired ratio |",
        '| Transitions including warmup | 352,000,000 | 48,000,000 | 7.333× fewer for the target |', '',
        '[Per-seed phase measurements and scope](gpu_thinning/iJO1366/sampling_phase_comparison.json). '
        'Transition counts come from chains × (retained + warmup) × thinning; '
        'they describe these tested budgets, not an optimality claim. The worker-timed table above excludes CPU pool overhead.', '']
    worker = read(B / 'audit/hopsy-worker-timing-verification/iJO1366-real-parity.json')
    if worker and worker['status'] == 'passed':
        lines += [f"The new seed-101 CPU instrumentation reproduces the baseline retained array bitwise. "
            f"Its worker draw-call critical path is **{worker['worker_draw_critical_path_seconds']:.5g} s**, "
            f"versus {worker['hopsy_sample_phase_inclusive_seconds']:.5g} s for the inclusive hopsy.sample phase. "
            'Draw-call timing excludes startup, IPC and output-array assembly, retaining binding/state-copy '
            'cost and descheduling. This validates the timer on one seed; it is not a new three-seed '
            'CPU/GPU speedup group. '
            '[Byte-parity and timing proof](../audit/hopsy-worker-timing-verification/iJO1366-real-parity.json).', '']

fresh = read(R / 'fresh16_validation/iJO1366/summary.json')
if fresh:
    lines += ['## Fresh-seed validation of the earlier 16-chain result', '',
        'New seeds 101001, 202001, 303001 test the selected fixed configurations without retuning. '
        'CPU retains 10,000 draws at thinning 2,000; Naja retains 2,000 at thinning 1,000 and TPB512. '
        'The target remains median split-Rhat ≤1.01, bulk ESS ≥100 and zero bound fraction at 1e-7.', '',
        '| Method | Fixed-budget seeds tested | Fixed-budget passes | Fixed-budget failures | Separately passing extensions |',
        '|---|---:|---:|---|---|']
    for method, group in fresh['fixed_groups'].items():
        extensions = [f"{r['seed']}: {r['extension']['draws_per_chain']} draws" for r in group.get('rows', [])
                      if r.get('extension', {}).get('quality_met')]
        lines.append('| ' + ' | '.join([method, str(len(group['tested_fixed_seeds'])),
            str(len(group['passing_fixed_seeds'])), ', '.join(map(str, group['failed_fixed_seeds'])) or '—',
            '; '.join(extensions) or '—']) + ' |')
    lines += ['', 'The original 2.73× operational result used the earlier selection seeds. '
        'Fresh fixed-budget failures prevent claiming that result generalizes across these seeds; '
        'passing extensions have distinct measured timings and retain the original failures. '
        '[Full validation and phase timings](fresh16_validation/iJO1366/summary.json).', '']

timeline = read(R / 'nsight_timeline/timeline_summary.json')
if timeline:
    lines += ['## H100 timeline and sampler scaling', '',
        '[SVG](nsight_timeline/naja_h100_utilization_timeline.svg) · '
        '[PDF](nsight_timeline/naja_h100_utilization_timeline.pdf) · '
        '[Raw 10 kHz CSV](nsight_timeline/timeline_source.csv) · '
        '[Phase CSV](nsight_timeline/phases.csv) · [Exact commands](nsight_timeline/commands.sh) · '
        '[Methods and counter definitions](nsight_timeline/README.md).', '',
        '![H100 utilization over actual elapsed seconds](nsight_timeline/naja_h100_utilization_timeline.png)', '',
        'iJO1366, seed 101, rotate, automatic thinning 96, TPB128; '
        '1,000 warmup + 10,000 retained draws per chain. Verified wide-index builds. '
        'Nsight Systems GH100 hardware metrics at 10 kHz; plotted smoothing window 5 ms. '
        '**SM active is not arithmetic throughput or achieved warp occupancy.** '
        'These are throughput/profiling runs, not convergence results.', '',
        '| Run | CHRR kernel (s) | Retained draws / kernel second | Mean SM active during CHRR | Profiler overhead |',
        '|---|---:|---:|---:|---:|']
    for row in timeline:
        chains = 16 if row['run'] == 'chains16' else 512
        lines.append(f"| {row['run']} | {number(row['sample_kernel_s'])} | "
            f"{number(chains*10000/row['sample_kernel_s'])} | "
            f"{row['sm_active_sample_mean_pct']:.2f}% | {row['profiled_target_overhead_pct']:.2f}% |")
    base, bulk = timeline[:2]
    ratio = 32 * base['sample_kernel_s'] / bulk['sample_kernel_s']
    lines += ['', f'512 chains deliver **{ratio:.3g}× retained-draw kernel throughput** versus 16 chains '
        'on this one H100. This is GPU scaling, not a CPU speedup or equal-quality comparison. '
        'The 512-chain/prefault native outputs are bitwise identical. '
        'Operational profiled walls are 6.273 / 29.153 / 19.462 s respectively; '
        'those include transfer and output and are secondary to the sampling metric.', '',
        '[Nsight Compute kernel counters](nsight_timeline/ncu/kernel_summary.csv) · '
        '[Roofline FLOP/byte inputs](nsight_timeline/ncu/roofline_source.csv). '
        'The general GH100 timeline set has no L2 bandwidth counter; Nsight Compute supplies it separately.', '']
    counters = R / 'nsight_timeline/ncu/kernel_summary.csv'
    if counters.exists():
        with counters.open() as f:
            samples = [r for r in csv.DictReader(f) if r['case'].startswith('sample_chains')]
        lines += ['| Chains | SM throughput (% peak) | Achieved occupancy (%) | L2 bytes (% peak) | DRAM bytes (% peak) |',
                  '|---:|---:|---:|---:|---:|']
        for row in sorted(samples, key=lambda r: int(r['case'].replace('sample_chains', ''))):
            lines.append('| ' + ' | '.join([row['case'].replace('sample_chains', '')] +
                [number(float(row[k])) for k in ['sm_throughput_peak_pct',
                 'achieved_occupancy_pct', 'l2_bytes_peak_pct', 'dram_bytes_peak_pct']]) + ' |')
        lines += ['', 'Short stationary-kernel Nsight Compute probes. Raw L2 hit-rate ratios '
                  'can exceed 100% across replay passes; those values remain flagged in the source. '
                  'A separate three-pass hit/(hit+miss) estimate gives 99.837%, 99.990%, 99.975% '
                  'at 16/128/512 chains ([raw counts and definitions](nsight_timeline/ncu/l2_hit_rate_validation.csv)). '
                  'These are replayed count estimates, not simultaneous single-pass counters. '
                  'Top warp stalls and FLOP/byte roofline inputs are saved separately.', '']

barrier = read(R / 'barrier_profile/summary.json')
if barrier:
    lines += ['## Barrier preparation: host rotation versus cuBLAS', '',
        'The original phase contains CPU Eigen A×Q, costing approximately 2md² FLOPs. '
        'An isolated build switches only that multiplication to cuBLAS, returning the result to the '
        'existing host pipeline. Both modes include transfers and allocations; mathematical '
        'eigenbasis construction is unchanged. NVTX ranges separate host stages from GPU kernels. '
        '[Full stage data, commands and trace references](barrier_profile/summary.json) · '
        '[Patch](barrier_profile/barrier.patch) · '
        '[CPU-path byte-parity and five analytic tests](barrier_profile/verification/verification.json).', '',
        '| Model | m × d | CPU A×Q (s) | GPU A×Q incl. copies (s) | Original barrier (s) | Offloaded barrier (s) | Full-phase speedup |',
        '|---|---:|---:|---:|---:|---:|---:|']
    barrier_rows = list(barrier.get('comparisons', []))
    # Publish a completed unprofiled pair while the expensive full reference
    # validation/CPU trace finishes; retain its explicitly preliminary scope.
    if not any(r['model'] == 'Recon3D_nullspace_only' for r in barrier_rows):
        cpu = read(R / 'barrier_profile/Recon3D_nullspace_only/cpu/unprofiled/result.json')
        gpu = read(R / 'barrier_profile/Recon3D_nullspace_only/gpu/unprofiled/result.json')
        if cpu and gpu and all(r['feasibility']['declared_bounds_passed'] for r in [cpu, gpu]):
            cpu_stages = {r['name']:r['seconds'] for r in cpu['stages']}
            gpu_stages = {r['name']:r['seconds'] for r in gpu['stages']}
            a, b = [r['profile']['timing']['barrier_rotation_s'] for r in [cpu, gpu]]
            barrier_rows.append(dict(model='Recon3D_nullspace_only (first pair)', m=18477, d=4861,
                cpu_rotation_seconds=cpu_stages['barrier.cpu_rotate_A'],
                gpu_rotation_including_copies_seconds=gpu_stages['barrier.gpu_rotate_A'],
                cpu_barrier_seconds=a, gpu_barrier_seconds=b, barrier_speedup=a/b))
    for row in barrier_rows:
        lines.append('| ' + ' | '.join([row['model'], f"{row['m']} × {row['d']}"] +
            [number(row[key]) for key in ['cpu_rotation_seconds', 'gpu_rotation_including_copies_seconds',
             'cpu_barrier_seconds', 'gpu_barrier_seconds', 'barrier_speedup']]) + ' |')
    lines += ['', 'Times are unprofiled same-binary observations; profiled runs provide attribution and '
        'overhead checks. Algebra-validation runs are excluded from performance. Short sampling '
        'checks establish feasibility, not convergence. Recon3D uses the separately labeled '
        'equality-only/no-MVE geometry; its one-time equality elimination and model preprocessing '
        'remain separate costs. RECON1 retains its declared 1e-6 geometry tolerance; other targets use 1e-7. '
        'A row marked first pair precedes completion of the full large-matrix reference check and CPU trace.', '']
    for run in barrier.get('runs', []):
        if run.get('role') == 'validation' and run.get('algebra_validation'):
            check = run['algebra_validation'][0]
            lines += [f"{run['model']} full CPU/GPU A×Q matrix comparison passed: relative Frobenius "
                f"error {check['relative_frobenius']:.4g}, maximum absolute error "
                f"{check['maximum_absolute']:.4g}. This validation is separate from timing.", '']

dimension = read(R / 'barrier_dimension_scaling/summary.json')
if dimension and (R / 'barrier_dimension_scaling/medians.csv').exists():
    lines += ['### Nine-model barrier dimension scaling: three timing repeats', '',
        'Same frozen binary, one H100, independent process repeats at fixed seed101. '
        'Short feasibility checks pass their declared tolerances; these are not convergence runs. '
        'Recon3D remains explicitly equality-only/no-MVE exploratory geometry.', '',
        '| Model | d | Original barrier median (s) | GPU-rotation median (s) | Phase speedup |',
        '|---|---:|---:|---:|---:|']
    for row in csv.DictReader((R / 'barrier_dimension_scaling/medians.csv').open()):
        lines.append('| ' + ' | '.join([row['model'], row['d']] + [number(float(row[k])) for k in
            ('cpu_barrier_median', 'gpu_barrier_median', 'phase_speedup')]) + ' |')
    lines += ['', '![Barrier phase costs across dimensions](barrier_dimension_scaling/barrier_dimension_scaling.png)', '',
        'The historical PolyRound MVE points and the Naja barrier points measure different '
        'operations on different preparation stages. Their separation is cost context, '
        'not evidence of equivalent conditioning quality. Hessian/eigensolve stage walls '
        'include initialization and host waits, and are not pure GPU-kernel times.', '',
        '[SVG](barrier_dimension_scaling/barrier_dimension_scaling.svg) · '
        '[PDF](barrier_dimension_scaling/barrier_dimension_scaling.pdf) · '
        '[All timing-repeat CSV](barrier_dimension_scaling/source.csv) · '
        '[Methods and exact commands](barrier_dimension_scaling/README.md).', '']

bulk = read(R / 'high_throughput/iJO1366/summary.json')
if bulk:
    lines += ['## Matched-quality 512-chain experiment', '',
        '[All attempts, validation and selected candidates](high_throughput/iJO1366/summary.json). '
        'Fresh, nonoverlapping replicate seed ranges. CPU variants use 48 or 64 distinct physical cores, '
        'explicitly separated. Completed full-node comparisons are audited separately below; '
        'partial configurations support no three-seed headline.', '',
        '| Candidate | Validated passing seeds | Status |', '|---|---:|---|']
    for name, group in bulk.get('groups', {}).items():
        lines.append(f"| {name} | {len(group.get('rows', []))}/3 | {group['status']} |")
    lines += ['', 'Sampling-phase and worker draw-call measurements lead this comparison. '
        'Process-wall and transfer effects remain separately recorded in the source.', '']

bulk_audit = read(R / 'high_throughput/iJO1366/comparison.json')
if bulk_audit:
    lines += ['### Full 64-core CPU versus one H100: independently audited comparison', '',
        '512 chains each, three fresh seeds. CPU: 12,000 retained draws, thinning 2,000, '
        '64 distinct physical cores with 128 logical CPUs reserved. Naja: 5,000 retained '
        'draws, thinning 500, TPB128, host prefault8. Both discard 1,000 warmup draws per chain. '
        'CPU uses worker draw-call critical paths; GPU uses its sampler phase. Output is excluded.', '',
        f"Audit status: `{bulk_audit['status']}`. The original-model assay excludes 878 reactions; "
        'the conservative cached-target assay excludes only 478 verified-nearconstant reactions. '
        'They use identical saved samples and the same frozen estimator, with separate metric files.', '',
        '| Diagnostic assay | CPU/GPU sampler-time ratio | Median paired GPU/CPU ESS-rate ratio | Both methods pass all three seeds |',
        '|---|---:|---:|---|']
    for assay, pair in bulk_audit.get('paired', {}).items():
        lines.append(f"| {assay} | {number(pair.get('compute_time_ratio_of_medians'))} | "
            f"{number(pair.get('paired_effective_sampling_rate_ratio_median'))} | {pair.get('all_six_runs_pass')} |")
    lines += ['', 'These budgets deliver roughly 77–81k pooled reaction ESS under the original assay. '
        'This is a fixed-512-chain effective-sampling comparison, rather than a globally minimal '
        'time-to-quality claim across all chain counts. Naja uses 8.667× fewer transitions '
        'including warmup; the gain combines sampling efficiency and hardware throughput. '
        '[Audited table and interpretation](high_throughput/iJO1366/AUDITED_COMPARISON.md) · '
        '[CSV](high_throughput/iJO1366/comparison.csv) · [Proofs and source hashes](high_throughput/iJO1366/comparison.json).', '']

larger = read(R / 'larger_quality/summary.json')
if larger:
    lines += ['## Additional model quality coverage', '',
        '[Per-seed records and operational times](larger_quality/table.csv) · '
        '[Validated groups](larger_quality/summary.json). '
        '16 chains, median split-Rhat ≤1.01 and bulk ESS ≥100; bounds checked independently.', '',
        '| Model | Method | Passing seeds | Max sample bound fraction at 1e-7 |', '|---|---|---:|---:|']
    for group in larger['groups']:
        rows = group.get('rows', [])
        bound = max((r['sample_bound_violation_fraction'] for r in rows), default=None)
        lines.append(f"| {group['model']} | {group['method']} | {len(rows)}/3 | {number(bound)} |")
    lines += ['', 'Passing convergence diagnostics do not imply zero bound violations. '
        'iMM904 residuals around 2e-7 and the iML1515 seed-specific residual remain visible; '
        'no samples are silently filtered. Output-excluded cross-model timing summaries are being assembled.', '']

precision = [(path, read(path)) for path in sorted((R / 'precision_quality').glob('*/summary.json'))]
if precision:
    lines += ['## Float64 export follow-ups', '',
        'Separate verified float64 builds repeat the same sampling protocols to isolate export precision. '
        'A corrected group requires the shared quality target and zero original-bound violations at 1e-7; '
        'the paired cast-back comparison tests whether the earlier float32 trajectory is reproduced. '
        'Canonical float32 records are preserved.', '',
        '| Model | Seeds passing quality and strict bounds | Three-seed group | Median sampler phase (s) | Evidence |',
        '|---|---:|---|---:|---|']
    for path, group in precision:
        if not group:
            continue
        passed = [r for r in group.get('rows', []) if r.get('quality_met') and r.get('strict_bounds_met')]
        lines.append(f"| {group['model']} | {len(passed)}/3 | {group.get('complete_three_seed_group', False)} | "
            f"{number(group.get('median_sampling_phase_seconds'))} | [per-seed checks]({path.relative_to(R)}) |")
    lines += ['', 'The sampler phase excludes download/writes; export precision changes are not '
        'automatically a sampler-compute speedup. Partial attempts remain visible in the source.', '']

proof = read(R / 'coordinate_kernel/verification.json')
lines += ['## Isolated coordinate-kernel specialization', '',
    'The experimental build removes unused mixture paths only when all noncoordinate probabilities are zero. '
    'Compiler register usage at TPB128 falls from 104 to 64 per thread, with no spills; '
    'the generic mixture path is retained. This changes the register residency ceiling from four to eight blocks/SM. '
    'Compiler resource savings alone are not a measured speedup.', '',
    f"GPU analytic/trajectory validation: **{proof['status'] if proof else 'pending'}**. "
    '[Patch](coordinate_kernel/kernel.patch) · [Compiler resource records](coordinate_kernel/compiler_resources.json) · '
    '[Complete source snapshot](coordinate_kernel/source.tar.gz) · [Validation](coordinate_kernel/verification.json). '
    'Canonical source/builds/results are unchanged by this isolated experiment.', '']
paired = read(R / 'coordinate_kernel/paired/summary.json')
if paired:
    lines += ['[Paired timings and commands](coordinate_kernel/paired/summary.json). '
        'Single-seed, same-GPU comparison; all probabilities for noncoordinate moves are zero. '
        'Both builds use host prefault; timing below excludes download, writing and normalization.', '',
        '| Chains | Original sampler (s) | Specialized sampler (s) | Sampling speedup | Full retained arrays identical |',
        '|---:|---:|---:|---:|---|']
    for pair in paired['pairs']:
        rows = [r for r in paired['rows'] if r['chains'] == pair['chains']]
        times = {r['build']:r['sampler_phase_seconds'] for r in rows}
        lines.append(f"| {pair['chains']} | {number(times['reference'])} | {number(times['coordinate'])} | "
                     f"{number(pair['sampling_phase_speedup'])}× | {pair['arrays_bitwise_equal']} |")
    lines += ['', 'This is a paired implementation/throughput experiment. '
        'It does not itself establish a new CPU speedup or convergence result.', '']
for size, folder in [(512, 'clock_audit'), (1024, 'clock_audit1024')]:
    audit = read(R / f'coordinate_kernel/{folder}/summary.json')
    if not audit:
        continue
    ratios = [r['sampling_speedup'] for r in audit['pairs']]
    lines += [f'### Same-GPU specialization replication: {size} chains', '',
        'Fresh paired runs balance build order across seeds on one allocation, preserve complete '
        'retained-array identity, and log GPU identity/clocks/power as diagnostics. '
        'Those clock logs are separate from the Nsight utilization figure.', '',
        '| Seed | Original sampler (s) | Specialized sampler (s) | Paired ratio |',
        '|---:|---:|---:|---:|']
    for pair in audit['pairs']:
        values = {r['build']:r['sampler_phase_seconds'] for r in audit['rows'] if r['seed']==pair['seed']}
        lines.append(f"| {pair['seed']} | {number(values['reference'])} | {number(values['coordinate'])} | "
                     f"{number(pair['sampling_speedup'])}× |")
    lines += ['', f"Median paired ratio: **{statistics.median(ratios):.4g}×**. " +
        ('The earlier single-seed 512-chain improvement did not replicate here; it is not a general speedup claim. '
         if size == 512 else '') +
        f'[Raw pairs, commands and clock diagnostics](coordinate_kernel/{folder}/summary.json).', '']

recon = read(R / 'exploratory_recon3d_barrier/seed101/probe_summary.json')
if recon:
    lines += ['## Recon3D exploratory barrier-only probe', '',
        f"Successfully sampled an equality-eliminated Recon3D target: n={recon['n']}, d={recon['d']}, m={recon['m']}. "
        'This separate geometry skips PolyRound MVE rounding, uses GPU whitening and float64 export, '
        'and preserves the original model target. It does not replace the still-running canonical preprocessing.', '',
        f"Barrier preparation {recon['profile']['timing']['barrier_rotation_s']:.4g} s; "
        f"sampler phase {recon['profile']['timing']['gpu_sampling_s']:.4g} s. "
        f"16 chains, 100 discarded warmup + 200 retained draws per chain. "
        f"All {recon['checked_samples']} deterministically selected samples passed original bounds at 1e-7. "
        '**Feasibility/throughput probe only: convergence was not established.** '
        '[Evidence](exploratory_recon3d_barrier/seed101/probe_summary.json) · '
        '[Jobs and reproduction](exploratory_recon3d_barrier/jobs.json).', '']

lines += ['## Conditioned-rounding prototype', '',
    'A separate iJO1366 PGI capacity restriction [-0.1, 0.1] compares full rerounding, '
    'stale shared rounding, and shared rounding plus Naja correction. '
    'This is explicitly a capacity restriction, not an exact knockout. '
    '[Live phase/quality table](conditioned_prototype/TABLE.md) · '
    '[Protocol and exact-target audit](../audit/conditioned_reuse_assessment.md). '
    'Setup/correction and sampling efficiency are primary; total process times are secondary.', '']
prep = read(R / 'conditioned_prototype/iJO1366/preparation.json')
dimension_audit = read(R / 'conditioned_prototype/dimensionality_warning.json')
if dimension_audit:
    proof = read(R / 'conditioned_prototype/iJO1366/models/global_reround/global_reround_proof.json')
    lines += ['The independent raw-model reround produced 582 dimensions versus 581 in the '
        'inherited geometry. Those original full-reround results are **incompatible as a same-target '
        'reference**, despite passing sample feasibility checks. Earlier raw-reround speedup figures '
        'are suspended. [Dimension audit](conditioned_prototype/dimensionality_warning.json).', '']
    if proof and proof.get('status') == 'passed':
        lines += [f"The repaired reference globally rerounds the exact cached {proof['dimension']}-D "
            f"system without repeating equality elimination. All {proof['rows']} transformed inequalities "
            f"match the shared target (relative A error {proof['transformed_A_relative_identity_error']:.3g}; "
            f"relative b error {proof['transformed_b_relative_identity_error']:.3g}); the map is invertible "
            f"and the conditional diagnostic mask matches. Global rounding itself took "
            f"{proof['global_round_algorithm_seconds']:.5g} s. Sampling quality and fresh selected-budget "
            'timings for this repaired reference are being measured. '
            '[Algebraic target-equivalence proof](conditioned_prototype/iJO1366/models/global_reround/global_reround_proof.json).', '']
if prep and prep.get('status') == 'complete' and not dimension_audit:
    lines += [f"Both conditional geometries passed validation at 1e-7. Full PolyRound rerounding took "
        f"**{prep['full_polyround_seconds']:.5g} s**; shared geometry assembly plus strict-interior LP took "
        f"**{prep['shared_assembly_seconds']+prep['shared_interior_lp_seconds']:.5g} s**. "
        'The GPU correction cost and sampling quality are separate, still pending. '
        f"Base preprocessing ({prep['base_preprocessing_seconds']:.5g} s) is a one-time cached setup cost. "
        'These setup measurements alone do not establish an equal-quality sampling advantage. '
        '[Preparation evidence](conditioned_prototype/iJO1366/preparation.json).', '']

conditional_quality = read(R / 'conditioned_prototype/quality_summary.json')
if conditional_quality:
    def conditional_link(value):
        return str(Path(value).relative_to(R))
    lines += ['### Same-target convergence curves and fresh time-to-target', '',
        'PGI capacity prototype, median split-Rhat ≤1.05 and bulk ESS ≥100, with original/conditional '
        'bounds and mass balance ≤1e-7. The repaired global reference and inherited methods use the '
        'same cached target and conservative common LP mask. Earlier 878-mask curves are historical. '
        'Only separately executed selected budgets populate times; prefix points have no inferred times.', '',
        f"[Rhat SVG]({conditional_link(conditional_quality['curve_svg'])}) · "
        f"[PDF]({conditional_link(conditional_quality['curve_pdf'])}) · "
        f"[Source curve CSV]({conditional_link(conditional_quality['curve_csv'])}) · "
        f"[Fresh time-to-target CSV]({conditional_link(conditional_quality['time_to_target_csv'])}).", '']
    preview = conditional_quality.get('curve_png')
    if preview and Path(preview).exists():
        lines += [f'![Conditioned-target convergence curves]({conditional_link(preview)})', '']
    if conditional_quality.get('transition_curve_svg'):
        lines += [f"[Actual-transition-axis figure]({conditional_link(conditional_quality['transition_curve_svg'])}); "
            'this includes each method’s thinning and retained warmup count.', '']
    lines += ['| Pipeline | Seed | Status | Selected retained draws | Sampler compute proxy (s) | Rhat | Bulk ESS |',
              '|---|---:|---|---:|---:|---:|---:|']
    for row in conditional_quality.get('rows', []):
        lines.append('| ' + ' | '.join([row['variant'], str(row['seed']), row['status'],
            number(row.get('selected_draws')), number(row.get('measured_sampler_seconds')),
            number(row.get('split_rhat_median')), number(row.get('bulk_ess_median'))]) + ' |')
    lines += ['', 'CPU compute is the accumulated draw-call critical-path proxy; Naja uses its sampler '
        'phase including allocation/slack/RNG initialization. Setup/correction costs are separate. '
        'Across the three measured passing seeds, sampler times are essentially tied; the reuse benefit '
        'is avoiding the 61.2 s global rerounding step. Real zero-flux knockout results follow separately. '
        '[All statuses, source runs and target proof](conditioned_prototype/quality_summary.json).', '']
    stale_endpoints = []
    for row in conditional_quality.get('rows', []):
        if row.get('variant') == 'stale_cpu' and row.get('status') == 'no_quality_within_cap':
            curve = read(Path(row['prefix_curve_json']))
            if curve:
                stale_endpoints.append(max(curve, key=lambda point: point['draws_per_chain']))
    if stale_endpoints:
        lines += [f"Stale inherited rounding misses the target in {len(stale_endpoints)}/3 seeds at "
            f"{min(p['draws_per_chain'] for p in stale_endpoints):,} retained draws per chain: "
            f"median split-Rhat across seeds ranges from {min(p['split_rhat_median'] for p in stale_endpoints):.4f} "
            f"to {max(p['split_rhat_median'] for p in stale_endpoints):.4f}, and median ESS from "
            f"{min(p['bulk_ess_median'] for p in stale_endpoints):.1f} to "
            f"{max(p['bulk_ess_median'] for p in stale_endpoints):.1f}. "
            'This is the finite 50k-draw cap, not a claim of permanent nonconvergence.', '']

ko_catalog = read(R / 'knockout_panel/gene_catalog.json')
ko_ready = read(R / 'knockout_panel/ready_manifest.json')
if ko_catalog and ko_ready:
    nontrivial_ready = [r for r in ko_ready['conditions'] if r.get('changed_bound_rows', 0) > 0]
    skipped = [r for r in nontrivial_ready if r.get('LP_calls') == 0]
    lines += ['## Real gene-knockout panel: geometry and starting points', '',
        f"The catalog contains {ko_catalog['real_gene_count']} real genes and "
        f"{ko_catalog['unique_bound_change_signatures']} distinct bound-change signatures. "
        f"{ko_ready['unique_ready_count']} unique targets currently have validated sampling geometry "
        f"({ko_ready['nontrivial_unique_ready_count']} nontrivial plus a control). " +
        ('Geometry preparation is complete for the panel; the detached sampling consumer is continuing. '
         if ko_ready.get('status') == 'complete' else 'The geometry producer is continuing toward approximately 50. ') +
        'Geometry readiness does not imply sampling convergence.', '',
        'The final validated panel has 41 nontrivial targets with 1–5 changed rows and '
        '9 with 6–20, plus one control. It is not fully stratified: 21–50-row candidates '
        'exist biologically but attempted faces failed the geometry certification gates; '
        'the catalog has no 51–200-row gene knockouts. '
        '[Independent final geometry audit](knockout_panel/final_geometry_audit.json).', '',
        f"The unchanged base point is weakly feasible at 1e-7 for "
        f"{100*ko_catalog['all_real_gene_base_point_weak_feasible_fraction']:.2f}% of all genes, but only "
        f"{100*ko_catalog['nontrivial_gene_base_point_weak_feasible_fraction']:.2f}% of the "
        f"{ko_catalog['nontrivial_varying_gene_count']} genes classified as affecting LP-varying reactions. "
        'The larger population includes many unchanged/redundant targets and is not a nontrivial skip rate.', '',
        f"Analytic projection avoided all interior LP calls for {len(skipped)}/{len(nontrivial_ready)} "
        'currently ready nontrivial targets. This is a validity-selected subset of the gene catalog. '
        'PGI/b4025 is a genuine zero-flux knockout with a validated 580-D target and zero interior LP calls. '
        'Unstable affine-rank/dual certificates remain unsupported, with rejection evidence retained; '
        'no bound relaxation is used.', '',
        '[Full catalog and denominator definitions](knockout_panel/gene_catalog.json) · '
        '[Ready target manifests](knockout_panel/ready_manifest.json) · '
        '[All geometry attempts](knockout_panel/geometry_summary.json).', '']
    analytic = ko_catalog.get('cached_analytic_variation_certificate')
    if analytic:
        lines += [f"A stronger cached-target certificate uses the feasible ball of radius "
            f"{analytic['base_strict_ball_radius']:.6g}: reaction variation is at least "
            '2 × radius × the norm of its affine-map row. '
            f"Among {analytic['certified_nontrivial_real_genes']} genes affecting a reaction with "
            f"certified variation above 1e-7, **{analytic['weak_feasible_genes_in_that_group']}** "
            'can keep the unchanged base point at the knockout bounds. Analytic projection, rather '
            'than unchanged-start reuse, is therefore the useful LP-skipping route in the validated examples.', '']

panel_quality = read(R / 'knockout_panel/quality_summary.json')
if panel_quality:
    from collections import Counter
    panel_counts = {}
    for condition in panel_quality.get('conditions', []):
        for row in condition.get('summary', {}).get('rows', []):
            variant = row.get('variant', 'unknown')
            panel_counts.setdefault(variant, Counter())[row.get('status', 'unknown')] += 1
    lines += ['### Live panel sampling progress', '',
        'Counts below describe individual method/seed tasks on validated targets. '
        'Geometry-ready targets without a sampling task are still pending; mechanical failures '
        'and bounded retries remain in their source records.', '',
        '| Method | Completed | Recorded unfinished | Other recorded states |',
        '|---|---:|---:|---|']
    for variant, counts in sorted(panel_counts.items()):
        other = ', '.join(f'{k}: {v}' for k, v in sorted(counts.items())
                          if k not in ('complete', 'running')) or '—'
        lines.append(f"| {variant} | {counts['complete']} | {counts['running']} | {other} |")
    lines += ['', '[Live per-target curves, measured timings and statuses](knockout_panel/quality_summary.json).', '']
    matched_panel = []
    for condition in panel_quality.get('conditions', []):
        rows = condition.get('summary', {}).get('rows', [])
        complete = {row.get('variant'): row for row in rows if row.get('status') == 'complete'}
        if all(v in complete for v in ('global_cpu', 'stale_cpu', 'naja_rotate')):
            condition_id = complete['naja_rotate']['condition_id']
            if condition_id != 'b0002':
                matched_panel.append((condition_id, complete))
    if matched_panel:
        lines += ['### Completed three-pipeline exact-knockout comparisons', '',
            'One seed per target; each entry is retained draws per chain / measured sampler proxy seconds. '
            'All entries pass median Rhat ≤1.05, ESS ≥100 and unmasked feasibility checks. '
            'Setup is separate; the unchanged control is excluded.', '',
            '| Gene KO | Full reround + CPU | Inherited + CPU | Shared + Naja correction |',
            '|---|---:|---:|---:|']
        for condition_id, complete in matched_panel:
            values = [f"{complete[v]['selected_draws']:,} / {complete[v]['measured_sampler_seconds']:.3f}"
                      for v in ('global_cpu', 'stale_cpu', 'naja_rotate')]
            lines.append('| ' + ' | '.join([condition_id, *values]) + ' |')
        lines += ['', 'Inherited rounding here includes the affine-hull projection required for exact '
            'zero-flux knockouts. That projection can improve mixing: stale inheritance passes on '
            'the first exact PGI knockout, unlike the finite-width capacity restriction above. '
            'These targets therefore do not establish a universal stale-rounding failure.', '']

panel_audit = read(R / 'knockout_panel/overnight_audit/summary.json')
if panel_audit:
    lines += ['### Completed overnight panel audit', '',
        '50 nontrivial exact knockouts; the unchanged control is excluded. Independent '
        'cached-target containment, full-reround affine equivalence and common-mask checks pass. '
        'One seed per target; these are fixed finite-cap results.', '',
        '| Pipeline | Pass / 50 | Capped at 50k | Unavailable |', '|---|---:|---:|---:|']
    for group in panel_audit['groups']:
        lines.append(f"| {group['variant']} | {group['passed']} | {group['capped']} | {group['unavailable']} |")
    pairs = [p for p in panel_audit['paired_comparisons'] if p['reference'] == 'global_cpu']
    if pairs:
        incremental = statistics.median(p['cpu_over_gpu_incremental_preparation_plus_sampler_time'] for p in pairs)
        common = statistics.median(p['cpu_over_gpu_time_with_common_conditional_geometry'] for p in pairs)
        sampler = statistics.median(p['cpu_over_gpu_sampler_time'] for p in pairs)
        lines += ['', f"On the **{len(pairs)}/50** targets passing both full-global CPU and Naja, "
            f"the median paired ratio for **rerounding/correction plus sampling** is **{incremental:.2f}×** "
            f"in Naja's favor. Including the common conditional-geometry preparation gives **{common:.2f}×**. "
            f"Sampler time alone has median ratio **{sampler:.3f}×**. These are separate scopes; "
            'the preparation-reuse benefit must not be called a sampling-kernel speedup. '
            'Ratios exclude output/download/write; common geometry preparation includes file assembly. '
            'The passing subset does not represent success on all 50 targets. Per-case ratios include '
            'a Naja slowdown, and failed/capped methods remain in the denominator.', '']
    fastest = panel_audit.get('fastest_tested_valid_cpu_comparisons', [])
    if fastest:
        advantage = statistics.median(p['cpu_over_gpu_preparation_plus_sampler'] for p in fastest)
        common_advantage = statistics.median(p['cpu_over_gpu_with_common_conditional_preparation'] for p in fastest)
        contingency = panel_audit['rotate_vs_inherited_pass_contingency']
        lines += [f"Allowing the fastest tested passing CPU pipeline per target gives a smaller "
            f"median paired preparation-plus-sampling ratio of **{advantage:.3f}×** "
            f"(**{common_advantage:.3f}×** with shared condition preparation). "
            'This post-hoc CPU choice can avoid MVE where inherited rounding already passes. '
            'The 14× figure above is specifically against full rerounding, not this fastest CPU selector. '
            f"Rotate/inherited pass contingency: {contingency['both_pass']} both, "
            f"{contingency['rotate_only']} rotate only, {contingency['inherited_only']} inherited only, "
            f"{contingency['neither_pass']} neither at the tested cap.", '']
    lines += ['[Standalone audit and definitions](knockout_panel/overnight_audit/CONDITIONED_BENCHMARK.md) · '
        '[All targets CSV](knockout_panel/overnight_audit/per_condition.csv) · '
        '[Paired timings CSV](knockout_panel/overnight_audit/paired_comparisons.csv) · '
        '[Independently recomputed proofs](knockout_panel/overnight_audit/summary.json).', '']
    if (R / 'knockout_panel/overnight_audit/conditioned_panel_summary.svg').exists():
        lines += ['![Finite-cap success and paired preparation-reuse costs](knockout_panel/overnight_audit/conditioned_panel_summary.png)', '',
            '[SVG](knockout_panel/overnight_audit/conditioned_panel_summary.svg) · '
            '[PDF](knockout_panel/overnight_audit/conditioned_panel_summary.pdf) · '
            '[Figure source CSV](knockout_panel/overnight_audit/figure_source.csv) · '
            '[Definitions](knockout_panel/overnight_audit/figure_definitions.json).', '']

lp_reuse = read(R / 'knockout_panel/lp_microbench_retained/summary.json')
if lp_reuse:
    lines += ['## Starting-point LP: persistent solver reuse', '',
        'The PGI-capacity microbenchmark changes the actual neighboring slab bounds while retaining '
        'the same LP matrix. Starts are checked against the full saved inequalities; sampling targets '
        'are unchanged. This tests solver-state reuse, not exact-knockout affine-hull handling.', '',
        '| HiGHS threads | New model, initialized pool (s) | Explicitly reset basis, changed slab (s) | Internal basis/factorization retained, changed slab (s) |',
        '|---:|---:|---:|---:|']
    for case in lp_reuse['cases']:
        if not case['case'].startswith('highs'):
            continue
        def role_median(role):
            values = [r['seconds'] for r in case['repetitions'] if r['role'] == role]
            return statistics.median(values) if values else None
        lines.append('| ' + ' | '.join([str(case['accepted_threads_option'])] + [number(role_median(role)) for role in
            ['new_model_cold_basis_pool_already_initialized', 'cached_basis_changed_capacity_target',
             'cached_internal_basis_changed_capacity_no_reset']]) + ' |')
    lines += ['', 'Retained-state solves take about 13 ms here, versus roughly 1.5 s for new models. '
        'Calling setBasis repeatedly discards useful internal state and costs about 0.24 s. '
        'Imports, model assembly and the first solve are one-time costs saved separately. '
        'Thread-count changes did not explain the latency difference in this test. '
        'The real-knockout panel will separately measure base-start feasibility, analytic projection, '
        'facial reduction and required LP calls. '
        '[Raw calls, accepted options and residual checks](knockout_panel/lp_microbench_retained/summary.json).', '']

lines += ['## Follow-up priorities requested at 04:30 UTC', '',
    '- Conditioned quality: three same-target pipelines, split-Rhat versus retained chain length '
    'and fresh measured time-to-target; then a panel of approximately 50 real gene knockouts.',
    '- Starting-point cost: base/projection feasibility checks, cold versus repeated LP solves, '
    'and explicit HiGHS model/basis reuse. Trivial/redundant and nontrivial knockout denominators remain separate.',
    '- iML1515: add the validated Naja/guideline-CPU time-to-target row and complete the thinning-2,000 '
    'fixed-budget seed audit; unrun seeds are not counted as failures.',
    '- Fresh fixed-budget 16-chain validation and completion of the 512-chain matched-quality groups '
    'use new seed ranges and preserve failed budgets separately.', '']

out = R / 'RESULTS_SINCE_UTILIZATION_REQUEST.md'
tmp = out.with_suffix('.tmp')
tmp.write_text('\n'.join(lines))
tmp.replace(out)
print(out)
