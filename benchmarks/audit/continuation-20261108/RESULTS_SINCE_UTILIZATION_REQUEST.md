# Naja: results since the utilization-plot request

Updated 2026-11-08T11:10:11+00:00.

New profiling, sampling analyses and experiments from this phase of the campaign. [Earlier campaign report](BENCHMARK_TABLE.md) is retained separately. Lead metrics are sampler time to a shared quality target and effective samples per sampling second; load, correction, transfer, writing and total process time remain separate. Pure GPU kernel timing and inclusive CPU sampling-phase proxies are explicitly distinguished.

[Cross-model sampling-phase source table](SAMPLING_BENCHMARKS.md) · [CSV](SAMPLING_BENCHMARKS.csv) · [Full source/provenance JSON](SAMPLING_BENCHMARKS.json). This provisional phase table includes CPU pool overhead; isolated worker draw-call timing is reported separately as it becomes available.

## Sampling efficiency with CPU worker timing

Three seeds per completed group. CPU timing accumulates C++ draw calls per worker and reports the critical path, excluding pool startup, IPC and output-array assembly. Binding/state-copy and descheduling remain included. GPU timing is the native sampler phase, including allocation, initial slack GEMM and RNG initialization, excluding download/writes. These explicit scopes supersede inclusive CPU pool timing for the compute comparison.

| Model | CPU worker draw critical path (s) | GPU sampler (s) | CPU/GPU time | GPU/CPU ESS rate |
|---|---:|---:|---:|---:|
| iAB_RBC_283 | 0.66033 | 0.606 | 1.0897 | 2.4739 |
| iJO1366 | 15.999 | 7.8132 | 2.0477 | 1.9228 |
| iML1515 | 69.055 | 8.1171 | 8.5074 | 6.6573 |

[Per-seed worker timing, byte-parity and scope](WORKER_MATH_BENCHMARKS.json) · [CSV](WORKER_MATH_BENCHMARKS.csv). Selected budgets and transition counts remain separately reported.

iML1515's guideline-thinning CPU and float64 Naja groups both pass 3/3 seeds. The separately tested CPU thinning-2,000 configuration passes 0/3 at its 10,000-draw cap; median Rhats range from 1.012195 to 1.012413. A larger budget could pass, so the 8.51× comparison above is against the tested guideline configuration, not a globally optimized CPU baseline.

## Secondary inclusive-phase context and transition budgets

16 chains; three seeds. Both methods pass median split-Rhat ≤1.01 and median bulk ESS ≥100. Naja retains 2,000 draws at thinning 1,000; native hopsy retains 10,000 at thinning 2,000. Both discard 1,000 retained warmup draws per chain. CPU uses 16 physical cores. The CPU hopsy.sample phase includes worker startup, serialization and IPC; the Naja sampler phase includes buffer allocation, initial slack and RNG initialization. Both exclude output download/file writing; these are phase proxies, not isolated CPU/GPU math counters.

| Metric | CPU | Naja | Advantage |
|---|---:|---:|---:|
| Median sampling phase (s) | 20.583 | 7.8132 | 2.6344× time ratio |
| Median pooled reaction ESS / sampling second | 105.7 | 257.9 | 2.4029× median paired ratio |
| Transitions including warmup | 352,000,000 | 48,000,000 | 7.333× fewer for the target |

[Per-seed phase measurements and scope](gpu_thinning/iJO1366/sampling_phase_comparison.json). Transition counts come from chains × (retained + warmup) × thinning; they describe these tested budgets, not an optimality claim. The worker-timed table above excludes CPU pool overhead.

The new seed-101 CPU instrumentation reproduces the baseline retained array bitwise. Its worker draw-call critical path is **15.999 s**, versus 21.371 s for the inclusive hopsy.sample phase. Draw-call timing excludes startup, IPC and output-array assembly, retaining binding/state-copy cost and descheduling. This validates the timer on one seed; it is not a new three-seed CPU/GPU speedup group. [Byte-parity and timing proof](../audit/hopsy-worker-timing-verification/iJO1366-real-parity.json).

## Fresh-seed validation of the earlier 16-chain result

New seeds 101001, 202001, 303001 test the selected fixed configurations without retuning. CPU retains 10,000 draws at thinning 2,000; Naja retains 2,000 at thinning 1,000 and TPB512. The target remains median split-Rhat ≤1.01, bulk ESS ≥100 and zero bound fraction at 1e-7.

| Method | Fixed-budget seeds tested | Fixed-budget passes | Fixed-budget failures | Separately passing extensions |
|---|---:|---:|---|---|
| cpu | 3 | 2 | 303001 | 303001: 15000 draws |
| gpu | 3 | 1 | 101001, 303001 | 101001: 3000 draws; 303001: 3000 draws |

The original 2.73× operational result used the earlier selection seeds. Fresh fixed-budget failures prevent claiming that result generalizes across these seeds; passing extensions have distinct measured timings and retain the original failures. [Full validation and phase timings](fresh16_validation/iJO1366/summary.json).

## H100 timeline and sampler scaling

[SVG](nsight_timeline/naja_h100_utilization_timeline.svg) · [PDF](nsight_timeline/naja_h100_utilization_timeline.pdf) · [Raw 10 kHz CSV](nsight_timeline/timeline_source.csv) · [Phase CSV](nsight_timeline/phases.csv) · [Exact commands](nsight_timeline/commands.sh) · [Methods and counter definitions](nsight_timeline/README.md).

![H100 utilization over actual elapsed seconds](nsight_timeline/naja_h100_utilization_timeline.png)

iJO1366, seed 101, rotate, automatic thinning 96, TPB128; 1,000 warmup + 10,000 retained draws per chain. Verified wide-index builds. Nsight Systems GH100 hardware metrics at 10 kHz; plotted smoothing window 5 ms. **SM active is not arithmetic throughput or achieved warp occupancy.** These are throughput/profiling runs, not convergence results.

| Run | CHRR kernel (s) | Retained draws / kernel second | Mean SM active during CHRR | Profiler overhead |
|---|---:|---:|---:|---:|
| chains16 | 4.0813 | 39203 | 11.71% | 5.60% |
| chains512 | 4.5612 | 1.1225e+06 | 97.22% | 2.70% |
| chains512_prefault8 | 4.6424 | 1.1029e+06 | 96.24% | 0.90% |

512 chains deliver **28.6× retained-draw kernel throughput** versus 16 chains on this one H100. This is GPU scaling, not a CPU speedup or equal-quality comparison. The 512-chain/prefault native outputs are bitwise identical. Operational profiled walls are 6.273 / 29.153 / 19.462 s respectively; those include transfer and output and are secondary to the sampling metric.

[Nsight Compute kernel counters](nsight_timeline/ncu/kernel_summary.csv) · [Roofline FLOP/byte inputs](nsight_timeline/ncu/roofline_source.csv). The general GH100 timeline set has no L2 bandwidth counter; Nsight Compute supplies it separately.

| Chains | SM throughput (% peak) | Achieved occupancy (%) | L2 bytes (% peak) | DRAM bytes (% peak) |
|---:|---:|---:|---:|---:|
| 16 | 1.3347 | 6.25 | 0.48688 | 0.005814 |
| 128 | 10.773 | 6.25 | 3.8953 | 0.047273 |
| 512 | 38.519 | 23.861 | 14.38 | 0.17094 |

Short stationary-kernel Nsight Compute probes. Raw L2 hit-rate ratios can exceed 100% across replay passes; those values remain flagged in the source. A separate three-pass hit/(hit+miss) estimate gives 99.837%, 99.990%, 99.975% at 16/128/512 chains ([raw counts and definitions](nsight_timeline/ncu/l2_hit_rate_validation.csv)). These are replayed count estimates, not simultaneous single-pass counters. Top warp stalls and FLOP/byte roofline inputs are saved separately.

## Barrier preparation: host rotation versus cuBLAS

The original phase contains CPU Eigen A×Q, costing approximately 2md² FLOPs. An isolated build switches only that multiplication to cuBLAS, returning the result to the existing host pipeline. Both modes include transfers and allocations; mathematical eigenbasis construction is unchanged. NVTX ranges separate host stages from GPU kernels. [Full stage data, commands and trace references](barrier_profile/summary.json) · [Patch](barrier_profile/barrier.patch) · [CPU-path byte-parity and five analytic tests](barrier_profile/verification/verification.json).

| Model | m × d | CPU A×Q (s) | GPU A×Q incl. copies (s) | Original barrier (s) | Offloaded barrier (s) | Full-phase speedup |
|---|---:|---:|---:|---:|---:|---:|
| iJO1366 | 750 × 581 | 0.056966 | 0.0037453 | 0.1491 | 0.1051 | 1.4186 |
| RECON1 | 2120 × 931 | 0.4059 | 0.050546 | 0.5078 | 0.1938 | 2.6202 |
| Recon3D_nullspace_only | 18477 × 4861 | 92.932 | 0.49388 | 93.86 | 1.4728 | 63.729 |

Times are unprofiled same-binary observations; profiled runs provide attribution and overhead checks. Algebra-validation runs are excluded from performance. Short sampling checks establish feasibility, not convergence. Recon3D uses the separately labeled equality-only/no-MVE geometry; its one-time equality elimination and model preprocessing remain separate costs. RECON1 retains its declared 1e-6 geometry tolerance; other targets use 1e-7. A row marked first pair precedes completion of the full large-matrix reference check and CPU trace.

Recon3D_nullspace_only full CPU/GPU A×Q matrix comparison passed: relative Frobenius error 2.396e-15, maximum absolute error 6.106e-15. This validation is separate from timing.

iJO1366 full CPU/GPU A×Q matrix comparison passed: relative Frobenius error 1.124e-15, maximum absolute error 1.251e-12. This validation is separate from timing.

## Matched-quality 512-chain experiment

[All attempts, validation and selected candidates](high_throughput/iJO1366/summary.json). Fresh, nonoverlapping replicate seed ranges. CPU variants use 48 or 64 distinct physical cores, explicitly separated. Partial groups support no three-seed CPU/GPU headline.

| Candidate | Validated passing seeds | Status |
|---|---:|---|
| cpu | 3/3 | complete |
| gpu | 1/3 | partial |
| gpu500 | 1/3 | partial |
| cpu48 | 1/3 | one_seed_preliminary |
| gpu_prefault | 2/3 | partial |
| gpu_coordinate | 3/3 | complete |

Sampling-phase and worker draw-call measurements will lead this comparison. Process-wall and transfer effects remain separately recorded in the source.

## Additional model quality coverage

[Per-seed records and operational times](larger_quality/table.csv) · [Validated groups](larger_quality/summary.json). 16 chains, median split-Rhat ≤1.01 and bulk ESS ≥100; bounds checked independently.

| Model | Method | Passing seeds | Max sample bound fraction at 1e-7 |
|---|---|---:|---:|
| iAF1260 | gpu | 3/3 | 0 |
| iAF1260 | cpu2000 | 3/3 | 0 |
| iAF1260 | cpujade | 3/3 | 0 |
| iML1515 | gpu | 3/3 | 3.125e-05 |
| iML1515 | cpu2000 | 0/3 | pending |
| iML1515 | cpujade | 3/3 | 0 |
| iMM904 | gpu | 3/3 | 0.028188 |
| iMM904 | cpu2000 | 3/3 | 0.029587 |
| iMM904 | cpujade | 3/3 | 0.030125 |

Passing convergence diagnostics do not imply zero bound violations. iMM904 residuals around 2e-7 and the iML1515 seed-specific residual remain visible; no samples are silently filtered. Output-excluded cross-model timing summaries are being assembled.

## Float64 export follow-ups

Separate verified float64 builds repeat the same sampling protocols to isolate export precision. A corrected group requires the shared quality target and zero original-bound violations at 1e-7; the paired cast-back comparison tests whether the earlier float32 trajectory is reproduced. Canonical float32 records are preserved.

| Model | Seeds passing quality and strict bounds | Three-seed group | Median sampler phase (s) | Evidence |
|---|---:|---|---:|---|
| iJR904 | 3/3 | True | 2.7324 | [per-seed checks](precision_quality/iJR904/summary.json) |
| iML1515 | 3/3 | True | 8.1171 | [per-seed checks](precision_quality/iML1515/summary.json) |

The sampler phase excludes download/writes; export precision changes are not automatically a sampler-compute speedup. Partial attempts remain visible in the source.

## Isolated coordinate-kernel specialization

The experimental build removes unused mixture paths only when all noncoordinate probabilities are zero. Compiler register usage at TPB128 falls from 104 to 64 per thread, with no spills; the generic mixture path is retained. This changes the register residency ceiling from four to eight blocks/SM. Compiler resource savings alone are not a measured speedup.

GPU analytic/trajectory validation: **passed**. [Patch](coordinate_kernel/kernel.patch) · [Compiler resource records](coordinate_kernel/compiler_resources.json) · [Complete source snapshot](coordinate_kernel/source.tar.gz) · [Validation](coordinate_kernel/verification.json). Canonical source/builds/results are unchanged by this isolated experiment.

[Paired timings and commands](coordinate_kernel/paired/summary.json). Single-seed, same-GPU comparison; all probabilities for noncoordinate moves are zero. Both builds use host prefault; timing below excludes download, writing and normalization.

| Chains | Original sampler (s) | Specialized sampler (s) | Sampling speedup | Full retained arrays identical |
|---:|---:|---:|---:|---|
| 16 | 11.87 | 9.1922 | 1.2913× | True |
| 512 | 13.056 | 10.085 | 1.2947× | True |
| 1024 | 26.041 | 13.181 | 1.9756× | True |

This is a paired implementation/throughput experiment. It does not itself establish a new CPU speedup or convergence result.

### Same-GPU specialization replication: 512 chains

Fresh paired runs balance build order across seeds on one allocation, preserve complete retained-array identity, and log GPU identity/clocks/power as diagnostics. Those clock logs are separate from the Nsight utilization figure.

| Seed | Original sampler (s) | Specialized sampler (s) | Paired ratio |
|---:|---:|---:|---:|
| 101001 | 13.026 | 13.26 | 0.98231× |
| 202001 | 12.991 | 13.048 | 0.99559× |
| 303001 | 13.086 | 13.023 | 1.0049× |

Median paired ratio: **0.9956×**. The earlier single-seed 512-chain improvement did not replicate here; it is not a general speedup claim. [Raw pairs, commands and clock diagnostics](coordinate_kernel/clock_audit/summary.json).

### Same-GPU specialization replication: 1024 chains

Fresh paired runs balance build order across seeds on one allocation, preserve complete retained-array identity, and log GPU identity/clocks/power as diagnostics. Those clock logs are separate from the Nsight utilization figure.

| Seed | Original sampler (s) | Specialized sampler (s) | Paired ratio |
|---:|---:|---:|---:|
| 101001 | 25.987 | 26.178 | 0.9927× |
| 202001 | 26.029 | 25.883 | 1.0056× |
| 303001 | 26.159 | 25.838 | 1.0124× |

Median paired ratio: **1.006×**. [Raw pairs, commands and clock diagnostics](coordinate_kernel/clock_audit1024/summary.json).

## Recon3D exploratory barrier-only probe

Successfully sampled an equality-eliminated Recon3D target: n=10600, d=4861, m=18477. This separate geometry skips PolyRound MVE rounding, uses GPU whitening and float64 export, and preserves the original model target. It does not replace the still-running canonical preprocessing.

Barrier preparation 93.9 s; sampler phase 8.029 s. 16 chains, 100 discarded warmup + 200 retained draws per chain. All 1024 deterministically selected samples passed original bounds at 1e-7. **Feasibility/throughput probe only: convergence was not established.** [Evidence](exploratory_recon3d_barrier/seed101/probe_summary.json) · [Jobs and reproduction](exploratory_recon3d_barrier/jobs.json).

## Conditioned-rounding prototype

A separate iJO1366 PGI capacity restriction [-0.1, 0.1] compares full rerounding, stale shared rounding, and shared rounding plus Naja correction. This is explicitly a capacity restriction, not an exact knockout. [Live phase/quality table](conditioned_prototype/TABLE.md) · [Protocol and exact-target audit](../audit/conditioned_reuse_assessment.md). Setup/correction and sampling efficiency are primary; total process times are secondary.

The independent raw-model reround produced 582 dimensions versus 581 in the inherited geometry. Those original full-reround results are **incompatible as a same-target reference**, despite passing sample feasibility checks. Earlier raw-reround speedup figures are suspended. [Dimension audit](conditioned_prototype/dimensionality_warning.json).

The repaired reference globally rerounds the exact cached 581-D system without repeating equality elimination. All 752 transformed inequalities match the shared target (relative A error 0; relative b error 1.57e-13); the map is invertible and the conditional diagnostic mask matches. Global rounding itself took 61.204 s. Sampling quality and fresh selected-budget timings for this repaired reference are being measured. [Algebraic target-equivalence proof](conditioned_prototype/iJO1366/models/global_reround/global_reround_proof.json).

### Same-target convergence curves and fresh time-to-target

PGI capacity prototype, median split-Rhat ≤1.05 and bulk ESS ≥100, with original/conditional bounds and mass balance ≤1e-7. The repaired global reference and inherited methods use the same cached target and conservative common LP mask. Earlier 878-mask curves are historical. Only separately executed selected budgets populate times; prefix points have no inferred times.

[Rhat SVG](conditioned_prototype/iJO1366/quality/rhat_vs_chain_length.svg) · [PDF](conditioned_prototype/iJO1366/quality/rhat_vs_chain_length.pdf) · [Source curve CSV](conditioned_prototype/iJO1366/quality/rhat_curve.csv) · [Fresh time-to-target CSV](conditioned_prototype/iJO1366/quality/time_to_target.csv).

![Conditioned-target convergence curves](conditioned_prototype/iJO1366/quality/rhat_vs_chain_length.png)

[Actual-transition-axis figure](conditioned_prototype/iJO1366/quality/rhat_vs_transitions.svg); this includes each method’s thinning and retained warmup count.

| Pipeline | Seed | Status | Selected retained draws | Sampler compute proxy (s) | Rhat | Bulk ESS |
|---|---:|---|---:|---:|---:|---:|
| global_cpu | 101 | complete | 2000 | 3.8167 | 1.0419 | 480.69 |
| global_cpu | 202 | complete | 2000 | 3.8109 | 1.0426 | 461.35 |
| global_cpu | 303 | complete | 2000 | 3.8067 | 1.0422 | 455.42 |
| naja_rotate | 101 | complete | 500 | 3.8936 | 1.0384 | 466.56 |
| naja_rotate | 202 | complete | 500 | 3.8738 | 1.0364 | 522.66 |
| naja_rotate | 303 | complete | 500 | 3.8919 | 1.0383 | 522.68 |
| stale_cpu | 101 | no_quality_within_cap | pending | pending | pending | pending |
| stale_cpu | 202 | no_quality_within_cap | pending | pending | pending | pending |
| stale_cpu | 303 | no_quality_within_cap | pending | pending | pending | pending |

CPU compute is the accumulated draw-call critical-path proxy; Naja uses its sampler phase including allocation/slack/RNG initialization. Setup/correction costs are separate. Across the three measured passing seeds, sampler times are essentially tied; the reuse benefit is avoiding the 61.2 s global rerounding step. Real zero-flux knockout results follow separately. [All statuses, source runs and target proof](conditioned_prototype/quality_summary.json).

Stale inherited rounding misses the target in 3/3 seeds at 50,000 retained draws per chain: median split-Rhat across seeds ranges from 1.1315 to 1.1334, and median ESS from 53.8 to 54.4. This is the finite 50k-draw cap, not a claim of permanent nonconvergence.

## Real gene-knockout panel: geometry and starting points

The catalog contains 1366 real genes and 803 distinct bound-change signatures. 51 unique targets currently have validated sampling geometry (50 nontrivial plus a control). Geometry preparation is complete for the panel; the detached sampling consumer is continuing. Geometry readiness does not imply sampling convergence.

The final validated panel has 41 nontrivial targets with 1–5 changed rows and 9 with 6–20, plus one control. It is not fully stratified: 21–50-row candidates exist biologically but attempted faces failed the geometry certification gates; the catalog has no 51–200-row gene knockouts. [Independent final geometry audit](knockout_panel/final_geometry_audit.json).

The unchanged base point is weakly feasible at 1e-7 for 48.46% of all genes, but only 1.40% of the 714 genes classified as affecting LP-varying reactions. The larger population includes many unchanged/redundant targets and is not a nontrivial skip rate.

Analytic projection avoided all interior LP calls for 21/50 currently ready nontrivial targets. This is a validity-selected subset of the gene catalog. PGI/b4025 is a genuine zero-flux knockout with a validated 580-D target and zero interior LP calls. Unstable affine-rank/dual certificates remain unsupported, with rejection evidence retained; no bound relaxation is used.

[Full catalog and denominator definitions](knockout_panel/gene_catalog.json) · [Ready target manifests](knockout_panel/ready_manifest.json) · [All geometry attempts](knockout_panel/geometry_summary.json).

A stronger cached-target certificate uses the feasible ball of radius 0.999988: reaction variation is at least 2 × radius × the norm of its affine-map row. Among 704 genes affecting a reaction with certified variation above 1e-7, **0** can keep the unchanged base point at the knockout bounds. Analytic projection, rather than unchanged-start reuse, is therefore the useful LP-skipping route in the validated examples.

### Live panel sampling progress

Counts below describe individual method/seed tasks on validated targets. Geometry-ready targets without a sampling task are still pending; mechanical failures and bounded retries remain in their source records.

| Method | Completed | Recorded unfinished | Other recorded states |
|---|---:|---:|---|
| global_cpu | 50 | 0 | — |
| naja_rotate | 42 | 0 | no_quality_within_cap: 9 |
| stale_cpu | 33 | 0 | no_quality_within_cap: 18 |

[Live per-target curves, measured timings and statuses](knockout_panel/quality_summary.json).

### Completed three-pipeline exact-knockout comparisons

One seed per target; each entry is retained draws per chain / measured sampler proxy seconds. All entries pass median Rhat ≤1.05, ESS ≥100 and unmasked feasibility checks. Setup is separate; the unchanged control is excluded.

| Gene KO | Full reround + CPU | Inherited + CPU | Shared + Naja correction |
|---|---:|---:|---:|
| b4025 | 2,000 / 3.963 | 2,000 / 3.962 | 500 / 4.031 |
| b0343 | 2,000 / 3.895 | 2,000 / 3.915 | 250 / 3.329 |
| b0508 | 2,000 / 4.083 | 2,000 / 3.898 | 250 / 3.375 |
| b0019 | 2,000 / 3.894 | 2,000 / 3.894 | 250 / 3.363 |
| b0030 | 2,000 / 3.879 | 2,000 / 3.888 | 250 / 3.373 |
| b0032 | 2,000 / 3.897 | 2,000 / 3.923 | 250 / 3.358 |
| b0104 | 2,000 / 3.944 | 2,000 / 3.929 | 250 / 3.382 |
| b0114 | 2,000 / 3.971 | 2,000 / 3.909 | 500 / 4.015 |
| b0116 | 2,000 / 3.893 | 2,000 / 3.884 | 500 / 4.007 |
| b0404 | 2,000 / 3.884 | 2,000 / 3.949 | 250 / 3.357 |
| b0474 | 2,000 / 3.841 | 2,000 / 3.837 | 250 / 3.382 |
| b0721 | 2,000 / 3.966 | 2,000 / 3.906 | 500 / 4.001 |
| b1326 | 2,000 / 3.897 | 2,000 / 3.887 | 250 / 3.385 |
| b1681 | 2,000 / 3.874 | 2,000 / 3.835 | 500 / 4.057 |
| b1744 | 2,000 / 3.943 | 2,000 / 3.944 | 250 / 3.353 |
| b2028 | 2,000 / 4.071 | 2,000 / 3.952 | 250 / 3.351 |
| b2257 | 2,000 / 4.029 | 2,000 / 3.973 | 250 / 3.335 |
| b2436 | 2,000 / 4.097 | 2,000 / 3.898 | 250 / 3.380 |
| b2529 | 2,000 / 3.907 | 2,000 / 3.883 | 500 / 3.987 |
| b2920 | 2,000 / 3.919 | 2,000 / 3.960 | 250 / 3.365 |
| b3059 | 2,000 / 3.986 | 2,000 / 3.867 | 250 / 3.375 |
| b3417 | 2,000 / 3.880 | 2,000 / 3.874 | 250 / 3.360 |
| b3744 | 2,000 / 3.893 | 2,000 / 3.904 | 250 / 3.337 |
| b3918 | 2,000 / 3.899 | 2,000 / 3.874 | 250 / 3.390 |
| b3924 | 2,000 / 3.971 | 2,000 / 3.891 | 250 / 3.360 |
| b4041 | 2,000 / 4.067 | 2,000 / 3.878 | 250 / 3.376 |
| b4077 | 2,000 / 3.872 | 2,000 / 3.879 | 500 / 4.024 |
| b4481 | 2,000 / 3.919 | 2,000 / 3.841 | 500 / 4.035 |

Inherited rounding here includes the affine-hull projection required for exact zero-flux knockouts. That projection can improve mixing: stale inheritance passes on the first exact PGI knockout, unlike the finite-width capacity restriction above. These targets therefore do not establish a universal stale-rounding failure.

## Starting-point LP: persistent solver reuse

The PGI-capacity microbenchmark changes the actual neighboring slab bounds while retaining the same LP matrix. Starts are checked against the full saved inequalities; sampling targets are unchanged. This tests solver-state reuse, not exact-knockout affine-hull handling.

| HiGHS threads | New model, initialized pool (s) | Explicitly reset basis, changed slab (s) | Internal basis/factorization retained, changed slab (s) |
|---:|---:|---:|---:|
| 0 | 1.6621 | 0.25143 | 0.013104 |
| 1 | 1.5396 | 0.24099 | 0.012899 |
| 4 | 1.5184 | 0.2428 | 0.012857 |

Retained-state solves take about 13 ms here, versus roughly 1.5 s for new models. Calling setBasis repeatedly discards useful internal state and costs about 0.24 s. Imports, model assembly and the first solve are one-time costs saved separately. Thread-count changes did not explain the latency difference in this test. The real-knockout panel will separately measure base-start feasibility, analytic projection, facial reduction and required LP calls. [Raw calls, accepted options and residual checks](knockout_panel/lp_microbench_retained/summary.json).

## Follow-up priorities requested at 04:30 UTC

- Conditioned quality: three same-target pipelines, split-Rhat versus retained chain length and fresh measured time-to-target; then a panel of approximately 50 real gene knockouts.
- Starting-point cost: base/projection feasibility checks, cold versus repeated LP solves, and explicit HiGHS model/basis reuse. Trivial/redundant and nontrivial knockout denominators remain separate.
- iML1515: add the validated Naja/guideline-CPU time-to-target row and complete the thinning-2,000 fixed-budget seed audit; unrun seeds are not counted as failures.
- Fresh fixed-budget 16-chain validation and completion of the 512-chain matched-quality groups use new seed ranges and preserve failed budgets separately.
