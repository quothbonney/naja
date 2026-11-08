# Independent benchmark review

Review snapshot: 2026-11-07 23:40 UTC onward. The campaign is still producing results; this review validates the protocol and saved records inspected, not completion of every planned experiment.

## Outcome and a corrected reporting defect

The inspected canonical sampling records use matched model geometry, the intended chain/sample layout, correctly discarded warmup, common reaction-space diagnostics, and explicitly measured timing scopes. No sampling rerun was required by this review.

A **critical reporting contamination was found and corrected**: the original recursive results collector allowed iJO1366 scaling runs with 16 chains and 10,000 draws into the fixed-budget table, because their shape and seed matched the canonical protocol. That could mix TPB64/128/256/512 and count seed101 repeatedly. Fixed-budget reporting and paired speedups now accept only `results/naja/**` and `results/hopsy/**`. All exploration remains in `all_runs.csv`, labeled by experiment family. Sampler configuration includes method, actual/requested thinning, workers, barrier, TPB and pair probability. Duplicate canonical configuration/seed records are rejected and usable seeds are counted uniquely. The iJO1366 canonical Naja group contains its actual three seeds rather than the extra scaling executions.

## Targets, models and affine geometry

The same saved PolyRound `polytope.npz` (A, b, start, T, shift) is consumed by hopsy and Naja for each accepted model. Both target the inequalities `A z <= b`; reaction-space metrics use the identical `x = T z + shift`. Naja undoes its internal barrier rotation/whitening before saving reduced-space output, so applying T again is correct; it is not a double backmap. No growth/biomass cut or new medium is silently added. The original BiGG bounds can be very large and allow cycles; this is the documented raw model target, not a numerical-data corruption inferred from large fluxes.

`results/setup/model-static-checks.json` records independent source hashes, finite arrays, feasible starts, original bound checks and stoichiometric affine-map residuals for the seven accepted models. All source-hash checks pass; starting bound violations are below 1e-7; direction residuals are below 8e-9. Shift/start residuals are also saved and should be read at their actual scale rather than described as identically zero. e_coli_core preprocessing ran on the controller Xeon 8562Y+, while other preparations ran on the Slurm Xeon 8462Y+; the report explicitly separates this hardware difference.

RECON1's original model is feasible; failed PolyRound attempts are preprocessing/backend failures, not evidence of biological infeasibility. The unchanged source hash, unmodified bounds, backend/settings, job states and errors are preserved in `results/setup/RECON1_preprocessing_attempts.json`. A replacement preparation must pass its validation gate before entering canonical sampling. Recon3D/RECON1 status remains explicit until valid artifacts exist.

## Chain layout, RNG state and warmup

Actual Naja files for e_coli_core seed101, iAB_RBC_283 seed202 and iJO1366 seed303 were checked independently. Native C-order rows are draw-major interleaved chains. For multiple early/late draws and chains, normalized `[chain, draw]` vectors are bitwise identical to native row `((draw + 1000) * n_chains + chain)`. Native output length is exactly `chains * (draws + warmup)`. Evidence: `audit/native-layout-independent-check.json`.

Naja's localized explicit-seed override is reflected in profile metadata and has a saved same-seed/different-seed test: same seeds reproduce bitwise, different seeds differ, and synthetic cube outputs are finite and feasible (`results/seed-check/verification.json`). hopsy seeds are `seed + chain_index`; the three canonical base seeds 101/202/303 have disjoint 16-chain ranges. Common initial points with independent RNGs are not overdispersed initialization; Rhat is therefore a useful diagnostic but not a proof of full support exploration.

Installed hopsy source updates proposal and RNG state between sampling calls. An independent synthetic test confirms that serial sampling in two 32-draw calls matches a 64-draw call bitwise, and that one parallel 64-draw call matches serial sampling bitwise; different chains differ. Evidence: `audit/hopsy-state-check.json`.

A second parallel hopsy.sample call on returned chains failed model unpickling in a separate test and left its worker pool waiting. This hopsy1.7 behavior is preserved in `logs/hopsy-state-verification.multichunk-failure.log`; that synthetic job was cancelled. **Current canonical records are unaffected:** all inspected multiworker hopsy runs use a single chunk covering draws plus warmup, and `audit/multichunk-parallel-records.json` is empty. CPU wrapper defaults should continue to enforce the supported single-call path. Serial one-worker continuation is verified.

## Diagnostics, numerical constants and bounds

Primary schema-2 diagnostics use the same model-derived reaction mask for every method. Candidate tiny rows of T are checked by original-model GLPK minimum/maximum flux LPs; only successful solver-detected ranges <= 1e-7 are excluded, at declared solver feasibility tolerance 1e-9. Wider or unresolved candidates remain active. The mask affects diagnostic aggregation only and never changes the sampled polytope.

The independent audit in `audit/diagnostic-mask-independent-verification.json` recomputed source and geometry hashes, verified every saved array entry against per-reaction LP evidence and IDs, and confirmed all seven exclusion counts. Small negative min/max widths occur within declared tolerance; this is solver-detected nearconstancy, not an exact-arithmetic structural-blockage theorem. Exactly constant sampled reactions are additionally excluded, with counts saved. The old empirical 1e-12 aggregation remains in each new metrics file for comparison, along with measured ranges of solver-excluded reactions.

Classical split-Rhat uses two half-chains per original chain, within-chain sample variance, between-chain variance and the usual variance-ratio estimate. Bulk ESS uses the same ArviZ pooled rank-normalized implementation for both methods. Prefix optimization skips masked reactions only for prefix work; the full-budget legacy aggregate remains available. Synthetic IID versus separated-chain tests, affine mapping, layout, constants and independent bound counts pass (`logs/metrics-verification.log`).

All bounds diagnostics include every reaction and every sample, including excluded nearconstant reactions. The main/quality tolerance is absolute **1e-7 flux units**, and the report's violation value is a fraction, not a percentage. Coordinate fractions, any-reaction sample fractions, maximum absolute violation and exact zero-tolerance fractions are preserved separately. No post-filter zero-violation result is substituted for pre-filter validity.

At this snapshot, iMM904 Naja has roughly 2.7–3.1% of samples above the 1e-7 bound tolerance, with maximum violation about 2.0e-7; this is small numerical error, but it does **not** support an unqualified zero-violation claim. One inspected iJR904 20,000-draw quality run has one violating sample among 320,000 (fraction 3.125e-6). Exact zero-tolerance failures can occur even for otherwise accurate affine roundoff and are not interchangeable with the tolerance-qualified statistic. Uniformity/scaling artifacts use 1e-6 and explicitly disclose that different tolerance.

Old schema-1/provisional diagnostics still being refreshed are excluded from primary table quality cells and speedup ratios. A final stale-metrics sweep is required before publication tables are frozen; immutable prior metrics snapshots are retained.

## Timing and performance interpretation

CPU full-process timing begins before NumPy/hopsy imports and includes load, chain initialization, warmup, sampling and output writes. GPU native wall is externally measured around the executable; adapter time is displayed separately. The primary conservative GPU comparison adds adapter time, which includes a second output copy and is benchmark-harness overhead rather than required Naja work. Older CPU timings without import/startup are explicitly provisional and excluded from paired speedups. Diagnostics and PolyRound preprocessing are excluded from sampling wall consistently; preprocessing is reported separately.

Internal GPU phase timings are not independent bins: on the reduced rotated path, `backtransform_s` includes `download_from_gpu_s`. The report warns against summing phase columns. Naja internally samples in double precision but saves reduced samples as float32, whereas hopsy saves float64; this precision difference remains disclosed. Multiple Slurm jobs share the node/filesystem while reserving their own CPU/GPU resources, so these measurements are not a locked-clock, otherwise-idle-node microbenchmark.

Sixteen chains use at most 16 independent hopsy sampling workers; this is not a claim that one run uses all 128 logical CPUs. One-worker CPU variants are retained where tested, and the fastest completed tested CPU variant is selected per seed for the raw equal-sample comparison. Default hopsy thinning=1 means unthinned output; it is not the legacy `8*d^2` rule. The Jadebeck label records the actual `ceil(d^2/6)` guideline. Tested fixed-thinning variants remain separate.

Equal-sample speedup is not an equal-quality claim. Several larger models fail median Rhat<=1.01 at 10,000 retained draws for Naja, and unthinned hopsy often fails much more strongly. Both methods' Rhat/ESS must accompany any raw speedup. Selecting the fastest passing CPU variant from a limited tested set does not establish a global CPU optimum.

## Exact-budget quality records

An independent snapshot audit checked **21 completed measured quality records** for actual sample-array budget/shape, 16 chains, 1,000 discarded warmup draws, complete sampling status, a metrics file newer than the actual saved samples, schema-2 verified masks, the required median Rhat/ESS threshold, and equality of the reported time with the actual run's full-process timer. All checks passed. Each metric points to its measured run's samples rather than the source trajectory. Evidence: `audit/quality-independent-verification.json`.

The measured numbers are executions at selected passing budgets, not prefix-proportional time estimates. Prefixes or larger pilot budgets choose a finite-grid candidate. A cached actual execution at that exact budget can be reused as the measured record; this remains an actual measured budget, not a fresh statistically independent trajectory. Same-seed reruns are independent process executions, not independent RNG replicates. Separate seeds provide the replicate groups. Search/diagnostic costs are excluded, so these numbers should be described as measured wall for a selected passing budget, not total cost of an automated adaptive stopping procedure. Underconverged candidates retain explicit no-quality/cap/failure status; incomplete three-seed groups remain incomplete.

The convergence criterion tests median Rhat/ESS only. Passing it does not waive the separately reported bound checks.

## Uniformity and scaling scope

The e_coli_core empirical comparison uses one million retained samples per method, the shared 87-active-reaction model mask, two independently seeded hopsy references and equal-count empirical Wasserstein-1 distances. Sorting equal sample counts gives exact empirical univariate W1; normalization uses pooled reference standard deviation. Correlations use common reaction order and the Frobenius distance; division by reaction count is entrywise RMS over the square matrix. The independent hopsy-vs-hopsy comparison is the empirical floor. Naja rotate, whiten-plus-pair and the later verified whiten/Jacobi-pair/Dikin mixture provide empirical agreement evidence on this model, not a general uniformity theorem; the later mixture evidence is detailed below.

The chain/TPB sweep changes one factor at a time on iJO1366 at seed101 and fixed retained budget per chain. Kernel throughput includes/excludes warmup according to explicitly named fields. Native process and adapter walls are separate. One-second device telemetry can miss memory peaks and includes host phases, so its average utilization should not be called kernel occupancy. Feasibility checks sample only 1,024 deterministic reduced-space outputs per case at tolerance 1e-6; they are not exhaustive correctness or convergence checks. No multi-GPU/streaming/production result is implied by this single-GPU throughput sweep.

## Before paper-level claims

Freeze completed three-seed groups, refresh all stale diagnostic schemas, retain failed/pending models and preprocessing attempts, and cite the exact model/source/binary hashes and tolerances. Report measured passing-budget comparisons separately from raw equal-sample throughput. Avoid unsupported historical-speedup reproduction, all-nine completion, exact-zero violations, global CPU optimality, overdispersed initialization, general uniformity or multi-GPU scaling claims.

The production corpus is outside this prototype's scope and is not a blocker for these BiGG comparisons.

## Update: 2026-11-08 — new precision and build evidence

RECON1 GLPK/presolve preprocessing completed in 1,549.53 seconds with n=3741,
d=931, m=2120. Its original bounds and saved arrays remain unchanged. The strict
1e-7 check failed narrowly: center bound residual 1.14054e-7, pilot bound
residual 1.52823e-7, equality residual about 4.12e-8 and S/T direction residual
about 6.22e-9. The root explicitly declared a separate 1e-6 preprocessing
verification gate. `validation.strict1e-7.json` preserves the failure;
`validation.json` records the new tolerance and `strict_1e_minus_7_passed=false`.
This is a precision-limited preparation, not strict zero-bound verification.
Sample metrics still check all reaction bounds at 1e-7 and zero tolerance;
no sampling-target relaxation or filtering was added. The common FVA mask is
now cached, with 1,274 solver-detected nearconstant reactions excluded and
approximately 24.2 seconds of LP-mask work. CSV fields preserve validation
status/tolerance, strict-pass flag and center/pilot residuals.

The e_coli_core empirical comparison now additionally includes a verified
whiten/Jacobi-pair/Dikin mixture: pair probability 0.5, Dikin 0.15, coordinate
0.35, with 11 runtime Dikin directions. `mixture_verification.json` records
source/binary evidence and confirms the reduced-output path; the same cached
87-active-reaction mask is used. This broadens empirical agreement evidence on
this one model; it is still not a general Dikin/mixture uniformity theorem.

The new hopsy build experiment is explicitly a one-seed pilot, not canonical
three-seed evidence. The official SDIST source-generic and native builds share
build ID 9b787d9, GCC 11.4.0, source archive and runtime; the native flags add
`-march=native -mtune=native` to `-O3`. Source-generic sampling took 44.1321 s
and native sampling 31.3898 s (1.40594x); process times were 51.0117 and
36.9578 s (1.38027x). The installed wheel reports build dadab7b, so a
wheel/native comparison alone does not isolate compiler ISA flags. Source and
wheel hashes and environment signatures are saved. Analytic cube tests passed.
Near-cutoff Rhat changes slightly between floating-point trajectories; the
pilot does not establish improved mixing or a three-seed fastest-CPU result.
Native passing-budget groups remain incomplete until their actual seeds pass.

A separate optional 512-chain probe exposed a real wide-index failure in the
frozen Naja binary: signed 32-bit element products overflowed for 3,272,192,000
matrix elements. CUDA allocation/illegal-access and cuBLAS errors occurred
while the process returned exit status zero. The original probe records are
explicitly `invalid_backend_errors`, with empty performance rows; they are not
throughput evidence. The backend-log audit found these errors in the extension
attempts and did not reclassify the canonical experiments as failed.

A separate cast-only fix widens DMatrix allocation/copy products and IDX2C
offsets before multiplication. Binary da263d6060416d96da366392210b9468422890bda8200718da452941436a8f18
passed all five GPU tests and matched same-seed BOX output bitwise against the
frozen c7baf4... binary. Valid reduced-output 512-chain probes used the separate
build and approximately 50,479 MiB sampled GPU memory. TPB128 achieved
1,116,879.7 retained samples/s in the sampling phase; TPB512 achieved
458,912.9, so small-chain TPB tuning must not be generalized blindly. These are
single-pass exploratory measurements, not canonical seed groups or full
convergence checks. Full saved-array finiteness and only 1,024 sampled
feasibility points are distinct checks. The separate patch does not claim all
other large-array/backmap/reader paths were widened.

Runner guard tests now reject zero-exit GPU backend errors, unsupported INT_MAX
products, nonfinite warmup or retained output and wrong dimensions. Native and
experimental binary labels/hashes remain in CSV metadata. Canonical collectors
explicitly require the frozen Naja hash or wheel CPU build, in addition to
canonical family paths, so these exploratory builds cannot silently enter the
fixed-budget table or its ratios.

### RECON1 output warning supersedes any inference from the small pilot residual

The accepted preprocessing center/pilot residual is not a bound on later output
errors. Current frozen float32 reduced-output Naja runs have maximum original
reaction-bound errors approximately 0.00084 / 0.00177 / 0.0165, including varying
reactions such as DALAxt, GTHPi and ALASERNaEx. These are materially larger than
the 1e-6 preparation gate. The ~0.59 any-reaction sample fraction is largely
associated with blocked-reaction numerical residue; coordinate fractions are
much smaller (~0.0002), but this does not erase the large maxima. The paired float64-export/cast-back experiment below now isolates export
quantization for seed303 while retaining small blocked-row residue.
All pre-filter fractions and maxima remain evidence. Rhat/ESS passing would
not establish exact original-bound compliance. The report now states this
warning explicitly next to the RECON1 preparation caveat.

### Bounded parallel common diagnostics

The common estimator now supports hash-verified spawned workers for large workloads, capped at four and the Slurm CPU allocation. It uses readonly sample/backmap inputs, BLAS1 per worker, indexed per-reaction assembly and boolean OR for sample violations. Archived serial, refactored serial and four-worker results matched every scientific field to 1e-12 and their reaction CSVs matched bitwise on core and a real iJO1366 50k-draw/128-reaction audit projection. Current proof and timings are in metrics-parallel-verification.json; the estimator source is frozen at the recorded SHA. Small workloads remain serial (spawn was slower there); the heavy projection improved from36.05s to17.85s, about2.02x. Sampling wall times remain unchanged because diagnostics are excluded. Strict cache identity uses the producing imported module SHA; older results without that provenance remain valid but cannot be mislabeled as produced by the new source.


## Paired precision and host first-touch evidence (2026-11-08)

The same RECON1 seed303 double-precision trajectory, backmapped in float64
before/after explicit float32 reduced-coordinate quantization, establishes
export quantization as the cause of the observed large varying-reaction errors.
The quantized reduced array matches historical frozen output bitwise. Maximum
nonexcluded reaction-bound violation changes from 0.0165078408 to zero;
maximum full-output float64 violation remains 2.7091309e-7 on LP-excluded
nearconstant rows. The unfiltered any-reaction fraction at 1e-7 remains
0.59409375. No clipping/filtering or sampling-target change was applied.
This establishes the cause for this trajectory, not all models/seeds or strict
1e-7 feasibility. See [paired evidence](../results/precision/RECON1/seed303/summary.json).
The earlier statement that export precision was only a hypothesis is superseded.

The opt-in float64-export/wide-index binary SHA256 is
`5e6968eb7d3cebf5f27c16d2662a2e640cc3f6b7a633b7e7ae3bc9136a99f3d5`;
complete source patch SHA256 is
`33f9f14beb50cb2aa5b7aa37c61f2868f04b15cd66461f0537751a1fb3aa5d2a`.
[Build proof](../results/f64-build-verification/verification.json) records five GPU tests,
unchanged sampling math, historical/default float32 bitwise equality, and
float64 dtype/cast checks. It does not replace canonical float32 results.

The optional first-touch binary SHA256 is
`a656a1f93489af051acd540d08369092567dd0673af2b0e1ac922dc58a910603`;
complete source patch SHA256 is
`d891970c48c1c733243ec85812b420cf17f9ddb7349f201f770edcbef09bc568`.
[Build proof](../results/prefault-build-verification/verification.json) records five GPU
tests and bitwise output comparisons. Its small verification fixture alone did
not test large arrays; the later [512-chain iJO1366 paired experiment](../results/host_prefault/iJO1366/summary.json)
did execute the large path successfully. Three fresh-process off/on pairs used
one H100 and eight CPUs, seed101, TPB128, 10k retained + 1k warmup/chain; all
retained arrays have identical SHA256. Median paired speedups were 4.394x for
download, 1.471x native process and 1.232x including normalization. Median wall
including normalization changed from 52.1532 to 42.4494 s. Download includes
allocation, first touch, thread creation/joins, final-byte touch and cudaMemcpy;
normalization is an additional harness copy. Costs were not shifted outside the
timer. First touch defaults off. These shared-node measurements concern one
model/seed, not general convergence, exhaustive feasibility or global tuning.

## Physical-core CPU placement control

The separate native iJO1366 group completed all three seeds, reserving 32 logical
CPUs and pinning 16 workers to 16 unique physical cores (one socket/run).
Earlier 16-logical-CPU allocations can pack 16 workers onto eight physical
cores. Process walls are 25.8015, 26.0029 and 24.3063 s (median 25.8015 s).
All three pass median Rhat≤1.01 and ESS≥100; all bound fractions are zero at
1e-7. Each output is bitwise identical to the earlier native same-seed sample.
Against Naja TPB512 median inclusive 21.5712 s, median paired CPU/Naja is
1.165x (ratio of medians 1.196x). The earlier packed-group 1.650x is superseded
for a best-tested CPU claim. Different concurrent load/socket placement means
this comparison alone does not isolate affinity as the sole causal change.
Per-seed topology, reservation counts, exact module/sample hashes and fresh
metrics are in benchmarks/results/native_cpu/iJO1366/physical16_thin2000/summary.json.
The CPU group uses 10,000 retained draws/chain and Naja 50,000; both meet the
declared median criterion. Frozen canonical tables remain unchanged.


Completed physical-core wheel/native follow-ups also cover RBC (250 retained
per chain, thinning 2,817) and iJR904 (500, thinning 8,971), with 1,000 discarded
warmup draws. Each group has all three seeds, fresh schema2 verified-mask
metrics, 32-logical/16-physical affinity proof, and SHA256 equality against
same-build/same-seed earlier outputs. RBC wheel/native median process walls are
8.81473/6.72557 s; iJR904 wheel/native are 11.65039/10.06766 s. Relative to
Naja TPB128, median paired best-tested CPU/Naja ratios are 3.632x for RBC and
1.967x for iJR904. These supersede earlier packed-worker quality-search ratios
for best-tested CPU claims. Per-build groups remain separate in
results/native_cpu/{iAB_RBC_283,iJR904}/physical16_{wheel,native}_jade/summary.json
(relative to benchmarks); the plot shows a separate best physical-core series.
Affinity improvements are measured under shared-node load, so differences
alone do not identify a sole causal performance factor.

A final canonical-family diagnostic sweep completed all four remaining stale
schema1/provisional records with array412, preserving prior files. The fresh
sweep found no remaining stale canonical metrics. Exploration remains separate.


## GPU thinning follow-up: selected budgets (2026-11-08)

The completed iJO1366 sweep uses frozen binary c7baf4, 16 chains and TPB512.
Thinning 1,000 selected 2,000 retained draws/chain; three fresh exact final
processes measured 9.32797, 9.46715 and 9.72363 s including normalization.
All three fresh schema2 verified-mask diagnostics pass median Rhat ≤ 1.01 and
bulk ESS ≈ 2,000, with zero bound fraction at 1e-7. Against the physical-core
native CPU median 25.8015 s (16 physical cores, 32 logical CPUs reserved),
ratio of medians is 2.725368x; median same-seed CPU/GPU is 2.746645x.
CPU retained 10,000 draws/chain at thinning 2,000. Counts and warmup move counts
differ, so this is a selected-budget comparison. Pilot/search/diagnostic costs
are excluded and retained separately. Thin500 passes at 5,000 draws (median
9.90484 s); thin200 did not pass within its pilot cap. The earlier auto96 GPU
21.5712 s result remains historical and no longer supports a best-tested claim.
Tuning used the same three seeds; fresh nonoverlapping seeds are required for
the separate 512-chain study. No result from that study is assumed.
Evidence: benchmarks/results/gpu_thinning/iJO1366/summary.json.
Interpretation: benchmarks/results/BEST_USE_CASE.md. Kernel throughput is
reported separately from CPU end-to-end comparisons.
