# Recorded benchmark environment

Sampling runs use the Slurm H100 node, with one NVIDIA H100 80GB HBM3 per Naja
job and NVIDIA driver 580.126.20. The node has two Intel Xeon Platinum 8462Y+
sockets, 64 physical cores / 128 logical CPUs, and approximately 2 TiB RAM.
Sixteen-chain CPU runs use 16 multiprocessing workers in 16-CPU allocations;
explicitly labeled tuning runs use one worker. Independent experiments run
concurrently, so these are prototype measurements under shared node load.

Naja is built from Git commit
`23873f01d3268a79bbf84499a66adce5e88cc6cb` plus the saved `source.patch`, with
CUDA toolkit 12.9.86, GNU 11.4.0, C++17, Release/O3, architecture `sm_90`, and
`--use_fast_math`. The sampling binary SHA256 is
`c7baf4f0ed282b936ac0dbac4b896bfbfc040eb06da6f84aff6605fddf4b0fc6`.
The patch adds explicit sampling seeds and applies production fast-math flags
to the five GPU correctness tests. All five pass on H100 with those flags.
The driver reports CUDA compatibility 13.0; that is distinct from the installed
12.9 toolkit. Gurobi 13.0.2 is linked, but license-dependent LP starts are not
used: both samplers receive the same precomputed feasible point.

CPU environment: CPython 3.12.15, hopsy 1.7.0, PolyRound 0.5.0, COBRApy 0.32.1,
NumPy 2.5.4, SciPy 1.18.1, ArviZ 1.3.0 and swiglpk 5.0.13. The full exact
environment is saved in `cpu-requirements.lock.txt`; the GPU Conda explicit
environment is `gpu-environment.explicit.txt`. Optional HiGHS/OSQP packages
were added for documented preprocessing retries, without upgrading the initial
packages; the original CPU lock is retained separately.

The initial e_coli_core preprocessing run used the controller's Intel Xeon
Platinum 8562Y+ CPU; the other canonical preprocessing runs used the H100 node.
Each model's metadata records its actual hostname, backend, settings and phase
times. MATLAB was not found in the compute-node PATH; hopsy is the CPU baseline.

Raw hardware commands and final correctness output are in `setup/`; exact
commands, actual seeds, binary hashes and allocations accompany individual runs.

## Additional controlled and experimental builds

The canonical wheel-based CPU and frozen Naja results above remain unchanged.
The isolated native CPU pilot uses the same official hopsy 1.7.0 SDIST for a
source-generic build (`-O3`) and a native build
(`-O3 -march=native -mtune=native`), both GCC 11.4.0 on the Xeon 8462Y+ node.
Both source builds report hopsy build `9b787d9`; the installed wheel reports
`dadab7b`. The controlled compiler comparison is therefore source_generic versus
native, not wheel versus native. The SDIST SHA256 is
`24659f594c6859823d887adc576f11667abcd2548f8e22341c313841e809fc1e`.
Exact flags, compile commands, wheel/module hashes, environment signatures and
analytic cube checks are saved in `../audit/hopsy_native_build.json` and
`../audit/hopsy_source_generic_build.json`. Dedicated environment locks preserve
the generic-source and native environments; baseline env-cpu was not replaced.
The one-seed pilot is `native_cpu/pilot_summary.json`, separate from canonical
sampling and any incomplete three-seed quality groups.

A separate reduced-output wide-index Naja build casts allocation/copy and
matrix-offset products to size_t before multiplication. Its SHA256 is
`da263d6060416d96da366392210b9468422890bda8200718da452941436a8f18`;
it does not replace the frozen canonical `c7baf4...` binary. All five GPU
correctness tests passed, and same-seed synthetic BOX output matched the frozen
binary bitwise. The build proof and scope limits are in
`safe-build-verification/wide-index-proof.json`. Its valid 512-chain exploratory
results live only in `scaling-extension-safe/`; the original 512-chain overflow
attempts remain invalid and are excluded. Zero process exit status alone is
insufficient: GPU backend logs and output finiteness/shape are now guarded.

## RECON1 preprocessing precision

RECON1 (3,741 reactions; d=931; 2,120 inequalities) used GLPK with presolve on
the same H100-node CPU. Preprocessing took 1,549.53 seconds. Its raw bounds and
saved geometry are unchanged. The strict 1e-7 verification failed narrowly
(center bound residual 1.14054e-7, pilot bound residual 1.52823e-7), and the
explicitly declared preprocessing verification gate is 1e-6. The strict failure
is preserved beside validation.json; this model is labeled precision-limited.
Common sample-bound diagnostics still use 1e-7 and exact zero tolerance over
all reactions. The preprocessing gate is not a target relaxation and must not
be described as strict zero-bound validation.

RECON1's preprocessing gate does not guarantee output accuracy. Current frozen
float32 reduced-output Naja samples have maximum original-reaction bound errors
of approximately 0.00084, 0.00177 and 0.0165 across the three seeds. These are
larger than the small center/pilot residuals and also affect varying reactions.
The approximately 0.59 any-reaction sample violation fraction at 1e-7 is largely
associated with nearly blocked rows, with coordinate fractions near 0.0002.
The paired experiment below isolates export quantization for seed303. No exact-bound or zero-violation claim is
supported by this preparation gate or by passing convergence diagnostics.


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
1e-7 feasibility. See [paired evidence](precision/RECON1/seed303/summary.json).
The earlier statement that export precision was only a hypothesis is superseded.

The opt-in float64-export/wide-index binary SHA256 is
`5e6968eb7d3cebf5f27c16d2662a2e640cc3f6b7a633b7e7ae3bc9136a99f3d5`;
complete source patch SHA256 is
`33f9f14beb50cb2aa5b7aa37c61f2868f04b15cd66461f0537751a1fb3aa5d2a`.
[Build proof](f64-build-verification/verification.json) records five GPU tests,
unchanged sampling math, historical/default float32 bitwise equality, and
float64 dtype/cast checks. It does not replace canonical float32 results.

The optional first-touch binary SHA256 is
`a656a1f93489af051acd540d08369092567dd0673af2b0e1ac922dc58a910603`;
complete source patch SHA256 is
`d891970c48c1c733243ec85812b420cf17f9ddb7349f201f770edcbef09bc568`.
[Build proof](prefault-build-verification/verification.json) records five GPU
tests and bitwise output comparisons. Its small verification fixture alone did
not test large arrays; the later [512-chain iJO1366 paired experiment](host_prefault/iJO1366/summary.json)
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
