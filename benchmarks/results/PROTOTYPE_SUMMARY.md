# Prototype evidence summary

Snapshot 2026-11-08T03:32:38+00:00. Campaign remains active; this is a saved-artifact summary.

Four models have completed 3-seed measured passing-budget groups. Criterion: model-mask median split-Rhat ≤ 1.01 and pooled bulk ESS ≥ 100. Every time is an actual process execution including load/startup, 1,000 retained warmup draws/chain and writes; GPU times include normalization. Budget search and diagnostics are excluded. CPU choice uses the fastest completed passing tested variant separately per seed, including native builds. Completed physical-core controls are included for RBC, iJR904 and iJO1366; core uses the quality-search records. These have different explicit CPU allocations. This finite search does not establish global optimality.

| Model | Seeds | GPU variant | GPU median s | Best completed tested CPU median s | Median paired CPU/GPU | Native CPU group median s |
|---|---:|---|---:|---:|---:|---:|
| e_coli_core | 3/3 | naja128 | 1.307 | 3.325 | 2.564× | 3.744 (cpu_nativejade1) |
| iAB_RBC_283 | 3/3 | naja128 | 1.862 | 6.726 | 3.632× | 6.726 (physical16_native_jade) |
| iJR904 | 3/3 | naja128 | 5.117 | 10.07 | 1.967× | 10.07 (physical16_native_jade) |
| iJO1366 | 3/3 | naja512_thin1000 | 9.467 | 25.8 | 2.747× | 25.8 (native_physical16_thin2000) |

The iJO1366 physical-core control completed all three seeds, reserving 32
logical CPUs and pinning 16 workers to 16 unique physical cores. CPU process
walls are 25.8015, 26.0029 and 24.3063 s; all three median Rhats pass 1.01,
median bulk ESS exceeds 2,151 and bound fractions are zero at 1e-7. Each saved
array is SHA256-identical to its earlier same-seed native counterpart. With the new GPU thinning 1,000 selected budget, median paired CPU/Naja is **2.747×** and ratio of medians is **2.725×**. The historical automatic-thinning GPU comparison gave median paired 1.165×. The older
packed-worker comparison was 1.650× and is superseded for the best-tested CPU
claim. Different node load/socket placement prevents assigning the entire
improvement causally to affinity. [Three-seed placement, hashes and quality evidence](native_cpu/iJO1366/physical16_thin2000/summary.json).

For iJO1366, `naja512` denotes TPB512 with 16 chains, not 512 chains. The historical automatic-thinning configuration used 50,000 retained draws/chain; the new thinning 1,000 configuration uses 2,000; the completed physical-core native CPU group uses 10,000. Core and iJR904 Naja use median budgets 500 and 20,000; RBC uses 5,000. CPU thinning and selected budgets differ and are preserved per seed. Native builds are separately identified; only the controlled source-generic/native pilot isolates compiler flags. Native builds did not uniformly beat the wheel.

[Quality records, budgets, tested variants and per-seed ratios](quality/summary.json) · [fixed-budget full model table](BENCHMARK_TABLE.md) · [CSV provenance](all_runs.csv). Bound compliance is independent of the convergence criterion: the iJR904 20 k quality evidence includes one violating sample in 320,000 at 1e-7. Inspect every row’s bounds field; no zero-violation blanket claim is made.

Eight of nine panel models have saved accepted geometry and canonical sampling coverage: e_coli_core, iAB_RBC_283, iJR904, iAF1260, iJO1366, iML1515, iMM904 and RECON1. This does not mean eight completed equal-quality comparisons. RECON1 is explicitly precision-limited; hopsy jobs and diagnostics remain incomplete. Recon3D is the ninth model and remains preprocessing work, with failed/long-running attempts retained. [Preparation attempt evidence](setup/RECON1_preprocessing_attempts.json) · [Recon3D attempts](setup/Recon3D_preprocessing_attempts.json).

RECON1 seed 303 now has a causal same-trajectory export test: float 32 quantization produces maximum varying-reaction error 0.01650784; float 64 removes that error while retaining maximum 2.70913 e-7 on LP-excluded nearconstant reactions. The historical float 32 trajectory is reproduced bitwise. Unfiltered any-reaction violations at 1e-7 remain 0.59409375 even in float 64. [Paired precision evidence](precision/RECON1/seed303/summary.json) · [build hashes and scope](HARDWARE.md). This resolves the large export errors on this trajectory, not every model/seed or strict-bound feasibility.

The optional host first-touch experiment has three paired iJO1366 512-chain seed 101 repetitions, identical retained-array hashes, and median paired 1.232× speedup including normalization (1.471× native process; 4.394× download including allocation/thread/touch/copy costs). [Full paired evidence](host_prefault/iJO1366/summary.json). This uses a separate build, not canonical timings. The earlier frozen-binary 512-chain attempts overflowed and remain explicitly invalid; the separate wide-index fix and validated throughput probes are reported independently.

All canonical tables require frozen binary SHA256 `c7baf4f0ed282b936ac0dbac4b896bfbfc040eb06da6f84aff6605fddf4b0fc6` or the recorded baseline wheel. Native CPU, float 64, wide-index and first-touch builds cannot enter canonical rows. All four stale canonical diagnostic records were recomputed by array 412; a fresh canonical-family sweep found no schema 1/provisional mask records. Old metric snapshots remain preserved.

Limits: shared-node timing load; one H100; common starting points rather than overdispersed initialization; median convergence criterion rather than a uniformity proof; empirical long-reference agreement is only on e_coli_core; one-second telemetry can miss peaks; no all-nine completion, historical performance reproduction, production-corpus, MATLAB, general scaling or exact-zero claim. [Independent review](../audit/final_review.md) retains failure cases and protocol checks.

RBC/iJR904 wheel and native physical-core controls are separate validated selected-budget groups, with 32 logical CPUs reserved and 16 unique physical cores used. Completed groups enter the best-tested CPU choice; partial groups do not. [Physical-core control artifacts](native_cpu/) retain original per-seed hashes, masks and bounds checks.

The completed iJO1366 GPU thinning sweep uses frozen c 7 baf 4 output, TPB512,
16 chains and fresh exact final executions. Thinning 1,000 selects 2,000 retained
draws/chain at 9.46715 s median inclusive wall; every seed passes Rhat ≤ 1.01 and
bulk ESS ≈ 2,000, with zero bound fraction at 1e-7. The physical-core CPU uses
10,000 retained draws/chain at thinning 2,000. This is a selected-budget quality
comparison; sample counts and warmup move counts differ. Pilot/search/diagnostic
costs are separately recorded and excluded. Thin 500 also passes (median 9.90484 s,
5,000 draws); thin 200 reaches its pilot cap without passing. These same tuning
seeds do not establish a global optimum; the separate 512-chain study needs
fresh nonoverlapping seeds and its result remains pending.
[GPU thinning evidence](gpu_thinning/iJO1366/summary.json) · [When the GPU is useful](BEST_USE_CASE.md).
