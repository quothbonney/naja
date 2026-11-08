# Audited iJO1366: one H100 versus 64 physical CPU cores

Updated 2026-11-08T19:42:28.343124+00:00. Audit status: **complete**.

512 chains per method; CPU seeds 101001, 202001, 303001 have disjoint cross-replicate chain-seed ranges; Naja uses those distinct base seeds with per-chain CURAND subsequences. CPU: 12,000 retained + 1,000 warmup draws/chain, thinning 2,000. Naja: 5,000 retained + 1,000 warmup, thinning 500, rotation, TPB 128, prefault 8. Budgets were selected from first-seed pilot evidence, followed by distinct seeds 202001/303001. Both methods target the same cached 581-D polytope and original reaction bounds.

Lead CPU time is the maximum accumulated C++ draw-call interval over 64 workers. It includes binding/state-copy and descheduling, and excludes pool/IPC/output. Lead GPU time is the native sampler phase, including allocation/slack/RNG setup and excluding download/write. Neither is labeled pure arithmetic-only. Diagnostics and preprocessing are outside both timers.

## cached_target_mask478

All six quality/bounds checks pass: **True**. Median paired GPU/CPU effective-sampling rate: **14.473×**; CPU/GPU compute ratio of medians: **11.896×**.

| Method | Seed | Retained/chain | Compute s | Median split-Rhat | Median bulk ESS | ESS/compute s | Bounds fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | 101001 | 12,000 | 153.7519 | 1.008308 | 88813.0 | 577.64 | 0 |
| cpu | 202001 | 12,000 | 154.6579 | 1.008410 | 88461.6 | 571.98 | 0 |
| cpu | 303001 | 12,000 | 154.6950 | 1.008374 | 89226.7 | 576.79 | 0 |
| gpu_coordinate | 101001 | 5,000 | 10.0846 | 1.006102 | 109493.5 | 10857.50 | 0 |
| gpu_coordinate | 202001 | 5,000 | 13.0980 | 1.006321 | 107839.8 | 8233.30 | 0 |
| gpu_coordinate | 303001 | 5,000 | 13.0012 | 1.006166 | 108534.6 | 8348.04 | 0 |

The legacy878-mask row is a historical assay using original-model LP certificates. The conservative478-mask reassay certifies exclusions directly on cachedA/b; unresolved reactions stay active. Exactly constant sampled reactions are excluded by the unchanged frozen estimator. All-reaction bound checks are unmasked in both assays.

## legacy_raw_model_mask878

All six quality/bounds checks pass: **True**. Median paired GPU/CPU effective-sampling rate: **11.463×**; CPU/GPU compute ratio of medians: **11.896×**.

| Method | Seed | Retained/chain | Compute s | Median split-Rhat | Median bulk ESS | ESS/compute s | Bounds fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | 101001 | 12,000 | 153.7519 | 1.009368 | 80204.4 | 521.65 | 0 |
| cpu | 202001 | 12,000 | 154.6579 | 1.009398 | 80917.3 | 523.20 | 0 |
| cpu | 303001 | 12,000 | 154.6950 | 1.009377 | 80308.9 | 519.14 | 0 |
| gpu_coordinate | 101001 | 5,000 | 10.0846 | 1.008541 | 76563.3 | 7592.10 | 0 |
| gpu_coordinate | 202001 | 5,000 | 13.0980 | 1.008229 | 78554.7 | 5997.46 | 0 |
| gpu_coordinate | 303001 | 5,000 | 13.0012 | 1.008398 | 77338.8 | 5948.59 | 0 |

The legacy878-mask row is a historical assay using original-model LP certificates. The conservative478-mask reassay certifies exclusions directly on cachedA/b; unresolved reactions stay active. Exactly constant sampled reactions are excluded by the unchanged frozen estimator. All-reaction bound checks are unmasked in both assays.

## Interpretation and verification

- This is effective throughput at 512 chains and a shared quality threshold, not globally minimal latency across chain counts. Achieved median pooledESS differs (~89kCPU/~109kGPU under the conservative assay); ESS/s accounts for that difference. Budgets were selected under the historical878 assay and have not been re-minimized under478. CPU transitions total 13.312 billion versus GPU 1.536 billion, including warmup. Algorithm/basis/thinning efficiency contributes to the result; it is not a same-transition kernel acceleration claim.
- The native64-core CPU is the fastest verified core configuration in this bulk study. The finite thinning/basis search does not establish a globally optimal CPU sampler.
- All64cores and sibling reservations,512chain worker logs, accumulated timers, estimator/geometry/bounds/mask identities and guarded diagnostics reuse are audited by the source script.
- CPU seed101001 reuses48-core diagnostics only after whole retained-file byte identity. CPU seeds202001/303001 have independent full diagnostics. GPU first-seed diagnostics similarly require full-file identity. Different replicate seeds are independent; same-seed retimes are not additional independent observations.
- Original16-chain fixed configurations passCPU2/3 andGPU1/3 fresh seeds. Separately passing bounded extensions do not rescue fixed-budget generalization or a universal2.73× claim.

The conservative478 assay reaggregates full-budget per-reaction values produced by the unchanged frozen estimator. A fresh512-chain rerun matches all2,583 per-reaction rows and aggregate/feasibility summaries exactly. Only full-budget endpoints are reused; lower-budget prefix statistics are not. [Fresh exact-dataset validation](cached_target_audit/reaggregation_validation.json).

[Machine-readable audit and all sources](comparison.json) · [Per-seed CSV](comparison.csv).

Regenerate from workspace root:

```bash
benchmarks/env-cpu/bin/python benchmarks/scripts/audit_full64_comparison.py
```

Optional full retained-file SHA256 recheck:

```bash
benchmarks/env-cpu/bin/python benchmarks/scripts/audit_full64_comparison.py --verify-array-hashes
```

## Why the bulk configuration has a larger advantage

The separately frozen 16-chain validation passes all three new seeds per method and yields 170.06 CPU versus 403.47 GPU median reaction ESS/s. The 512-chain configurations yield 576.79 versus 8348.04: CPU effective throughput rises 3.39× while GPU throughput rises 20.69×. CPU workers increase from 16 to 64 physical cores. These are comparisons of independently selected configurations at a common quality threshold, not isolated chain-count scaling: retained budgets, GPU thinning, TPB, binary and prefault settings also differ. [New fixed 16-chain validation](../../fresh16_conservative/iJO1366/VALIDATION.md).

Separate Nsight throughput profiling measures mean SM-active of 11.71% at 16 chains and 97.22% at 512 chains. That protocol uses automatic thinning 96, TPB128 and 10,000 retained draws; it qualitatively supports the occupancy explanation but is not a convergence experiment or a quantitative multiplier for these quality-matched measurements. [Profiling methods and counters](../../nsight_timeline/README.md).
