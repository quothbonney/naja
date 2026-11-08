# Task3: iJO1366 rotation-OFF control

Production commit`fcc66fe63d5a6700786aebcf0b925374340e4c8b`, binarySHA256`5c01c66c90f46565eaafe0927265d135bcfa89af50a017df6c51de70044afe42`.

iJO1366, 16 chains, 1,000 warmup + 10,000 retained draws, thinning 96, TPB 128, barrier off, float32 export, host prefault 0; same cached polytope/start/legacy LP mask as historical table

The legacy LP mask excludes 878 reactions, matching the historical table. Geometry, start and mask SHA256s plus exact commands are saved per seed. This is separate from the 478-exclusion cached-target conservative assay.

| Seed | Median Rhat | Reaction p95 Rhat | Median bulk ESS | ESS /million actualupdates | GPU sampler s | Native total s |
|---|---:|---:|---:|---:|---:|---:|
| 101 | 1.193477 | 1.423806 | 75.04 | 4.441 | 4.0955 | 5.2429 |
| 202 | 1.178061 | 1.407040 | 81.68 | 4.834 | 4.0943 | 5.3232 |
| 303 | 1.186539 | 1.439281 | 79.40 | 4.700 | 4.1183 | 5.3696 |
| Median | 1.186539 | 1.423806 | 79.40 | 4.700 | 4.0955 | 5.3232 |

Actual updates = 16 × 11,000 × 96 = 16,896,000 including warmup. Retained-only updates = 15,360,000. ESS is the reaction median of pooled bulk ESS; the primary normalized value is ESS / 16.896. No estimated prefix wall times are used. Hopsy thinning specifies one coordinate update, not a full dimensional sweep; its cyclic scan differs from Naja random scan. [Sourceproof](sourceproof/THINNING_SEMANTICS.md).

[Per-seed components](task3_per_seed.csv) · [Full summary/provenance](task3_off_summary.json).
