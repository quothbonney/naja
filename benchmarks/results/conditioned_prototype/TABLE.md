# Conditional capacity study: saved progress

Primary measures separate geometry setup/correction from output-excluded sampling. Primary sampler column uses CPU MarkovChain.draw critical-path proxy (pybind/state-copy included, IPC excluded) or GPU sampling-phase proxy (device allocation/slack/RNG initialization included). These are explicitly scoped, not unqualified arithmetic-only timers. All final budgets are actual executions. Criterion: median Rhat <=1.05 and median bulk ESS >=100; conditioned/original bounds and mass balance independently require1e-7. A cap/failure is not a passing comparison.

| Pipeline | Seed | Status | Draws/chain | Geometry algorithm s | Local GPU correction s | Primary sampler proxy s | Median ESS / sampler s | Median Rhat | Bound fraction | Secondary inclusive process s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_cpu | 101 | invalid_comparator_affine_hull_mismatch | 10000 | 134.9 | — | 15.3987 | 136.009 | 1.00979 | 0 | 25.6006 |
| full_cpu | 202 | invalid_comparator_affine_hull_mismatch | 10000 | 134.9 | — | 15.9042 | 136.634 | 1.0101 | 0 | 26.1334 |
| full_cpu | 303 | invalid_comparator_affine_hull_mismatch | 10000 | 134.9 | — | 14.3351 | 146.133 | 1.00969 | 0 | 23.7346 |
| naja_off | 101 | no_quality_or_feasibility_within_cap | 50000 | 1.57466 | 0 | 132.353 | 0.502817 | 1.12873 | 0.0004525 | 139.609 |
| naja_rotate | 101 | passing | 10000 | 1.57466 | 0.1482 | 28.5765 | 287.518 | 1.00244 | 0 | 31.3827 |
| naja_rotate | 202 | passing | 10000 | 1.57466 | 0.1469 | 28.4658 | 299.083 | 1.00232 | 0 | 31.2531 |
| naja_rotate | 303 | passing | 10000 | 1.57466 | 0.1456 | 28.6253 | 295.504 | 1.0023 | 0 | 31.5032 |
| stale_cpu | 101 | no_quality_or_feasibility_within_cap | 50000 | 1.57466 | — | 70.9255 | 0.740969 | 1.14039 | 0 | 91.5082 |
| stale_cpu | 202 | no_quality_or_feasibility_within_cap | 50000 | 1.57466 | — | 73.6893 | 0.709832 | 1.14215 | 0 | 93.8813 |
| stale_cpu | 303 | no_quality_or_feasibility_within_cap | 50000 | 1.57466 | — | 65.1143 | 0.794304 | 1.14487 | 0 | 85.3753 |

Standalone preparation process and all output/adapter costs remain in JSON as secondary operational measures. One-time base preparation is separately recorded; diagnostic/search costs are excluded from selected-budget phase times and preserved. Only independently passing, bound-compliant rows support quality-speed claims. [Protocol](README.md) · [structured rows](summary.json).
