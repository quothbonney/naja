# Exact-knockout conditioned benchmark audit

Updated 2026-11-08T19:40:37.646534+00:00.

16 chains, seed101, CPU thinning2000 / Naja1000, warmup1000 retained draws, median classical splitRhat<=1.05, pooled bulkESS>=100, exact bounds/mass balance1e-7. Fresh selected-budget retimes only.

50 nontrivial unique exact gene knockouts; b0002 control kept separate. Validity-selected rank-stable panel, not fully stratified or population representative.

| Pipeline | Pass /50 | Capped at50k | Unavailable | Median sampler s among passes |
|---|---:|---:|---:|---:|
| global_cpu | 49/50 | 0 | 1 | 3.9426 |
| stale_cpu | 32/50 | 18 | 0 | 3.9013 |
| naja_rotate | 41/50 | 9 | 0 | 3.9873 |

These per-method timing medians use different passing subsets and must not be divided into a speedup. Paired comparisons in the CSV use only targets passing both methods. CPU draw-call critical path excludes startup/IPC/output; GPU sampler includes allocation/slack/RNG and excludes download/write. Setup and target geometry are separate.

| Paired reference | Common passing targets | Median CPU/GPU sampler-time ratio | Median GPU/CPU ESS-rate ratio |
|---|---:|---:|---:|
| global_cpu | 41 | 1.0168 | 0.8418 |
| stale_cpu | 28 | 1.1528 | 0.8922 |

## Reused geometry: preparation plus sampling

On the **41/50** nontrivial targets passing both methods, the median paired ratio (full-global MVE algorithm time + CPU draw critical path) / (Naja correction + GPU sampler) is **14.06×**. The paired range is 0.439–19.18×; Naja is slower on at least one case. Adding the common validated conditional-geometry preparation to both sides gives a median **9.87×**. Global-MVE time / Naja-correction time alone has median 365.7×.

The first ratio starts from the same ready shared conditional geometry; it includes mathematical rerounding/correction and sampling, excludes output transfer/writes, external validation, LP-mask compilation, imports, and one-time base preparation. It is a **preparation-reuse result**, not a sampling-kernel speedup. The second includes shared per-condition preparation, whose wall time also includes file assembly. All per-case components are saved in paired_comparisons.csv. Nine rotate failures and the unavailable global reference remain in the full denominator; passing-subset ratios cannot stand in for success on all50 targets.

Independently recomputed all-halfspace and reaction-map equivalence for 50 full-reround targets; common reaction masks match. The b0031 full-reround reference is unavailable because PolyRound reported 'Rounding transformation not full dimensional'; the valid shared target and this failure remain preserved.

Inherited CPU uses the mandatory orthonormal exact-KO affine-hull projection. It can already repair mixing. The finite-width PGI capacity example has a different geometry and should be described separately. The unchanged b0002 control is excluded from all aggregate performance rows. A single seed per KO does not establish replicate reliability.

[Per-target data](per_condition.csv) · [Paired comparisons](paired_comparisons.csv) · [Full proof/status audit](summary.json) · [Scheduler outcomes](slurm-accounting.psv).

## Fastest tested valid CPU and success contingency

Among50 nontrivial targets: **28** pass both inherited CPU and rotate; **13** pass rotate only; **4** pass inherited CPU only; **5** pass neither. Correction is not uniformly beneficial.

On the 41 rotate-passing targets, selecting the faster **passing** CPU option per target (inherited CPU, or full global MVE + CPU when needed), the median paired preparation-plus-sampling CPU/Naja ratio is **1.115×**, or **1.095×** with common conditional-geometry preparation included. This finite-grid selection uses observed quality; it is not an independently validated automatic selector. Thus the full-reround-only14× result must not be described as a speedup over the fastest available CPU pipeline. [Per-case components](fastest_cpu_comparisons.csv).

## Separate scaled-coordinate recovery of the last CPU reference

All50 nontrivial conditions now have a valid full-PolyRound CPU reference:49 default successes plus this separately labeled fallback. The original b0031 default failure and all original table rows remain preserved. Uniform initial coordinate scaling by1e-6 avoids the absolute PolyRound dimensionality gate; default PolyRound thresholds remain unchanged. The composed invertible square map, all original halfspaces, reaction map, common conservative mask and fresh selected samples were independently checked, with no target relaxation or second hull reduction.

| Gene | Initial units scale | Global MVE s | Selected draws | CPU sampler s | Rhat | ESS |
|---|---:|---:|---:|---:|---:|---:|
| b0031 | 1e-6 | 64.405 | 2000 | 3.9767 | 1.03725 | 529.7 |

[Recovery source/proof/commands](../../conditioned_extensions/b0031_scaled_global/) · [Audited fallback row](separate_fallback_references.csv). This does not change the41 original rotate-passing paired subset, because b0031 rotate failed the original50k cap.
