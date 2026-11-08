# Paper update: results completed since the overnight handoff

Snapshot: 2026-11-08T19:35:40.631011+00:00. The three-hour follow-up is still running.
This file separates completed, audited findings from pending extensions. The
[full live report](RESULTS_SINCE_UTILIZATION_REQUEST.md) contains additional detail.

## 1. Main completed sampling result: 512 chains on one H100

| iJO1366, three seeds | Native hopsy CPU | Naja H100 | Comparison |
|---|---:|---:|---:|
| Sampler compute proxy, median s | 154.658 | 13.001 | **11.90×** CPU/GPU time |
| Median reaction bulk ESS | 88,813 | 108,535 | Same quality criterion |
| Median reaction ESS / compute s | 576.79 | 8,348.04 | **14.47×** median paired rate gain |
| Quality + strict bounds | 3/3 | 3/3 | Both pass all seeds |

Suggested Results text:

> On iJO1366 with 512 parallel chains, Naja on one H100 required a median
> sampler time of 13.00 s, compared with 154.66 s of native hopsy worker
> compute on 64 physical CPU cores, a ratio of 11.90. Both methods passed
> median split-Rhat ≤1.01, median pooled bulk ESS ≥100 and zero observed
> reaction-bound violations at a tolerance of 1e-7 in all three replicates.
> Naja achieved a median paired effective-sampling-rate gain of 14.47×.

Methods needed alongside that paragraph:

- CPU reserves all 128 logical CPUs and uses 64 native workers, one per physical
  core. Both methods use 512 chains and discard 1,000 retained warmup draws per
  chain. CPU retains 12,000 draws at thinning 2,000; Naja retains 5,000 at thinning
  500, with barrier rotation, TPB128 and host prefault8.
- CPU time is the worker draw-call critical path, including binding/state-copy
  cost; GPU time includes sampler allocation/slack/RNG initialization. Both exclude
  backmap/download/output. Timing scope details are in the source table.
- The conservative assay diagnoses 2,105 reactions, excluding only 478 numerically
  certified nearconstant reactions of the cached target. The older original-model
  assay diagnoses 1,705 reactions: it yields the same 11.90× time ratio and an
  11.46× ESS-rate ratio. Report this sensitivity; retain the original metrics.
- Budgets were selected under the earlier assay. This is fixed-512-chain sampling
  efficiency at a shared quality threshold, with differing retained counts and ESS.
  A globally minimized time-to-target over all chain counts has not been established.
- Transitions including warmup are 13.312 billion CPU versus 1.536 billion GPU.
  The overall gain combines sampling efficiency and hardware throughput.

[Audited standalone table](high_throughput/iJO1366/AUDITED_COMPARISON.md) ·
[Per-seed CSV](high_throughput/iJO1366/comparison.csv) ·
[All proofs and source hashes](high_throughput/iJO1366/comparison.json) ·
[Fresh 512-chain diagnostic/reaggregation agreement](high_throughput/iJO1366/cached_target_audit/reaggregation_validation.json).

## 2. Central conditioned benchmark: 50 exact gene knockouts

| Pipeline | Pass / 50 | Capped at 50k draws | Unavailable default reference |
|---|---:|---:|---:|
| Full global reround + CPU | 49 | 0 | 1 |
| Inherited rounding + CPU | 32 | 18 | 0 |
| Shared geometry + Naja rotation | 41 | 9 | 0 |

One seed per target, 16 chains, median split-Rhat ≤1.05, bulk ESS ≥100,
all reaction bounds and mass balance checked at 1e-7. These are 50 unique,
nontrivial, validity-selected exact gene knockouts; the unchanged control is
excluded. The panel is not fully stratified or a random population sample.

Suggested Results text:

> Across 50 exact gene knockouts, shared geometry with barrier rotation
> reached the specified convergence target on 41 conditions, compared with
> 32 for inherited CPU sampling and 49 for default full rerounding. The Naja
> pipeline reached the target on 13 conditions where the inherited CPU pipeline
> failed the tested cap; the inherited CPU pipeline reached it on four conditions
> where the Naja pipeline failed. On the 41
> conditions passing both Naja and the full-rerounding reference, the median
> paired preparation-plus-sampling ratio was 14.06×, or 9.87× when common
> conditional-geometry preparation was included.

Comparison scopes:

- The 14.06× ratio includes CPU global MVE + CPU sampler versus Naja correction +
  Naja sampler, excluding output, imports, masks and one-time base preparation.
- Sampler-only ratio against the full-reround CPU reference is 1.017×.
- Allowing the fastest tested passing CPU pipeline per target, including inherited
  CPU where it passes, gives **1.115×** for preparation plus sampling and **1.095×**
  with common preparation. The 14.06× figure specifically compares with full rerounding.
- Pass contingency: 28 both rotate/inherited, 13 rotate only, four inherited only,
  five neither. Nine rotation caps remain visible in the full denominator.
- Shared cached conditional and full-rerounded targets have independently checked square invertible
  maps, all-halfspace equivalence, common reaction maps and cached-target masks.
  The unavailable default b0031 reround and a rescaled-coordinate recovery are
  recorded separately. That recovery has now passed its full equivalence, quality
  and strict-feasibility checks: 49 default successes plus one recovered valid
  full-CPU reference cover all 50 conditions. The default-failure row stays preserved.

[Complete conditioned audit](knockout_panel/overnight_audit/CONDITIONED_BENCHMARK.md) ·
[Per-condition CSV](knockout_panel/overnight_audit/per_condition.csv) ·
[Paired timing components](knockout_panel/overnight_audit/paired_comparisons.csv) ·
[Figure SVG](knockout_panel/overnight_audit/conditioned_panel_summary.svg) ·
[PDF](knockout_panel/overnight_audit/conditioned_panel_summary.pdf).

## 3. Repeated barrier-phase result

Three independent process timing repeats reproduce the CPU-rotation bottleneck.
On the explicitly exploratory equality-only Recon3D geometry, the original
barrier phase has median **94.618 s**, versus **1.4359 s** with GPU constraint
rotation: **65.89×**. Short feasibility checks pass; these are preparation timings,
not convergence experiments. The original single-pair result was 63.73×.

The nine-model dimension figure also shows historical PolyRound MVE costs as
context. MVE and local barrier correction are different operations; these phase
costs do not establish equivalent conditioning quality. Full Recon3D PolyRound
preprocessing timed out at nine hours, so a full-model matched-quality Recon3D
sampling speedup remains unavailable.

[Three-repeat table and methods](barrier_dimension_scaling/README.md) ·
[All-repeat CSV](barrier_dimension_scaling/source.csv) ·
[SVG](barrier_dimension_scaling/barrier_dimension_scaling.svg) ·
[PDF](barrier_dimension_scaling/barrier_dimension_scaling.pdf).

## 4. Updated cross-model CPU baselines

Under the separate conservative cached-target assay, iML1515 CPU at thinning
2,000 and 12,000 retained draws passes all three seeds: worker median **20.8185 s**,
versus Naja **8.1171 s**, a **2.5648×** time ratio and **3.2255×** paired ESS-rate gain.
The 10,000-draw CPU configuration fails the cached-assay Rhat check in all three
seeds. The earlier **8.51×** result compares against guideline-thinning CPU;
it should remain labeled as that baseline rather than the fastest tested CPU.

For iAF1260, the current cached-assay comparison is CPU **14.9873 s** versus
GPU **15.4979 s**: time ratio **0.9671×**, paired ESS-rate gain **3.6356×**.
These retained budgets deliver different ESS. Smaller-budget screens and
back-to-back GPU block-size follow-ups are running; use their final tables
before treating current configurations as optimized.

[Current cached-assay model table](crossmodel_extension/CACHED_TARGET_RESULTS.md) ·
[Machine-readable comparisons](crossmodel_extension/cached_assay_summary.json).

Retaining unresolved nearconstant reactions is conservative for LP-based exclusion;
it does not necessarily make a median-Rhat criterion harder. Such reactions can
lower the reaction median. New budget follow-ups therefore report both assays;
assay-specific passing budgets should retain their diagnostic-mask labels.

## 5. Still running; not paper claims yet

- Same-GPU off / rotate / whiten on nine previously capped conditions, one
  additional high-condition-number case and the capacity control.
- 200,000-draw rotation extensions for all nine capped conditions.
- Fixed-pair and native Jacobi/Dikin mixture follow-ups on predefined hard cases.
- Stronger cross-model budget checks and targeted same-GPU rescue-case controls.

[Hardware/software paragraph and timer definitions](continuation_20261108/ENVIRONMENT_AND_PROTOCOL.md).
All frozen canonical reports and metrics remain unchanged.

## Follow-up at 19:38 UTC: block-size control

A same-physical-GPU order-balanced check (512/128/128/512 threads per block)
on iML1515 with 16 chains gives sampler medians 8.1363 s at TPB512 and
13.0945 s at TPB128. All four complete F64 retained arrays are bit-identical
to the verified reference. Thus TPB128 is 1.61× slower for this small-chain
configuration; the favorable TPB128 choice in the 512-chain study should
remain specific to that workload. The historical 8.1171 s TPB512 timing is
reproduced by the new control.

[Order-balanced commands, timing and parity proof](crossmodel_extension/tpb_paired_control/summary.json).

## Completed follow-up: held-out 16-chain validation

A new protocol frozen before its three new seeds (444001,555001,666001)
passes all three CPU and all three GPU replicates under the conservative
478-exclusion cached-target assay. CPU retains 15,000 draws at thinning 2,000;
GPU retains 3,000 at thinning 1,000 with TPB512; both discard 1,000 retained
warmup draws per chain. Median CPU worker compute is 21.0998 s and GPU sampler
time 10.3878 s. Median paired gains are **2.031× time** and **2.427× ESS/s**.

This independently validates the larger fixed budgets. The original 10k CPU / 2k GPU
fixed-budget failures and their separately selected extensions remain visible;
the new protocol does not validate the old 2.73× estimate. The much larger 512-chain
effective-throughput advantage supports using enough chains to occupy the H100.

[Held-out validation](fresh16_conservative/iJO1366/VALIDATION.md) ·
[Per-seed table](fresh16_conservative/iJO1366/table.csv) ·
[Frozen protocol and all results](fresh16_conservative/iJO1366/summary.json).

## Remaining model limitation: iMM904

The original CPU and GPU runs share small reaction-bound violations around
2e-7 on the cached yeast geometry. A separate tighter PolyRound preparation
(threshold1e-9, LP feasibility1e-10, accepted tolerance violation1) failed
its MVE distortion convergence check. That attempt and its error are preserved;
there is no corrected strict-bound iMM904 headline row. Retain its explicit
numeric-tolerance limitation in a broad model table.

[Separate strict-preparation attempt](crossmodel_extension/iMM904_strict/preparation.json).

## Dual-assay cross-model update

The completed original-model assay selects iML1515 CPU at 15,000 retained
thinning-2,000 draws: 28.3382 s versus GPU 8.1171 s, giving **3.4912× time**
and **2.9119× paired ESS/s**. The cached-target assay permits CPU 12,000 draws
and gives the **2.5648× / 3.2255×** result reported above. Both CPU selections
have three passing seeds under their respective assays. Preserve these labels
when transferring the results to a paper table; the 12,000-draw CPU group
passes only two of three seeds under the original-model assay.

[Both assays, all groups and passing/failing budgets](crossmodel_extension/DUAL_ASSAY_RESULTS.md) ·
[Machine-readable dual-assay comparisons](crossmodel_extension/dual_assay_summary.json).
