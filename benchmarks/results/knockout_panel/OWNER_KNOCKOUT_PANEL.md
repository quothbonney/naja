# Exact gene-knockout panel handover

The durable geometry controller completed its target independently of Codex. Its last
PID is in `controller.pid` (357987 at this handover), PPID1; state and exact
submitted commands are in `controller_state.json`, with `controller.log`.
`ready_manifest.json` is the ONLY sampling-ready interface. Do not consume every
directory or the historical `geometry_summary.json` as an eligibility list.

Snapshot 2026-11-08 05:11 UTC:51 unique validated targets,50 nontrivial KOs plus
one zero-effect GPR control. The geometry goal is achieved; this is not a claim
that sampling comparisons are complete. The final manifest is saved; the sampling consumer remains active. Initial five genuinely nontrivial stable
KOs were b0154,b0181,b0343,b0508,b4025. Later targets are accepted by geometry
validity/equivalence only, never convergence, speed or a favorable result.

## Durable execution and deadlines

The controller launches at most two4-logical-CPU,32GB Slurm jobs concurrently,
partition `all`, nice1000,10min per condition. New jobs stop at10:30UTC;
the controller exits by11:00UTC on2026-11-08. The campaign's deadline cleanup
also recognizes their `bench-exact-KO-*` names. Each attempted condition gets
one capped job; failed/ambiguous cases are retained and replaced deterministically,
not repeatedly repaired. The hard attempt cap is400. It stops earlier after50
unique valid targets. Initial catalog/pilot jobs were539,543,544,550,558;
failed or intentionally stopped auxiliary attempts545/546/547 remain in logs.
Current production job IDs live in `controller_state.json`.

Restart the geometry controller from the workspace root if needed:

```bash
setsid -f env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 benchmarks/env-cpu/bin/python -u \
  benchmarks/scripts/knockout_panel_controller.py \
  >> benchmarks/results/knockout_panel/controller.log 2>&1 < /dev/null
```

A file lock prevents concurrent controllers. It recovers active named Slurm jobs
on restart and preserves completed geometry. A duplicate launch can exit with a
lock error while the original controller continues. To inspect:

```bash
cat benchmarks/results/knockout_panel/controller.pid
cat benchmarks/results/knockout_panel/controller_state.json
squeue -u "$USER" -o '%.12i %.32j %.9T %.10M %.5C %.24R'
```

For a single explicit geometry attempt:

```bash
sbatch --chdir="$PWD/benchmarks" benchmarks/scripts/exact_knockout_panel.sbatch \
  panel --genes b4025
```

`--retry-problematic` explicitly archives an incomplete/problematic attempt
before rerunning; production does not automatically retry unsupported cases.
Existing ready geometry is immutable. No original source models or canonical
benchmark results were changed.

## Target and numeric contract

`gene_catalog.json` enumerates all1366 actual b-locus model genes, evaluating
COBRA GPRs exactly. The spontaneous-reaction placeholder s0001 is excluded.
Source reaction ordering is the original iJO1366 ordering. For a real gene KO,
every GPR-disabled reaction receives bounds[0,0]; no capacity approximation,
extra-constraint epsilon, bound clipping or reaction filtering is used.

The reference polytope is the existing581-dimensional cached iJO1366 A/b,T,shift,
not a new raw-model equality reduction. The raw reduction's582-dimensional result
must not be substituted. Each condition supplies an embedding `z=z0+N*y`, a
model directory and forward containment identities relative to that same cache.

`conditions/<gene>/models/shared/rounding/polytope.npz` contains A,b,start,T,shift;
its gem files retain the original reaction ordering and stoichiometry, with
KO bounds changed to zero. `conditional_model.json` is the source/reference;
`condition.json` records hashes, ranks, certificates, LP times and bound checks.
`embedding.npz` stores z0,N,E,f and retained base-row indices. T/shift are composed
with the embedding and are never manually zeroed.

Affine rank uses relative threshold1e-12 and must remain identical at1e-11,
1e-12 and1e-13 before a target is published. Base halfspaces are projected onto
the exact declared KO equalities. A removed row needs a numerical equality-span
and constant-bound certificate, not merely small coefficients. Every retained
A/b row is checked against its original cached row under the embedding.

If the projected origin is not strictly interior, a row-normalized Chebyshev LP
finds a center or an exposing dual. Positive dual support may become new equality
rows only when its nonnegative weights sum to1 and cancellation of BOTH full
normalized A and b is≤1e-8. This is not the set of rows active at a single vertex.
Tiny base rows are normalized inside the LP; original sampling A/b are untouched.
Every LP solver warning/status and small-matrix cutoff is recorded. A rejected
certificate is unsupported evidence, not an implicit epsilon relaxation.

The precision contract is floating-point rank/certificate validation as stated
above and original reaction/KO/stoichiometric checks at1e-7; it is not a rational
exact-arithmetic proof. Every ready target has a strictly positive interior
radius in its reduced y coordinates,33 interior ball points checked against
all original reaction bounds, every cached base inequality and KO flux zeros,
and an external33-point original S*v=b check. The external stoichiometry test
is in the model directory, timed separately; its shared S load is one-time.
These points are feasibility diagnostics, not convergence or uniformity samples.

Numerical equivalence grouping uses the affine-face projector and offset
within1e-10/1e-9, respectively, with the same cached base. Duplicate faces are
reported rather than counted as additional sampling targets. b0175 and b0494
initial point checks passed but their facial ranks changed across thresholds;
they are preserved as `precision_ambiguous` and excluded from sampling.
b3821's exposing certificate failed the unchanged1e-8 gate and remains
unsupported. Auxiliary broad-support LP attempts were numerically unstable;
the default production path uses the verified sparse-dual method, with no
hours spent forcing one hard case through. The auxiliary path remains opt-in
only (`NAJA_KO_BROAD_DUAL=1`) for separately bounded research.

## Panel strata and denominators

Strata count actual reaction bound sides changed to0. Real-gene availability is
367 at k=0,936 at1–5,54 at6–20,9 at21–50, and0 at51–200. Counting two encoded
inequalities for every reaction would include unchanged/redundant bound sides;
that alternate count is recorded but is not used to invent a missing stratum.
The largest actual real-gene change has42 sides. The zero-effect category is
represented by one control, not hundreds of copies inflating a speedup.
`panel_manifest.json` retains the original stratified plan and extra pilot
candidates; `controller_state.json` records subsequent replacement attempts.

The old raw-S nearconstant mask was used only to choose initially nondegenerate
candidate genes, not as an exact cached-target certificate. The one-time exact
cached-base mask is in `base_cached_mask/`:478 verified exclusions; other bounded
LPs remain active/unresolved. Original878 exclusions must not be inherited as
though certified on cached A/b. The sampling consumer owns valid conditional
inheritance and fresh/transported masks.

A stronger cached-only denominator is in the catalog: its strict base ball has
radius0.999988. Opposite feasible ball points prove reaction variation width
at least2*r*norm(T_row).704 genes affect a reaction whose certified lower bound
exceeds1e-7, and NONE of those704 can reuse the base point weakly after KO.
The48.46% weak-feasible rate over all1366 genes is mostly trivial/nearconstant
cases and must not be advertised as a nontrivial KO LP-skip rate. Analytic
projection can still yield a strict center after correct affine reduction:
PGI/b4025 is the clean580D,zero-interior-LP example (~0.56s total geometry/file
preparation), while genuinely new facial reductions can cost seconds or longer.

## LP initialization/basis audit

Jobs528/532/535 retain the capacity-slab microbenchmark. This is explicitly a
separate PGI capacity target, not the real KO panel. Tables and raw JSON are in
`lp_microbench_retained/summary.csv` and `summary.json`; old failed logs remain
under `lp_microbench/prior-attempt528/`.

| Operation, HiGHS threads1 | Median seconds |
|---|---:|
|First cold call|1.6387|
|New model/cold basis after pool initialization|1.5396|
|Same model, explicit setBasis reset|0.2422|
|Changed slab bounds, explicit setBasis reset|0.2410|
|Same model, internal basis/factorization retained|0.0150|
|Changed slab bounds, internal basis/factorization retained|0.01290|

One-time model construction is~0.010s; import, geometry loading and CSC assembly
are separate. SciPy default first call was1.404s and repeated fresh calls had
median1.321s. Default HiGHS creates64 threads despite four allocated logical
CPUs; explicit1/4/default comparisons did not show this as the main wall-time
cause. Setting a basis explicitly resets useful internal state and is different
from retaining the live solver model. Neighboring slab targets really change
bounds; they are not identical cached solutions. These microbenchmarks do not
establish13ms per arbitrary gene KO, whose affine/facial system can change.

Repeat that isolated audit into a fresh localized directory:

```bash
NAJA_LP_MICRO_OUT="$PWD/benchmarks/results/knockout_panel/lp_microbench_repeat" \
  sbatch --chdir="$PWD/benchmarks" benchmarks/scripts/start_lp_microbench.sbatch
```

## Sampling consumer

The report/quality agent owns the three-pipeline consumer (stale shared CPU,
barrier-corrected Naja, and global PolyRound reround+CPU). It must read only
`ready_manifest.conditions`, use a cached-target mask, preserve one common
mask across the validated square global-round map, and use verified F64 export
for Naja. It must not claim50 completed sampling comparisons merely because50
geometries become ready. Consumer PID/job IDs and restart commands will be
appended here once supplied; until then this owner file guarantees geometry
continuation only and the exact sampling-ready entrypoint above.

## Live sampling handover, 05:11 UTC

Sampling consumer was replaced during deadline/affinity hardening; final PID359158,
PPID1, is verified running independently of Codex. See [conditioned owner handoff](../../OWNER_CONDITIONED.md)
and the live results/conditioned_prototype/quality_consumer.pid before restarting.
Its source is `scripts/conditional_quality_controller.py`; durable state, PID and
log are `results/conditioned_prototype/quality_consumer_state.json`,
`quality_consumer.pid`, `quality_consumer.log`. Initial real-KO global preparation
jobs631(PGI/b4025) and632(b0154) were running at handover. It reads only the final
ready manifest, prepares at most two4CPU targets concurrently, then at most two
32CPU baselines and two4CPU GPU jobs. First panel seed is101. Naja uses verified
F64 binary5e6968...9f3d5. It stops by11:00UTC; pending/underconverged sampling is
reported as such. The quality agent owns detailed `OWNER_CONDITIONED.md`.

Restart the sampling consumer, with its file lock preventing duplicates:

```bash
setsid -f env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 benchmarks/env-cpu/bin/python \
  benchmarks/scripts/conditional_quality_controller.py \
  >> benchmarks/results/conditioned_prototype/quality_consumer.log 2>&1 < /dev/null
```

Its exact per-condition API is `scripts/conditional_prepare.py` followed by
`conditioned_quality.py --condition-root DIR --variant global_cpu|stale_cpu|naja_rotate
--seed101`. The consumer records complete queued commands; use those rather
than treating this compact variant notation as a literal shell command.

## Final geometry audit and strata limitation

Job653 independently audited all51 unique published targets and PASSED in5.93s.
It invoked actual `cobra.Gene.knock_out`, checked each conditional source/bound
change, reconstructed every affine containment identity and every positive
exposing dual against the original cached geometry, and checked all stored
strict-start/bounds/KO/S certificates. Evidence: `final_geometry_audit.json`.
The audit source and Slurm wrapper are `scripts/knockout_panel_audit.py/.sbatch`.

The final published strata are:one k=0 control,41 targets at k=1–5 and9 at
k=6–20. There are NO certified k=21–50 targets in this prototype, despite nine
biologically available genes in that bin: higher-k attempts failed numerical
rank/certificate gates. k=51–200 is biologically absent for these single-gene
changes. Thus this is a validity-selected lower-stratum prototype, not a fully
stratified completed benchmark. All85 attempted condition records remain:
52 geometries ready before deduplication(one duplicate),23 rank-ambiguous and
10 problematic. No timing or mixing outcome was used for selection.

`geometry_table.csv` and `geometry_metrics_snapshot.json` describe completed
geometry only; sampling completion is separately tracked by the live consumer.
