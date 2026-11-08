# Closed out at user request — November 8, 20:23 UTC

Old conditioned campaigns are frozen. Controllers381746 and382068 were stopped at19:50;
queued cases and in-progress additional GPU captures were canceled. The four retained
CPU analyses finished; no selected GPU retimes were launched after closeout.
See `results/conditioned_extensions/CLOSEOUT_FREEZE.json` and each study summary for
completed versus interrupted/not-run cases. Do not restart these old controllers.

The replacement closeout tasks2/3 are complete on final production commit
`c9e5db8b8ed02c692b36e74e14b71eedcd871476`, with explicit GPU rotation backend for task2
and unchanged rotation-OFF legacy-F32 control for task3. Final results, all27 run checks,
source proof, commands, CSVs and SVG/PDFs are under
[results/closeout/task2_3/final-release/README.md](results/closeout/task2_3/final-release/README.md).
Job959 completed in149s (0.04139 GPU-h); no task2/3 jobs remain.

Everything below is historical handoff context and must not be treated as current restart authorization.

---

# Resumed conditioned work: November8 19:12–22:00 UTC

The historical handoff below is preserved. Its11UTC deadline and old PID359158 are expired.
All overnight tasks finished and were audited. New work is separate under
`results/conditioned_extensions/`; original50-target results and frozen metrics remain unchanged.

- [Audited50-KO table, proofs, fastest-validCPU and failures](results/knockout_panel/overnight_audit/CONDITIONED_BENCHMARK.md).
  Noncontrol coverage: global49/50, inherited32/50, rotate41/50;50k caps remain explicit.
  All51 inherited cached-parent containment/masks and50 global affine proofs independently recomputed.
- Same-GPU off/rotate/whiten hard-case array874_0-10%4:9 prior rotate failures,
  highest-SVD-kappa remaining targetb0003, and separate PGIcapacity.16chains,
  seed101,50kpilot,thin1000,warmup1000,TPB512,float64, then fresh selected retime.
  `results/conditioned_extensions/off_rotate_whiten_20261108/{manifest,summary}.json`.
  Initial expired-deadline wrapper attempt862 is preserved separately; replacement874 verified.
- All9 prior rotate failures extended to200k via a split pipeline.
  Current controller PID381746 is detached/PPID1; livePID under
  `results/conditioned_extensions/rotate_200k_20261108/pipeline.pid`.
  Original combined array884 was canceled after its firsttwo GPU captures completed;
  those15GB arrays are reused, sources/logs/accounting and transition history retained.
  GPU capture2concurrent,4logicalCPU/128GB/20min; frozen CPU diagnostics up4concurrent,
 8logicalCPU/4physicalcores/128GB/45min (firsttwo16logical jobs retained).
  Every predeclared pilot captures before GPU retimes. Full200k measured observations
  are separate from fresh selected-budget time-to-target.
- Mixture array889_0-2%1:6 configurations on frozenb0133,b0103(highestκamong prior
  rotate failures) plus PGIcapacity. Whiten+fixedpairs.25 versus native Jacobi.25+
  automatic Dikin.15 bundle. Actual direction count verified in verbose output;
  native profiler lacks mixture-probability fields, so.15 is source-inferred whenk>0.
- Separate b0031 full-reround recovery: uniform initial coordinatescale1e-6,
  unchanged default PolyRound settings and exact sameA/b. The default absolute
  dimensionality gate failed at its1e-9 cutoff, not a proven rank loss;
  the recovered composed map has P_min=8.37e-10.
  Full composed-map/halfspace/original feasibility proofs passed;64.405s MVE.
  CPU job887 passed at2000draws,3.97665s,Rhat1.03725,ESS529.7,bounds0.
  `results/conditioned_extensions/b0031_scaled_global/`; do not overwrite the original unavailable row.

All new jobs have Slurm deadline22:00 UTC.45min phases stop admission21:15;
20min capture phases stop21:40. The split controller cancels only its own leftovers
at22:00 and has a singleton lock. Root owns the global deadline cleanup.

Postprocess each study with `env-plot/bin/python scripts/summarize_conditioned_ablation.py
--study results/conditioned_extensions/STUDY`. This writes CSV/JSON/SVG/PDF/Markdown;
metrics/assays are frozen, prefix wall interpolation is never used for times.
The inherited-vs-rotate contingency is28both pass,13rotate-only,4inherited-only,5neither.
Full-reround-reference preparation+sampling median14.06× is **not** fastestCPU speedup;
fastest observed validCPU median is1.115×. Pure sampler ratio versus global CPU is1.017×.

---

# Owner handoff: conditioned quality benchmark

Snapshot: 2026-11-08 05:19 UTC. The Codex session may end near 05:22; Slurm and
the detached consumer continue. Do not depend on an agent remaining alive.

The durable sampling consumer is **PID 359158, PPID 1, SID 359158**, verified by
`ps`. Source: `scripts/conditional_quality_controller.py`. State, owned job IDs,
commands and heartbeat: `results/conditioned_prototype/quality_consumer_state.json`.
Log: `results/conditioned_prototype/quality_consumer.log`. Its file lock prevents
a second consumer from duplicating work. It reads **only**
`results/knockout_panel/ready_manifest.json` → `conditions`, never the initial
candidate catalog or obsolete geometry summaries.

It launches separate 4-CPU preparation/full-global-rounding jobs, then at most
two 32-logical-CPU/16-physical-core CPU jobs and two 4-CPU/one-GPU jobs at once.
Preparation has its own concurrency limit of two. Jobs have 45-minute limits.
New launches stop 10:15 UTC; subprocesses stop at 11:00 and the consumer cancels
only its own remaining jobs at 11:00. All work stays under `benchmarks/`.
Environment and plotting caches are localized to `home/`, `tmp/`, `cache/`;
no environment containing secrets is dumped.

## Priority milestone: capacity restriction

The capacity target is **PGI in[-0.1,0.1]**, not a zero knockout. Its shared
geometry is `results/conditioned_prototype/iJO1366/models/shared` (581D).
The independently reduced raw-model CPU reference produced 582D and is
**incompatible**, even though individual bounds/pilots passed. Original records
and old eligibility files remain preserved; they are not used for the main
comparison. `dimensionality_warning.json` contains the original warning and an
appended repaired-reference resolution.

The repaired full global PolyRound reference rounds the **same 581D A/b**,
without second simplification/equality elimination. Its square affine map
composes T/shift; proof is
`iJO1366/models/global_reround/global_reround_proof.json`.
A identities are exact; b relative error is1.6e-13. Global MVE costs61.2044 s.
Shared assembly plus strict interior LP costs1.57466 s; Naja correction is
additional (~0.15 s). One-time base preparation is separate. These setup costs
are not GPU sampling-kernel speedups.

The capacity quality assay is frozen to **732 verified cached-target LP
exclusions**,37 verified-wide candidates and179 unresolved candidates kept
active. Precision remains1e-7 for classification/bounds and1e-9 for the solver.
See `iJO1366/quality/mask_compilation.json`. The old878 raw-model-mask curves
are historical and are not mixed with this assay. Serial audit533 is additional
and must not silently replace this frozen common mask.

Measured seed 101 selected budgets under the repaired common assay:

| Method | Retained draws/chain | Actual sampler proxy s | Median Rhat | Median bulk ESS |
|---|---:|---:|---:|---:|
| Full global PolyRound + native CPU |2000|3.81672|1.04194|480.69|
| Shared + Naja rotation |500|3.8936|1.03843|466.56|

Both have zero violating samples at1e-7 and mass-balance maxima below3.5e-11.
There is **no sampling-latency win** in these first 16-chain timings. CPU timing
is accumulated C++draw-call critical path (binding/state-copy included;
IPC/startup/list assembly excluded). GPU is the sampler-phase proxy (device
allocation/slack/RNG initialization included; download/write/adapter excluded).
Total transitions/chain, including 1000 discarded warmup draws, are 6M CPU versus
1.5M Naja. The primary comparison separates setup/correction from sampling;
complete process/output time remains secondary.

Criterion: masked median classical split-Rhat <=1.05 and pooled bulk ESS >=100,
with original/conditional bounds and mass balance separately checked at1e-7.
This is distinct from the nine-model / large-chain benchmark’s 1.01 criterion. Grid:
250,500,1k,2k,5k,10k,20k,50k; 200k is opt-in only if bounded/necessary.
The earliest passing prefix with no later observed failure chooses a budget;
that exact budget is then **freshly executed and diagnosed**. No prefix wall
interpolation is used. Nonmonotonic/capped controls remain failures.

Initial common-mask jobs 559/560/561 cover seed 101. The detached consumer adopted
them, then launched 627/628(global/inherited CPU202),629/630(Naja202/303),
631/632(real-KO PGI/b0154 preparation). Live state is authoritative; later IDs
are appended automatically. Capacity has three planned seeds; the real panel
starts with seed 101 per target. Incomplete and failed tasks are never counted
as completed comparisons.

## Real exact KO consumer/API

Only rank-stable, exactly constrained ready geometries may be sampled. Inherited
CPU means **inherited rounding after required orthonormal affine-hull
projection**, not the finite-width capacity geometry. This required projection
may already improve mixing; do not assume Naja rotation adds a gain.

Each condition directory contains:

- `condition.json`: actual gene/GPR-disabled reaction IDs, fixed-reaction
  certificates, no-epsilon/rank gates, preparation/LP counts and validation.
- `embedding.npz`: `z0,N,E,f,kept_base_rows`, mapping base581D into conditional y.
- `models/shared/rounding/polytope.npz`: float64 `A,b,start,T,shift`.
- `models/shared/gem/`: original reaction order, original/KO bounds and S.
- `models/shared/meta.json`: conditional source/hash, dimensions and provenance.

`scripts/conditional_model_api.py` exposes:

- `compile_conditional_mask(condition_root)`: checks exact cached-base
  containment via embedding/kept rows, T/shift, sameS/IDs/objective and nested
  bounds. Inherits only the **478 verified cached-base** LP certificates in
  `results/knockout_panel/base_cached_mask/`, plus explicitly imposed fixed KO
  bounds. Other unresolved/new candidates stay active. It does not use raw-model
  LPs as a cached-target oracle or claim fresh LPs for inherited certificates.
- `prepare_global(sharedDIR,outputDIR)`: runs full global PolyRound directly on
  the exact projected A/b and transports the same mask under the square affine
  proof. No new raw-model reduction occurs.

CLI preparation: `scripts/conditional_prepare.py --condition-root DIR`.
CLI quality: `scripts/conditioned_quality.py --condition-root DIR --variant
{global_cpu,stale_cpu,naja_rotate} --seed 101`.
Real-KO Naja automatically uses the independently verified float64-export
binary `build-h100-f64/naja`, SHA256
`5e6968eb7d3cebf5f27c16d2662a2e640cc3f6b7a633b7e7ae3bc9136a99f3d5`,
with explicit `--sample-dtype float64`. Internal double sampling math is
unchanged; the export/build choice is recorded. Capacity c7/F32 history stays
separate. Every original bound and every KO equality remains in unfiltered
checks; no epsilon relaxation or clipping is allowed.

## Report/artifacts

Per condition:
`quality/summary.json`, `rhat_curve.csv`, `rhat_vs_chain_length.{png,svg,pdf}`,
`rhat_vs_transitions.{png,svg,pdf}`, `time_to_target.csv` and
`time_to_target.{png,svg,pdf}` when fresh selected records exist.
CSV includes actual thinning, retained transitions and total transitions with
warmup. Transition-axis plots are postprocessing, not new MCMC.

Aggregate capacity: `results/conditioned_prototype/quality_summary.json`.
Aggregate real panel: `results/knockout_panel/quality_summary.json`.
Source CSVs/proofs accompany every figure. Postprocessing CLI:
`env-plot/bin/python scripts/summarize_conditioned_quality.py --condition-root DIR`.
The controller refreshes changed plots and summaries automatically.

## Safe restart/check

From workspace root, check PID/state before starting another consumer. The
source file lock also prevents duplicates. Restart only if359158 is absent:

```bash
B=/mnt/home/jordan+123/history-native-20261015-r5/benchmarks
cd "$B/.."
env HOME="$B/home" TMPDIR="$B/tmp" XDG_CACHE_HOME="$B/cache" \
 MPLCONFIGDIR="$B/cache/matplotlib" PYTHONDONTWRITEBYTECODE=1 \
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 setsid "$B/env-cpu/bin/python" "$B/scripts/conditional_quality_controller.py" \
 >> "$B/results/conditioned_prototype/quality_consumer.log" 2>&1 < /dev/null &
```

`quality_consumer_state.json` retains submitted IDs, so restart does not
resubmit already-recorded tasks. Failed/time-capped jobs remain explicit and
require inspection rather than fabricated successful timings. Their arrays,
logs and scientific status are preserved. A ready geometry may still fail
rounding or sampling; the controller reports that without silently relaxing it.
For an intentional reviewed retry, preserve the prior run directory and remove
only that task’s state entry after creating a fresh output destination; never
reuse an incomplete sample file as completed data.


The original consumer PID358228 was replaced by PID359158 after adding a
per-job Slurm `--deadline=2026-11-08T11:00:00` and a10:15 admission cutoff for
45-minute jobs. Its source/state before the change are preserved as
`source_snapshots/conditional_quality_controller.before_slurm_deadline.py` and
`quality_consumer_state.before_slurm_deadline.json`. PPID1/SID359158 and the new
Slurm deadline argument were verified in the live state and submitted jobs651/652.
Future submissions checkpoint their job ID immediately, before the next loop.

Preparation/cache failures do not block the entire consumer. Their
`quality/preparation_status.json` records the error; other targets continue.
There is no blind automatic retry. If a shared-mask/containment check succeeded
but global MVE failed, inherited CPU and Naja may still run; the unavailable
global comparator stays an explicit missing/failed method. Existing successful
square-map proofs and geometry hashes are cached and checked before reuse.
All helper APIs are frozen after the deadline-safety change; ordinary geometry
and result artifacts may continue to appear without code edits.

Final affinity safety fix: the CPU helper restores the original Slurm allocation affinity before verifying/pinning each new budget. The previous failure came from checking siblings after the parent had already been pinned to16 cores. This preserves a real32-logical/16-physical reservation; the gate was not relaxed. Only that exact known orchestration error is retried once, with quality.failed_affinity.json preserved. Other failures still require review. The final PID and source hashes are in final_consumer.json.


Final freeze/validation: retry667 completed the repaired seed202 CPU selected
retime:3.810884 s, Rhat1.04259, no feasibility violations. Original pilot and
failure records remain preserved. The known-affinity retry handler covers all
four observed mechanical failures once each, including capacity202/303 and
real-KO PGI global/inherited CPU. No verification gate was relaxed. Controller
PID359158 is the final durable consumer; its source contains per-job11:00 Slurm
deadlines and10:15 admission cutoff. Do not start a duplicate while it is alive.

The new exact-KO panel and capacity assay use criterion1.05/ESS100. The older
nine-model and large-chain benchmarks use1.01. Zero-effect controlb0002 remains
separately labeled and must be excluded from performance headlines/aggregates;
rerounding an unchanged base would be artificial setup overhead.
