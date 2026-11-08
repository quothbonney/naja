> **Closeout / publication snapshot (2026-11-08T21:04:30+00:00).** Benchmark campaigns are closed. Release fixes and tag `preprint-closeout-20261108` are pushed to `origin/publication-prep`. The one bounded Recon3D attempt and CPU diagnostics are complete: feasible at1e-7, DID_NOT_MEET_TARGET (median split-Rhat2.014367). No benchmark or diagnostic jobs remain. Use the workspace-root `CLOSEOUT.md` and `results/closeout/recon3d_bounded/summary.json` for final status. Report refresher377586 was stopped and pending cleanup851 canceled. Do not restart historical controllers or submit new benchmark campaigns. The older continuation instructions below are historical.

# Resume the Naja benchmark campaign

Workspace: `/mnt/home/jordan+123/history-native-20261015-r5`.
Read this file and `results/RESULTS_SINCE_UTILIZATION_REQUEST.md` before changing anything.

**New continuation:** the user returned on November 8 at 19:12 UTC and authorized
three additional hours, through **22:12 UTC**. The original campaign below
completed and its controllers stopped normally. Current jobs have a 22:00
deadline, scheduler cleanup is job851 at 22:02, and report refresh ends 22:07.
Start with [the paper update](results/PAPER_UPDATE_20261108.md) and
[the continuation window](audit/continuation-20261108/window.json).
Do not restart the old overnight controllers with their expired 11:00 deadlines.

The user requested a natural stopping point before ending Codex around **2026-11-08 05:22 UTC**.
Slurm jobs and detached controllers are intended to continue after that session ends.
The original **12-hour campaign deadline remains 2026-11-08 11:15:53 UTC**.
Controllers stop launching earlier; work is cancelled at 11:05 and reports finish by 11:10.

## First commands

From the workspace root:

```bash
python3 benchmarks/scripts/benchmark_status.py
squeue -u "$USER"
benchmarks/env-cpu/bin/python benchmarks/scripts/new_results_report.py
```

The status command is read-only and writes `results/RESUME_STATUS.json`.
Check live PIDs before restarting controllers: do not submit duplicate jobs.
Controller lock files enforce singleton ownership where implemented.

## Reports and provenance

- `results/RESULTS_SINCE_UTILIZATION_REQUEST.md`: primary live report, refreshed every minute.
- `results/BENCHMARK_TABLE.md` and `PROTOTYPE_SUMMARY.md`: historical snapshots; do not append new results there.
- `results/new-report-boundary.json`: historical Markdown SHA256 checksums.
- `results/SAMPLING_BENCHMARKS.*`: inclusive sampler-phase comparison and finite tested configurations.
- `results/WORKER_MATH_BENCHMARKS.*`: CPU draw-call critical paths, parity proofs and GPU sampler comparisons.
- `results/nsight_timeline/`: completed 10 kHz H100 timeline, SVG/PDF, SQLite, NCU and roofline inputs.
- `results/barrier_profile/`: completed CPU/GPU rotation comparison and full matrix validation.
- `results/precision_quality/`: float64 iJR904/iML1515 export checks, all three seeds each.

All code, environments, caches and artifacts stay under this workspace.
Do not install globally or print/copy license credentials. Existing file-start runs use GLPK/HiGHS preprocessing.
Do not rebuild over frozen binaries. Experimental builds have separate source copies and hashes.
Do not edit frozen `metrics.py` to change estimators or hide failures.

## Current priorities

1. Three-pipeline conditioned quality versus chain length, with fresh measured time-to-target; then roughly 50 real gene knockouts.
2. Starting-point reuse/projection and persistent LP model/basis reuse; count trivial and nontrivial cases separately.
3. Include iML1515 time-to-target against guideline CPU; thinning-2,000 has now actually failed the 10k cap on all three seeds.
4. Fresh 16-chain validation and completion of the 512-chain full-node CPU comparison.

Owner handoffs: [conditioned sampling](OWNER_CONDITIONED.md),
[CPU sequencing](audit/high-throughput/OWNER_HANDOFF.md), and
[knockout geometry](results/knockout_panel/OWNER_KNOCKOUT_PANEL.md).
The final session snapshot is `results/SESSION_HANDOFF.md`.

## Results that must retain their scope

- Recon3D barrier phase: 93.8597 → 1.4728 s, 63.73× in the isolated GPU-rotation build.
  CPU A×Q was 99% of the old phase. Rotation including copies improved 188×.
  Full matrix relative error is 2.40e-15. This uses cached equality-only/no-MVE geometry;
  initial equality elimination remains a separate cost. Short checks do not establish convergence.
- The earlier 2.73× iJO1366 result used selection seeds and whole-process timing.
  Fresh fixed budgets pass CPU 2/3 and GPU 1/3. Separate bounded extensions pass;
  they do not rescue fixed-budget generalization. See `fresh16_validation/iJO1366/summary.json`.
- The 512-chain coordinate-specialized GPU group has three quality/bounds-passing seeds.
  Its actual sampler median is about 13.0 s. The earlier single-pair 1.29× kernel improvement
  did **not** replicate in a three-seed same-GPU audit. The exploratory 1,024-chain 1.98×
  improvement also failed replication (median paired ratio about 1.006×). Both audits
  preserve bitwise-identical outputs. A CUDA trace confirms the specialized bool=true
  kernel is dispatched. Reduced compiler register usage alone is not a runtime win.
  Keep exploratory and replicated evidence separate.
- iML1515: guideline native CPU worker draw median 69.055 s; verified float64 Naja sampler
  phase median 8.117 s. Ratio 8.507×; paired ESS-rate ratio 6.657×.
  CPU draw includes bindings/state copies; GPU sampler includes allocation/slack/RNG initialization.
  This is a guideline/finite-tested comparison, not a globally optimized CPU claim.
- LP retained internal state: neighboring PGI slabs solve in about 13 ms versus ~1.5 s
  for new models. Calling setBasis repeatedly forces ~0.24 s work. This is a capacity
  reoptimization result; exact knockout hull changes can limit reuse.

## Conditioned geometry correction

The original PGI capacity condition is [-0.1,0.1], **not an exact knockout**.
Independent raw-model rerounding produced 582 dimensions; inherited geometry has 581.
That original raw reference is invalid for same-target comparisons, although point checks passed.

The repaired reference globally rounds the exact cached conditioned 581-D A/b with no second
equality reduction. Its invertible square map and all 752 inequalities are checked.
See `conditioned_prototype/iJO1366/models/global_reround/global_reround_proof.json`.
Rounding cost is 61.204 s; do not reuse the incompatible old 134.9 s workflow claim.

`conditioned_prototype/quality_summary.json` is the current curve/timing API.
Criteria here are median Rhat ≤1.05, bulk ESS ≥100, bounds/mass balance at 1e-7;
the nine-model and 512-chain comparisons use Rhat ≤1.01.
Only separately executed selected-budget runs count as measured time-to-target.

Exact knockouts use affine/facial reduction, no epsilon relaxation. Rank/dual-certificate
failures remain unsupported and recorded. Point feasibility alone does not certify affine equivalence.
Parent LP-mask certificates must apply to the **cached A/b target**, not silently to a
different raw-model equality system. All-reaction bound checks remain unmasked.

## Full-node CPU sequence and holds

Validated CPU budget is 12,000 retained draws, 1,000 warmup, thinning 2,000.
The 48-core first-seed retime passed and supplies a whole-array identity reference.
Full-node sampling is queued as **451 → 552 → 553** on 128 logical / 64 physical CPUs.
Diagnostics **554/555/556** depend on the last sampling job, so all three samples are captured first.
Job 451 waits for original Recon3D preparation 150_6, whose 9-hour limit ends by 08:23 UTC.

`scripts/full64_sequence_controller.py` owns deferred-job holds and restoration.
State: `audit/high-throughput/full64-state.json`; holds: `full64-held-jobs.json`.
Original diagnostic 453 was replaced/cancelled; mask 202_8 is intentionally held.
Do not release that mask early and prevent the 128-CPU sequence from fitting.
The sequencer restores/cancels its own holds on completion/deadline.

## Detached controllers and bounded cleanup

Key PID files:

- `results/refresh.pid`: report/deadline loop (`scripts/refresh_results.py`).
- `audit/high-throughput/controller.pid`: broad 512-chain continuation.
- `audit/high-throughput/full64-controller.pid`: full-node sample/diagnostic sequencing.
- `audit/recon3d-completion/watcher.pid`: validation/publication of any completed Recon3D preparation.

Additional study PID/job manifests are listed by `benchmark_status.py` and owner handoffs.
Controllers run detached (`setsid`, PPID1), not through a live chat turn.
Several older controller PIDs are intentionally dead after completing or delegating their
groups. Do not revive the old broad/prefault/coordinate controllers when the full64
sequencer already owns the remaining work.
For a dead controller, inspect its state, lock and log before restarting its exact script.
Use localized Python and redirect logs into `benchmarks/`; do not use a global environment.

Independent scheduler cleanup job **557** starts at 11:05 UTC. It cancels only this user's
benchmark-named jobs whose work directory is this workspace or `benchmarks/`, then regenerates
the new report. It is independent of Codex and the login-process report loop.
Its record is `results/slurm-deadline-cleanup.json`; source is `scripts/slurm_deadline_cleanup.py`.

If resuming after the campaign deadline, inspect completed artifacts first. Further computation
requires a new user time window; do not restart the expired campaign automatically.
