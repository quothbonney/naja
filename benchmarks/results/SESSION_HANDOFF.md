> **Closeout / publication snapshot (2026-11-08T21:04:30+00:00).** Benchmark campaigns are closed. Release fixes and tag `preprint-closeout-20261108` are pushed to `origin/publication-prep`. The one bounded Recon3D attempt and CPU diagnostics are complete: feasible at1e-7, DID_NOT_MEET_TARGET (median split-Rhat2.014367). No benchmark or diagnostic jobs remain. Use the workspace-root `CLOSEOUT.md` and `results/closeout/recon3d_bounded/summary.json` for final status. Report refresher377586 was stopped and pending cleanup851 canceled. Do not restart historical controllers or submit new benchmark campaigns. The older continuation instructions below are historical.

# Session handoff before disconnect

Snapshot: 2026-11-08T05:17:21.110289+00:00. Read [../RESUME.md](../RESUME.md) first.
This is a handoff snapshot; live state and Slurm accounting supersede job/PID snapshots.

The requested 30-minute stopping window ends 05:22:37 UTC on November 8.
The original campaign deadline remains 11:15:53 UTC. New conditioned jobs stop
being admitted at 10:15; their Slurm deadline is 11:00. Final cleanup is 11:05,
report refresh ends 11:10. Independent scheduler cleanup is job557, so batch
cleanup does not rely only on this chat or the login controllers surviving.

## First commands on return

From the workspace root:

```bash
python3 benchmarks/scripts/benchmark_status.py
squeue -u "$USER"
benchmarks/env-cpu/bin/python benchmarks/scripts/new_results_report.py
```

Do not restart dead historical controllers or submit duplicate jobs.
Inspect live PID/state/locks and each owner handoff before restarting anything.
All environments, code, caches, artifacts and commands remain under this folder.

## Work continuing autonomously

- Conditioned quality consumer: see [../OWNER_CONDITIONED.md](../OWNER_CONDITIONED.md).
  It measures full-global-reround CPU, inherited/stale CPU and Naja rotation,
  selected-budget fresh retimes, actual prefix curves and explicit failures/caps.
  Capacity has three seeds; real-KO panel initially one seed each. It consumes
  only validated ready-manifest targets, with separate bounded prepare jobs.
  A mechanical CPU-affinity restoration bug was found in selected-budget retimes;
  failed attempts are preserved and only that exact failure is eligible for a bounded retry.
- Full-node 512-chain CPU runs:451→552→553, then metrics554/555/556.
  They wait for original Recon3D preparation150_6 (9h cap, about08:23 UTC).
  [CPU owner handoff](../audit/high-throughput/OWNER_HANDOFF.md) gives states,
  restart commands and the independent deadline guard. Diagnostic mask202_8 is
  deliberately held; do not release it early and block the128-CPU allocation.
- Recon3D completion watcher validates completed artifacts before publication.
  Its separate owner state is under audit/recon3d-completion/.
- Live report refresh runs every60s; [RESUME_STATUS.json](RESUME_STATUS.json)
  records actual job/PID state when the status command last ran.

## Completed evidence and limits

- iML1515: GPU verified float64 and guideline CPU pass3/3 seeds.
  CPU worker draw median69.055s / GPU sampler-phase median8.117s =8.507×.
  Paired ESS/s ratio6.657×. This is a guideline/finite-tested comparison.
  CPU thinning2000 fails the fixed10k budget3/3; that does not prove it never converges.
- Recon3D barrier:93.8597→1.4728s,63.73×; CPU constraint rotation accounts for99%
  of the old phase. Full-matrix numerical parity is validated. This isolated
  setup result uses cached equality-only geometry, not a convergence run.
- 512-chain GPU group passes3/3; actual sampler median13.0012s.
  Full64CPU comparison remains pending. Do not publish an unmeasured full-node speedup.
- Fresh16 fixed-budget validation fails to generalize the old2.73× result:
  CPU passes2/3,GPU1/3. Separately extended budgets pass, with failures retained.
- The coordinate specialization reduces registers but its initial kernel speedups
  do not replicate in controlled512/1024-chain audits. Paired arrays match bitwise.
- Nsight timeline is complete: real10kHz metrics, three requested runs, SVG/PDF,
  rawCSV, SQLite, phase inference, exact commands, NCU and overhead in
  [nsight_timeline/README.md](nsight_timeline/README.md).

## Central conditioned claim: what is measured

The original PGI[-0.1,0.1] condition is a capacity restriction, not a zeroKO.
The original raw-model582D full-reround reference was incompatible with581D
inherited geometry. The repaired comparison globally rounds exactly the same
cached581D A/b; invertible-map and all-halfspace proof is saved. Global MVE
cost61.204s. First seed actual sampler timings are3.8167s CPU vs3.8936s Naja;
there is no sampling-latency gain there. The observed advantage is avoiding
full rerounding; correction costs about0.15s. Do not reuse the invalid134.9s
reference or infer time-to-target by interpolating prefix wall times.

Exact realKO geometry is complete:50 nontrivial unique targets plus one control.
Independent final audit passes all51. The validated strata are41 at1–5 changed
rows and9 at6–20; attempted21–50 cases failed certification and remain recorded.
This is validity-selected, not fully stratified or a random population estimate.
Analytic projection skips all interior LP calls in21/50 nontrivial accepted targets.
Typical total preparation is still seconds, not universally0.1s. Zero unchanged
base-start reuse is certified for704 nontrivial genes. Retained HiGHS state gives
~13ms in a neighboring-capacity microbenchmark; exactKO dimension changes limit reuse.
Sampling across the50 targets is active/incomplete, not50 completed comparisons.

## Reports and integrity

All new findings belong in [RESULTS_SINCE_UTILIZATION_REQUEST.md](RESULTS_SINCE_UTILIZATION_REQUEST.md),
which is regenerated from source JSON and includes live panel task counts.
Old canonical Markdown remains frozen; checksum verification at this snapshot:

- BENCHMARK_TABLE.md: PASS
- PROTOTYPE_SUMMARY.md: PASS

Source provenance, failed attempts, unsupported geometries and timing scope remain
explicit. Output writing is secondary to the lead sampling/worker metric.

Final health check05:18 UTC: detached refresh350073, full64 sequencer357882,
deadline guard357889, Recon watcher342818 and conditioned consumer359158
all alive with PPID1 and private sessions. Slurm cleanup557 remains pending
for11:05. Affinity repair validated by job667: capacity/globalCPU seed202
fresh selected2000 draws completed, sampler3.810884s,Rhat1.042586,ESS461.35,
zero violating samples. Seed303 retime669 is now running. No sampler math,
quality threshold or target geometry changed in this orchestration fix.

## Five-minute follow-up, 05:22 UTC

The repaired capacity three-pipeline comparison is now complete for all three
seeds. Full-global CPU selected2000 draws in3.8067–3.8167s; Naja selected500
draws in3.8738–3.8936s. Both pass3/3. Stale inherited CPU fails the50k-draw
cap3/3: median Rhat1.1315–1.1334 and ESS53.8–54.4 under the common assay.
The latest report now states this completed result and contains a live table
of completed exactKO three-pipeline comparisons.

First exactKO b4025 has all three selected-budget retimes complete:
globalCPU2000/3.963s, inheritedCPU2000/3.962s, Naja500/4.031s.
ExactKO inherited geometry requires affine-hull projection, which may already
repair mixing; stale inheritance passes here. Keep this distinct from the
finite-width capacity restriction. The unchanged b0002 control is excluded
from the matched performance table. Original frozen Markdown checksums still pass.

Final interpretation note: `running` in per-method quality.json is the last
recorded task state, not proof that its Slurm job is still alive. Cross-check
squeue/sacct for any unfinished rows, especially after deadline cancellation.
The live report labels these as recorded unfinished and never counts them as
completed or assigns inferred timings. Resume by inspecting those job IDs;
do not automatically retry scientific failures or interrupted capped studies.

Final five-minute-window check 2026-11-08T05:22:55.144136+00:00: consumer heartbeat current, no controller error; nine panel Naja tasks and three full-global CPU tasks complete, one exactKO all-three comparison complete. Counts include the separately labeled control where applicable. Jobs remain active and cleanup557 is queued. All new source changes in this window were report/handoff changes only.
