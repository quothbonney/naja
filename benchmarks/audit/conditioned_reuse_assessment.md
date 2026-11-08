# Minimal conditioned reuse prototype

The new target is an explicitly labeled **PGI capacity knockdown**,
`-0.1 <= v_PGI <= 0.1`, on iJO1366. It is not an exact knockout. Original bounds,
medium, reaction order, stoichiometry and absence of a growth objective remain
unchanged apart from these two PGI bounds. No epsilon relaxation is permitted.
All new artifacts live in `results/conditioned_prototype/`.

For the saved base map `v = T z + shift`, the exact added inequalities are
`[T_PGI; -T_PGI] z <= [0.1 - shift_PGI; 0.1 + shift_PGI]`.
Both inherited pipelines read the identical augmented A/b, T/shift and strict
interior start in `models/shared/rounding/polytope.npz`. Stacking rows directly
ensures that the CPU wrapper consumes the conditional constraints. No CLI delta
is needed. Exact delta arrays are also retained in `exact_delta.npz`.

A read-only HiGHS LP maximized the Euclidean inscribed radius of this augmented
system. Radius was 0.0195258903 for iJO1366 (1.23 s) and 0.0145919120 for core.
These are full-dimensional capacity targets. The initial unnormalized LP returned a positive radius but violated tiny rows;
the strict raw-slack gate caught this in prep477 before sampling. A mathematically
equivalent positive row-normalized LP now checks every original raw slack. For
iJO1366 it gives radius0.0195258903450 and minimum original slack4.87408e-10.
The preparation script recomputes/persists that strict certificate/start on
the allocated node; saved target A/b are unchanged.

The three main pipelines are:

- `full_cpu`: set PGI bounds in a separate raw JSON, execute fresh full
  PolyRound simplification/equality elimination/rounding, then native hopsy CHRR.
- `stale_cpu`: inherit the base map, append the exact capacity rows, compute a
  strict interior start, then native hopsy CHRR without fresh rounding.
- `naja_rotate`: use exactly that same inherited geometry and start, with fresh
  Naja barrier rotation. `naja_off` is a separately identified correction control.

CPU runs reserve 32 logical CPUs and pin 16 workers to distinct physical cores,
with their siblings reserved. Both CPU pipelines test thinning 2,000. Naja uses
16 chains, TPB512, thinning 1,000, frozen binary c7baf4, both eps flags zero and
no filtering. These finite tested settings do not establish global optimality.
Augmented rows are embedded in A/b, so the code path does not auto-enable
extra-row Dikin moves. Rotation correction is timed inside the Naja process.

Seed101 runs first. Each pipeline executes 10,000 retained draws/chain, then a
fresh 50,000-draw process only if needed; 1,000 retained warmup draws are
included in process time and discarded before metrics. The explicitly declared
criterion is median classical split-Rhat <= 1.05 and median pooled bulk ESS
>= 100. Acceptance additionally requires no original/conditioned-bound violations
at 1e-7 and all-sample stoichiometric residual <= 1e-7. No failed cap becomes a
passing claim. Seeds202/303 are follow-ups only when seed101 is viable and
resources/deadline permit. Sampling is capped at 30 minutes per Slurm job and
no new budget starts at/after 11:00 UTC.

Each geometry gets a fresh conditional-source LP diagnostic mask. A common
conservative mask uses only reactions independently excluded by both masks;
independent certificates are retained. Base-mask files are not copied. Metrics
include every reaction and sample in bounds checks regardless of convergence
mask. Reaction-space mapping, all-sample PGI bounds and mass balance are checked
independently. Preparation validates source hashes, strict interior, map
residuals and a short hopsy pilot at 1e-7.

Per-condition totals include full preprocessing process wall for `full_cpu`,
or the full shared-preparation subprocess (imports, copying/assembly, strict interior LP and geometry writes), then sampling
process inclusive wall. Naja normalization and correction are included in that
process wall; correction must not be added twice. One-time base preprocessing
is reported separately. Diagnostics and failed/search-budget costs are retained
and excluded from selected-budget process comparisons. Preparation validation
and LP-mask audit work are separate diagnostic costs.

Source interfaces, relative to the workspace:

- `benchmarks/cpu_baseline.py:preprocess`: COBRA JSON ->
  `PolyRoundApi.cobra_model_to_polytope`, `simplify_polytope`,
  `transform_polytope`, `round_polytope`; writes A/b/start/T/shift and GEM data.
  CLI: `env-cpu/bin/python cpu_baseline.py prep --model CONDITION.json --out DIR`.
- `benchmarks/cpu_baseline.py:sample`: loads only NPZ A/b/start; it does **not**
  append `extra.npz`, which is why direct stacking is necessary here.
- `benchmarks/naja_runner.py`: native row-interleaved output normalization,
  preserved T/shift external backmap, both relaxation eps flags explicitly zero,
  backend-error/shape/finiteness guards and frozen binary hash metadata.
- `naja-test-jack/naja/src/cli/prepare/cmd_prepare.cpp:build_extra_from_bounds`:
  confirms the exact row/RHS formulas above.
- `naja-test-jack/naja/src/rounding/inherit.cpp:inherit_rounding_impl`: shares
  the full rounding bundle and clears a stale per-condition start sidecar.
- `naja-test-jack/naja/src/engine/job_sampling.cu`: augmentation precedes start
  validation and barrier correction; rotation is undone before reduced output.
- `naja-test-jack/naja/src/pipeline/feasible_start_files.cpp`: generic prepare
  accepts starts at 1e-3 and may invoke Gurobi; this prototype uses a strict
  external LP and avoids that expired-license path.
- The optional `--affine-hull-tol` uses rows tight at one start, not globally
  certified equalities. It is not sufficient evidence for exact zero knockout.
  Exact PGI=0 would require certified equality/nullspace elimination, a distinct
  target and additional validation; it is not silently approximated here.

Ready localized commands from workspace root:

```bash
sbatch --cpus-per-task=4 --mem=32G \
  benchmarks/scripts/conditioned_prototype.sbatch prepare
# After successful preparation and CPU-resource coordination:
sbatch --cpus-per-task=32 --mem=64G --nice=1000 \
  benchmarks/scripts/conditioned_prototype.sbatch run full_cpu stale_cpu
sbatch --cpus-per-task=4 --mem=32G --gpus=1 --nice=1000 \
  benchmarks/scripts/conditioned_prototype.sbatch run naja_rotate naja_off
benchmarks/env-cpu/bin/python benchmarks/scripts/conditioned_prototype.py summarize
```

These are prototype selected-budget comparisons, not a demonstrated reuse win.
There were no pre-existing measured conditioned quality artifacts in this
benchmark family. `scripts/run_bulk.sh` is a toy setup and documentation's
historical bulk claims do not supply this experiment's evidence.

Initial submissions are recorded in `results/conditioned_prototype/jobs.json`: prep477 waits for high-priority CPU512-chain pilot468; CPU478 and GPU479 wait for successful prep477. Seed101 only. All jobs use nice 1000 and 30-minute limits; no additional allocation is introduced ahead of 468.


The primary performance interpretation separates conditioning setup/correction
from sampling. Report measured selected-budget sampling-phase time and median
bulk ESS per sampling-phase second, with process/output handling secondary.
Current CPU phase wraps `hopsy.sample`, including multiprocessing and IPC;
current GPU phase wraps `CoordinateHitAndRun`, including device sample-buffer
allocation, slack GEMM and RNG initialization. Both exclude file output and
external adapters, but neither is silently equated to a pure compute timer.
Validated worker-compute and CUDA-kernel instrumentation will be added when
available. Full PolyRound setup versus inherited assembly/interior-LP plus local
barrier correction is a separate workflow setup comparison, not a GPU CHR-kernel
speedup. All complete-process and output costs remain secondary evidence.

The successful normalized preparation is job495. Its original producing script
SHA256 `f72964d529d493b07d46f6b23587edb3785f47511a013f8cc7e366b04b75603f`
is preserved in `results/conditioned_prototype/source_snapshots/conditioned_prototype.pre_worker_timer.py`.
Sampling uses the separately preserved worker-timer script version. CPU timing
bootstraps unchanged `cpu_baseline.py` through `scripts/timed_cpu_baseline.py`;
its frozen instrumentation SHA matches the real iJO parity sidecar. It records
C++ draw-call critical-path proxy (binding/state copy included) and total worker
thread CPU, with inclusive sample-call and wrapper process secondary. No
installed sampling source was changed.
