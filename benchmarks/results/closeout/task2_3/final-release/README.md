# Final production closeout: tasks 2 and 3

Production commit: `c9e5db8b8ed02c692b36e74e14b71eedcd871476`.
Binary SHA256: `2fc4654b47d9a2f8a3fb46df6fd5a70dc1fe1bbbb48f758d816a2b9a565b10b1`.

[Task 2: eight-model barrier phases](TASK2_BARRIER.md) · [Task 3: rotation-OFF control](TASK3_ROTATION_OFF.md) · [All 27 run checks](artifact_checks.json) · [Exact Slurm command](slurm_job.json).

Task 2 explicitly uses the production `--barrier-rotate-backend gpu` option. The release defaults to the CPU rotation backend for legacy trajectory compatibility. Task 3 disables rotation entirely, keeps float32 reduced output and host prefault OFF, and uses the unchanged cached iJO1366 polytope/start and the legacy diagnostic mask with 878 exclusions.

The three-seed Task 3 medians are split-Rhat **1.186539**, reaction p95 split-Rhat **1.423806**, bulk ESS **79.4047**, and **4.69962 ESS per million actual coordinate updates**. GPU sampler time is **4.0857 s**; native total **5.2683 s** and adapter total **6.3781 s** are secondary. This fixed-budget control does **not** satisfy the 1.01 Rhat / 100 ESS target. All three runs have zero bound violations at the declared 1e-7 tolerance.

The [thinning source proof](sourceproof/THINNING_SEMANTICS.md) confirms one hopsy thinning unit is one coordinate update, not one full sweep. Hopsy uses cyclic coordinate scan; Naja uses random scan. The actual Task 3 update count includes all chains and warmup: 16 × 11,000 × 96 = 16,896,000. [All three final OFF arrays match the retained FCC OFF arrays bitwise](task3_FCC_OFF_parity.json).

[Barrier dimension figure: SVG](task2_barrier_dimension_scaling.svg) · [PDF](task2_barrier_dimension_scaling.pdf). Blue points are the medians of three distinct seed-tagged production phase measurements; error bars show min/max. The grey CPU series contains single historical PolyRound round-stage observations. These are different stages with different inputs: GPU barrier correction starts from already-rounded cached geometry. The figure does not claim that the additional barrier phase replaces full base preprocessing. The e_coli_core CPU reference used the login host; the other CPU references used the H100 node. No scaling exponent was fitted.

Job **959** completed successfully in **149 s**, reserving **0.04139 GPU-hours**. The retained FCC repetition and this final release repetition together reserved **0.08333 GPU-hours**. [Resource accounting](resource_accounting.json) and [Slurm record](slurm_accounting.psv) are saved. The previous FCC results remain in the parent directory and are not substituted into the final-release tables.

All commands, source snapshots, geometry/start/mask hashes, native profiles, raw/normalized samples and frozen-metric output remain alongside these summaries. No task 2/3 jobs or controllers remain running.
