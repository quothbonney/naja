# Closeout tasks 2 and 3

Await the merged production binary and its git commit before launching GPU work. The helper takes an explicit binary path and commit, records binary/source hashes and every native command, and leaves original canonical results unchanged. All runs use one H100; the single bounded Slurm allocation is at most 45 minutes, below the one GPU-hour limit.

Task 2 measures the complete production `barrier_rotation_s` phase on e_coli_core, iAB_RBC_283, iJR904, iMM904, iAF1260, iJO1366, iML1515 and RECON1. Each model runs seeds 101, 202 and 303, with 16 chains, 8 draws, no warmup, thinning 1, TPB 128, rotation enabled and float64 export. No trace-only environment toggles or experimental binaries are used. This is a timing probe, not a convergence test. Report medians and min/max across the three distinct seed tags. Historical CPU round-stage observations are separate reference data: the additional barrier phase operates on already-rounded cached geometry and does not replace full base preprocessing.

Task 3 is a fixed-budget control on iJO1366: rotation OFF, thinning 96, 16 chains, 1,000 warmup + 10,000 retained draws per chain, TPB 128, float32 reduced output and host prefault OFF, seeds 101/202/303. Use the original cached polytope/start and legacy LP diagnostic mask (878 exclusions) that the matched-thinning table used. This is explicitly separate from the conservative cached-target assay with 478 exclusions.

Report per-seed and median reaction split-Rhat, reaction p95 split-Rhat, median pooled bulk ESS, ESS per million actual coordinate updates, GPU sampler phase and secondary native/adapter/output walls. Actual updates include all chains and warmup: 16 × 11,000 × 96 = 16,896,000. Retained-only updates are 15,360,000 and are secondary. No prefix-time interpolation or new convergence-budget search is performed.

The [source proof](sourceproof/THINNING_SEMANTICS.md) confirms one hopsy thinning unit is one coordinate update, not a full sweep. Hopsy cycles through coordinates; Naja scans them randomly. A native-module three-dimensional box check confirms the source semantics, and the module hash matches the benchmark build manifest.

GPU launch helper: `scripts/closeout_task2_3.sbatch`, which invokes `scripts/closeout_task2_3.py --binary PATH --commit FULL_SHA`. Postprocessing plot: `scripts/plot_closeout_barrier.py`. Frozen diagnostics remain `metrics.py` with the recorded source SHA.
