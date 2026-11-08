# Naja preprint closeout

Snapshot: 2026-11-08 20:31 UTC. Tasks 1–7 are measured or lookup results; task 6 was waived. The single bounded Recon3D attempt is running. No further benchmark campaigns are planned and nothing has been uploaded.

Local release: **`c9e5db8b8ed02c692b36e74e14b71eedcd871476`**, tag **`preprint-closeout-20261108`**, on `publication-prep` in `naja-test-jack/naja`. Executable: `benchmarks/build-closeout-release/naja`; SHA256 `2fc4654b47d9a2f8a3fb46df6fd5a70dc1fe1bbbb48f758d816a2b9a565b10b1`.

Commands below run from `/mnt/home/jordan+123/history-native-20261015-r5`, with:

```bash
B="$PWD/benchmarks"
N="$B/build-closeout-release/naja"
```

## 1. Merge, release and correctness

**Merged and locally tagged.** The release includes checked 64-bit element/byte offsets and step arithmetic, checked CUDA/cuBLAS/cuSOLVER errors with nonzero exits, genuine float64 exports in run/bulk, host prefaulting, and GPU `A*Q`. Legacy cuBLAS axis dimensions remain limited to `INT_MAX`; oversized axes fail explicitly rather than silently narrowing. Original working changes were imported in `b52a606` and separately preserved in the [original patch](benchmarks/audit/closeout/main-working-tree-before.patch) and stash `b0b7ff67fe8e9d41199319f7825fe3cb81f9d877`.

Defaults:

- `--host-prefault-threads -1`: automatic, enabled for at least 64 chains, capped at eight CPUs within the process affinity. `0` disables it. This avoids serial page faults during large downloads without changing the walk.
- `--barrier-rotate-backend cpu`: compatibility default. Select `gpu` explicitly for GPU `A*Q`; it changes floating-point accumulation and therefore the subsequent seeded trajectory. Export precision is independent of this choice.

**All five requested H100 tests pass**, using Release/SM90/production `--use_fast_math`: cube moments, simplex moments, feasibility, backmap and thin polytope. The GPU wide-index/error regression and host size/float64 regression also pass: **7/7** total. The backmap test includes 600,000 columns, exceeding the previous launch-grid limit. Five real negative probes return nonzero, including invalid GPU, backend/TPB/axis errors and allocation failure.

A positive allocation/copy/export smoke also passed with **2,147,484,160 doubles / 17,179,873,280 bytes**: sentinels beyond `2^31` survived a full device-to-host copy and float64 output streamed to `/dev/null`. Actual cuBLAS backmaps were tested with both a 17.18 GB input and a 17.18 GB output, including shift, download and full output-value checks. [Allocation evidence](benchmarks/results/closeout/task1/wide_allocation_result.json), [large-output evidence](benchmarks/results/closeout/task1/wide_output_result.json).

**Bit identity:** iJO1366, 16 chains, seed 101, 1,000 warmup + 10,000 retained draws, thinning 96, TPB128: the final default output matches the reference at **all 102,256,000 float32 scalar values**, with zero mismatches. The public reference `23873f01d3268a79bbf84499a66adce5e88cc6cb` has no explicit seed option; its only instrumentation was a seed-101 override, [patch SHA256 `c9f227…61339`](benchmarks/results/closeout/task1/reference-seed-only.patch). This is not an unmodified-CLI seed-101 claim. Explicit GPU rotation reproduces the earlier merged GPU trajectory bitwise, but differs from the legacy CPU-rotation trajectory; compatibility mode is required for legacy replay.

**RECON1 export check**, 160,000 retained points, every original reaction checked without filtering at tolerance `1e-7`:

| Actual export | Maximum original-bound violation | Fraction of samples with a violation |
|---|---:|---:|
| Reduced float32, external float64 backmap | 2.3576166920e-7 | 0.59701875 |
| On-device reaction-space float32 | 2.3576167507e-7 | 0.59701875 |
| Reduced float64, external float64 backmap | 2.3576167335e-7 | 0.59701875 |
| On-device reaction-space float64 | 2.3576167335e-7 | 0.59701875 |

Thus the on-device path also shows the RECON1 problem. Float64 does not repair this cached geometry's small residual floor. All four pass its previously declared `1e-6` tolerance; none supports “zero violations at `1e-7`.” Float64-to-float32 counterfactual casts reproduce the native float32 arrays exactly. These export measurements used merged candidate `fcc66fe63d5a6700786aebcf0b925374340e4c8b` with the GPU backend; the final release changes the default and exposes the same GPU path explicitly.

Exact build and test form:

```bash
"$B/scripts/enter.sh" cmake -S "$B/closeout-release" -B "$B/build-closeout-release" \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DNAJA_ENABLE_GPU_TESTS=ON \
  -DNAJA_EIGEN_ROOT="$PWD/naja-test-jack/deps/eigen-3.4.0" \
  -DNAJA_GUROBI_ROOT="$B/gurobi1302/linux64"
"$B/scripts/enter.sh" cmake --build "$B/build-closeout-release" --target naja \
  test_gpu_sampling_feasibility test_gpu_sampling_cube_moments \
  test_gpu_sampling_simplex_moments test_gpu_backmap_correctness \
  test_gpu_thin_polytope test_large_sizes_and_npy test_gpu_index_width -j 8
```

The [final gate](benchmarks/results/closeout/task1/final_compatibility_default/FINAL_GATE.md), [complete test log](benchmarks/results/closeout/task1/final_compatibility_default/gpu-correctness-tests.log), [per-command verification](benchmarks/results/closeout/task1/final_compatibility_default/summary.json), [export/identity data](benchmarks/results/closeout/task1/summary.json) and [release record](benchmarks/results/closeout/release.json) preserve the exact commands, reference patch, commits and binary hashes. The 64-chain prefault smoke resolves eight threads within an eight-CPU affinity.

## 2. Table 2 barrier column: final merged build

Median of seeds **101/202/303**, one H100, 16 chains, eight native draws **per chain** (128 total), thinning 1, explicit GPU backend; phase-only timings, not convergence runs.

| Model | d | m | Median barrier s | Three-seed range s |
|---|---:|---:|---:|---:|
| e_coli_core | 24 | 36 | 0.0643 | 0.0641–0.0647 |
| iAB_RBC_283 | 130 | 188 | 0.0801 | 0.0708–0.1531 |
| iJR904 | 232 | 291 | 0.0702 | 0.0698–0.0830 |
| iMM904 | 288 | 413 | 0.0711 | 0.0704–0.0714 |
| iAF1260 | 524 | 671 | 0.0877 | 0.0862–0.0878 |
| iJO1366 | 581 | 750 | 0.0888 | 0.0881–0.0890 |
| iML1515 | 635 | 801 | 0.0896 | 0.0895–0.0897 |
| RECON1 | 931 | 2120 | 0.2459 | 0.1058–0.2757 |

Actual seed-101 iJO1366 command (the other 23 calls are preserved in the linked run records; replay into a fresh output root):

```bash
"$B/scripts/enter.sh" "$N" sample run --model-dir "$B/data/models/iJO1366" \
  --out-root "$B/results/closeout/task2_3/final-release/task2_barrier/iJO1366/seed101/native" \
  --gpu 0 --n-chains 16 --n-samples 8 --seed 101 \
  --tpb 128 --thinning 1 --constraint-eps 0 --extra-constraint-eps 0 \
  --pair-prob 0.0 --bounds-policy ignore --write-npy \
  --barrier-rotate --barrier-rotate-backend gpu --sample-dtype float64
```

Commit: `c9e5db8`. [All 24 measured calls](benchmarks/results/closeout/task2_3/final-release/task2_per_seed.csv), [median table](benchmarks/results/closeout/task2_3/final-release/task2_barrier_summary.csv), [protocol/commands](benchmarks/results/closeout/task2_3/final-release/manifest.json), [SVG](benchmarks/results/closeout/task2_3/final-release/task2_barrier_dimension_scaling.svg), [PDF](benchmarks/results/closeout/task2_3/final-release/task2_barrier_dimension_scaling.pdf). Cached PolyRound preparation is separate; these are additional barrier-phase costs, not interchangeable global-rounding timings.

## 3. Rotation-off control and hopsy thinning units

Final-build three-seed medians: split-Rhat **1.186539**, p95 Rhat **1.423806**, bulk ESS **79.4047**, ESS per million actual coordinate updates **4.69962**. Sampler: **4.0857 s**; native process wall: **5.6754 s**; internal profiled total: **5.2683 s**; native process plus normalization: **6.3781 s**; complete Python adapter process: **6.8261 s**. All three have zero bound-violation fraction at `1e-7`; the fixed budget does not meet the convergence criterion. All three final sample arrays are bitwise identical to the earlier merged-candidate rotation-off arrays.

The protocol is 16 chains, 1,000 discarded warmup + 10,000 retained draws, thinning 96, TPB128, reduced float32, prefault off, seeds 101/202/303, canonical cached polytope/start and legacy 878-exclusion LP mask. There are **16,896,000 coordinate updates** including warmup; retained-only count is 15,360,000.

```bash
"$B/scripts/enter.sh" "$N" sample run --model-dir "$B/data/models/iJO1366" \
  --out-root "$B/results/closeout/task2_3/final-release/task3_rotation_off/iJO1366/seed101/native" \
  --gpu 0 --n-chains 16 --n-samples 11000 --seed 101 \
  --tpb 128 --thinning 96 --constraint-eps 0 --extra-constraint-eps 0 \
  --pair-prob 0.0 --bounds-policy ignore --write-npy
```

No barrier flag is supplied. These native calls write all 11,000 draws; the adapter removes warmup and the frozen metrics code applies the legacy mask. The full capture-plus-diagnostics replay entry point is:

```bash
sbatch --chdir="$PWD" "$B/scripts/closeout_task2_3.sbatch" \
  --binary "$N" --commit c9e5db8b8ed02c692b36e74e14b71eedcd871476 \
  --rotation-backend gpu --stage both --out "$B/results/reproduce-closeout/task2_3"
```

The [captured submission](benchmarks/results/closeout/task2_3/final-release/slurm_job.json), per-seed `closeout_command.json` and `closeout_metrics_command.json` preserve the original full invocations. **One hopsy CHRR thinning unit is one coordinate proposal/update, not a sweep.** hopsy uses a cyclic coordinate scan; Naja uses random scan. [Exact source chain and native smoke proof](benchmarks/results/closeout/task2_3/final-release/sourceproof/THINNING_SEMANTICS.md), [source hashes](benchmarks/results/closeout/task2_3/final-release/sourceproof/thinning_semantics.json), [final per-seed CSV](benchmarks/results/closeout/task2_3/final-release/task3_per_seed.csv).

## 4. Figure 2 traffic and output bytes

Measured with Nsight Compute on the final release, 512 chains × 1,200 native draws × thinning 96 = **58,982,400 chain-coordinate steps**. Counters include all native draws; file sizes below are payload bytes, excluding NPY headers.

| Measured input | Actual m / d | L2 bytes per chain per step | DRAM bytes per chain per step | Reduced float32 bytes/sample | Reaction float32 bytes/sample |
|---|---:|---:|---:|---:|---:|
| Base iJO1366 | 750 / 581 | 16,356.76 | 48.8549 | 2,324 | 10,332 |
| Exact PGI knockout | 750 / 580 | 16,352.21 | 48.5888 | 2,320 | 10,332 |

The supplied artifacts contain no knockout with `m≈1,500`; that requested configuration remains unverified. No synthetic duplicated constraints were substituted. **Use about 16.4 decimal kB/step for these measured 750-row inputs**, rather than 36 kB. The output columns are the exact payload calculations `4*d` and `4*n_reactions`; the reduced-space NCU run does not itself emit reaction-space output. The “about 10 kB/sample” statement is correct for reaction-space float32, not reduced float32. Internally the sampler stores double output; internal device bytes and final float32 file bytes are different quantities.

Exact command family (full invocations are saved per input):

```bash
"$B/tools/nsight/extracted/opt/nvidia/nsight-compute/2025.2.1/ncu" \
  --metrics lts__t_bytes.sum,dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum \
  --clock-control none --launch-count 1 -k 'regex:.*chrKernel.*' --export REPORT \
  "$N" sample run --model-dir MODEL --out-root OUT --gpu 0 \
  --n-chains 512 --n-samples 1200 --seed 101 --tpb 128 --thinning 96 \
  --constraint-eps 0 --extra-constraint-eps 0 --pair-prob 0 --bounds-policy ignore \
  --write-npy --sample-dtype float32 --host-prefault-threads 0 \
  --barrier-rotate --barrier-rotate-backend gpu
```

Commit: `c9e5db8`. [Counts, denominators and commands](benchmarks/results/closeout/task4/final_release/summary.json), [base command](benchmarks/results/closeout/task4/final_release/base/command.sh), [knockout command](benchmarks/results/closeout/task4/final_release/exact_PGI_KO/command.sh). Raw `.ncu-rep` and CSV files remain beside each command. The standalone reproducible driver is [closeout_task4_ncu.py](benchmarks/scripts/closeout_task4_ncu.py).

## 5. Runnable prepare/bulk CLI

The following isolated, already projected exact-KO fixture was parser/contract tested with the final release. `prepare` and bulk `--dry-run` both return zero. Remove `--dry-run` to sample; no new corpus sampling was performed.

```bash
F="$B/results/closeout/cli_fixture"
"$B/scripts/enter.sh" "$N" sample prepare --models-root "$F/models" \
  --model-list "$F/all.txt" --out-model-list "$F/prepared.txt"
"$B/scripts/enter.sh" "$N" sample bulk --models-root "$F/models" \
  --model-list "$F/prepared.txt" --out-root "$F/runs" --name closeout \
  --gpus 0 --n-chains 16 --n-samples 11000 --seed 101 \
  --tpb 512 --thinning 1000 --sample-dtype float64 --start-policy file \
  --extra-constraints auto --constraint-eps 0 --barrier-rotate \
  --barrier-rotate-backend gpu --backmap --bounds-policy filter \
  --bounds-eps 1e-7 --write-npy --dry-run
```

For a pre-rounded external library, `prepare` additionally accepts `--base-model-dir BASE --mode symlink`. Native prepare assembles inherited deltas; exact zero-width KOs need their affine-hull reduction beforehand. `--n-samples` includes warmup; benchmark adapters discard warmup externally. The prepared-name list alone is not a proof that every feasible-start solve passed. [Actual final parser validation and help](benchmarks/results/closeout/cli_validation.json), commit `c9e5db8`.

## 6. rs10k corpus facts

**Waived explicitly by the user.** No corpus-count or exchange-bound claim is made.

## 7. Existing fresh-seed extension timings

These measured processes already existed; nothing was rerun for this lookup.

| Method / seed | Retained draws/chain | CPU worker s | Inclusive sampler s | Actual process wall s |
|---|---:|---:|---:|---:|
| CPU / 303001 | 15,000 | 21.034593 | 26.812905 | 31.528857 |
| Naja / 101001 | 3,000 | — | 10.400100 | 12.052567 |
| Naja / 303001 | 3,000 | — | 10.458100 | 12.014802 |

They pass the historical reaction-mask criterion and unmasked `1e-7` bound checks. The original smaller fixed-budget failures remain separate. Exact historical binary/module hashes and commands: [extension timing CSV](benchmarks/results/closeout/fresh_seed_extension_timings.csv), [lookup record](benchmarks/results/closeout/task_notes.json).

The separately completed fixed 16-chain protocol on new seeds 444001/555001/666001 passes **3/3 each method under both diagnostic assays**, CPU 15k/GPU 3k retained. It gives **2.031× paired compute ratio** under either assay, **2.427× paired ESS/s** under the conservative 478-exclusion cached-target assay, and **1.868× paired ESS/s** under the original 878-exclusion mask. The latter is a read-only lookup of existing per-reaction statistics. [Frozen protocol and actual per-seed data](benchmarks/results/fresh16_conservative/iJO1366/summary.json), [original-mask lookup and exact source CSV paths](benchmarks/results/closeout/fresh16_original_mask_lookup.json), [lookup command](benchmarks/scripts/closeout_fresh16_existing_lookup.py). This is additional completed evidence, not a rerun of the old extension measurements.

## 8. One bounded Recon3D attempt

**Running; no convergence claim yet.** One H100, at most one GPU-hour, immutable merged build `fcc66fe` (all five fixes, GPU rotation explicitly selected). Equality-only cached geometry: `d=4,861`, `m=18,477`, 10,600 reactions. Its provenance records equality elimination **128.842 s**, MVE rounding **0 s**, and total preparation **287.720 s**, including original-target GLPK FVA. This uses PolyRound's equality-elimination routine but no iterative PolyRound rounding. The 2,723 pruned entries are strictly inactive inequality rows, not removed reactions; no reactions were zero-pruned.

Pilot: 512 chains, 64 warmup + 64 retained, thinning 810, sampler **28.7482 s**; median lag-1 coordinate correlation **0.969697**. The predeclared lag rule requests thinning 60,612, reduced by the remaining-runtime cap to **5,409**. Main: 512 chains, 1,000 warmup + **800 retained**, float64, barrier whitening, TPB128; predicted sampler approximately 45 min. The 1,800 native draws require about **71.68 GB** for both reduced output buffers, leaving the recorded workspace margin. Diagnostics run after releasing the GPU; no repeat is planned.

Executed main command (immutable merged candidate, not an instruction to launch another attempt):

```bash
"$B/scripts/enter.sh" "$B/build-closeout/naja" sample run \
  --model-dir "$B/results/closeout/recon3d_bounded/model" \
  --out-root "$B/results/closeout/recon3d_bounded/main/native" --gpu 0 \
  --n-chains 512 --n-samples 1800 --seed 101 --tpb 128 --thinning 5409 \
  --constraint-eps 0 --extra-constraint-eps 0 --pair-prob 0 \
  --bounds-policy ignore --write-npy --sample-dtype float64 \
  --barrier-whiten --barrier-rotate-backend gpu --start-policy file --resync-interval 0
```

Final observations will be added here. [Live attempt report](benchmarks/results/closeout/recon3d_bounded/RECON3D_ATTEMPT.md), [protocol](benchmarks/results/closeout/recon3d_bounded/protocol.json), [pilot decision](benchmarks/results/closeout/recon3d_bounded/pilot_decision.json), [capture status](benchmarks/results/closeout/recon3d_bounded/capture.json), [source geometry validation](benchmarks/results/closeout/recon3d_bounded/cached_geometry_validation.json).

## 9. Data-availability manifest

Nothing has been uploaded. The current manuscript source is absent locally and no connected document session was available. From the explicit task descriptions, **Table 2's barrier column** maps to task 2 and **Figure 2's byte annotations** map to task 4. The remaining numbered assignments for Tables 2–5, S1–S5 and Figures 1–3 require the manuscript/captions; they are not guessed.

The [file-exact inventory](benchmarks/results/closeout/data_availability_candidates.csv) lists result paths and byte sizes; the [machine-readable inventory](benchmarks/results/closeout/data_availability_inventory.json) records the inspected scope. The six retained 512-chain arrays alone total **103,520,256,768 bytes**. The [numbered mapping](benchmarks/results/closeout/paper_item_mapping.csv) distinguishes verified components from unassigned assets. The final closeout assets and Recon3D outcome are being added to that snapshot.

<!-- sized-inventory:start -->

File-size snapshot 2026-11-08T20:41:39+00:00: 131 existing unique files. Sizes are logical file bytes; aliases and embedded archive members should not be added again to estimate a deposit total. Unassigned assets are organized by content, without inventing manuscript numbering.

| Component / dataset | Existing file | Bytes |
|---|---|---:|
| bulk512 (unassigned) | [benchmarks/results/high_throughput/iJO1366/comparison.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/high_throughput/iJO1366/comparison.csv) | 6,932 |
| bulk512 (unassigned) | [benchmarks/results/high_throughput/iJO1366/comparison.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/high_throughput/iJO1366/comparison.json) | 35,380 |
| bulk512 (unassigned) | [benchmarks/results/high_throughput/iJO1366/AUDITED_COMPARISON.md](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/high_throughput/iJO1366/AUDITED_COMPARISON.md) | 6,537 |
| crossmodel (unassigned) | [benchmarks/results/crossmodel_extension/cached_assay_table.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/crossmodel_extension/cached_assay_table.csv) | 27,552 |
| crossmodel (unassigned) | [benchmarks/results/crossmodel_extension/cached_assay_summary.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/crossmodel_extension/cached_assay_summary.json) | 47,489 |
| conditioned50 (unassigned) | [benchmarks/results/knockout_panel/overnight_audit/per_condition.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/knockout_panel/overnight_audit/per_condition.csv) | 221,011 |
| conditioned50 (unassigned) | [benchmarks/results/knockout_panel/overnight_audit/paired_comparisons.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/knockout_panel/overnight_audit/paired_comparisons.csv) | 11,355 |
| conditioned50 (unassigned) | [benchmarks/results/knockout_panel/overnight_audit/fastest_cpu_comparisons.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/knockout_panel/overnight_audit/fastest_cpu_comparisons.csv) | 4,262 |
| conditioned50 (unassigned) | [benchmarks/results/knockout_panel/overnight_audit/summary.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/knockout_panel/overnight_audit/summary.json) | 479,480 |
| conditioned50 (unassigned) | [benchmarks/results/knockout_panel/overnight_audit/conditioned_panel_summary.svg](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/knockout_panel/overnight_audit/conditioned_panel_summary.svg) | 48,096 |
| conditioned50 (unassigned) | [benchmarks/results/knockout_panel/overnight_audit/conditioned_panel_summary.pdf](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/knockout_panel/overnight_audit/conditioned_panel_summary.pdf) | 15,960 |
| conditioned50 (unassigned) | [benchmarks/results/knockout_panel/overnight_audit/figure_source.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/knockout_panel/overnight_audit/figure_source.csv) | 3,143 |
| capacity_curves (unassigned) | [benchmarks/results/conditioned_prototype/iJO1366/quality/rhat_curve.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/conditioned_prototype/iJO1366/quality/rhat_curve.csv) | 26,286 |
| capacity_curves (unassigned) | [benchmarks/results/conditioned_prototype/iJO1366/quality/rhat_vs_chain_length.svg](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/conditioned_prototype/iJO1366/quality/rhat_vs_chain_length.svg) | 76,508 |
| capacity_curves (unassigned) | [benchmarks/results/conditioned_prototype/iJO1366/quality/rhat_vs_chain_length.pdf](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/conditioned_prototype/iJO1366/quality/rhat_vs_chain_length.pdf) | 23,986 |
| capacity_curves (unassigned) | [benchmarks/results/conditioned_prototype/iJO1366/quality/time_to_target.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/conditioned_prototype/iJO1366/quality/time_to_target.csv) | 9,147 |
| nsight_timeline (unassigned) | [benchmarks/results/nsight_timeline/naja_h100_utilization_timeline.svg](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/nsight_timeline/naja_h100_utilization_timeline.svg) | 291,124 |
| nsight_timeline (unassigned) | [benchmarks/results/nsight_timeline/naja_h100_utilization_timeline.pdf](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/nsight_timeline/naja_h100_utilization_timeline.pdf) | 98,512 |
| nsight_timeline (unassigned) | [benchmarks/results/nsight_timeline/timeline_source.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/nsight_timeline/timeline_source.csv) | 27,117,751 |
| nsight_timeline (unassigned) | [benchmarks/results/nsight_timeline/phases.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/nsight_timeline/phases.csv) | 1,903 |
| nsight_timeline (unassigned) | [benchmarks/results/nsight_timeline/ncu/kernel_summary.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/nsight_timeline/ncu/kernel_summary.csv) | 3,906 |
| nsight_timeline (unassigned) | [benchmarks/results/nsight_timeline/ncu/roofline_source.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/nsight_timeline/ncu/roofline_source.csv) | 2,678 |
| nsight_timeline (unassigned) | [benchmarks/results/nsight_timeline/commands.sh](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/nsight_timeline/commands.sh) | 9,488 |
| barrier_dimension (unassigned) | [benchmarks/results/barrier_dimension_scaling/source.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/barrier_dimension_scaling/source.csv) | 13,965 |
| barrier_dimension (unassigned) | [benchmarks/results/barrier_dimension_scaling/medians.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/barrier_dimension_scaling/medians.csv) | 679 |
| barrier_dimension (unassigned) | [benchmarks/results/barrier_dimension_scaling/barrier_dimension_scaling.svg](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/barrier_dimension_scaling/barrier_dimension_scaling.svg) | 65,941 |
| barrier_dimension (unassigned) | [benchmarks/results/barrier_dimension_scaling/barrier_dimension_scaling.pdf](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/barrier_dimension_scaling/barrier_dimension_scaling.pdf) | 21,021 |
| barrier_dimension (unassigned) | [benchmarks/results/barrier_dimension_scaling/summary.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/barrier_dimension_scaling/summary.json) | 35,066 |
| fresh16_fixed (unassigned) | [benchmarks/results/fresh16_conservative/iJO1366/table.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/fresh16_conservative/iJO1366/table.csv) | 2,211 |
| fresh16_fixed (unassigned) | [benchmarks/results/fresh16_conservative/iJO1366/summary.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/fresh16_conservative/iJO1366/summary.json) | 7,388 |
| fresh16_fixed (unassigned) | [benchmarks/results/fresh16_conservative/iJO1366/VALIDATION.md](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/fresh16_conservative/iJO1366/VALIDATION.md) | 1,849 |
| fresh16_extensions (unassigned) | [benchmarks/results/closeout/fresh_seed_extension_timings.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/fresh_seed_extension_timings.csv) | 2,778 |
| fresh16_extensions (unassigned) | [benchmarks/results/fresh16_validation/iJO1366/summary.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/fresh16_validation/iJO1366/summary.json) | 15,617 |
| uniformity (unassigned) | [benchmarks/results/uniformity/summary.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/uniformity/summary.json) | 83,404 |
| chain_scaling (unassigned) | [benchmarks/results/figures/chain_scaling_source.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/figures/chain_scaling_source.csv) | 831 |
| chain_scaling (unassigned) | [benchmarks/results/figures/h100_chain_scaling.svg](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/figures/h100_chain_scaling.svg) | 19,371 |
| verification (unassigned) | [benchmarks/results/setup/gpu-correctness.log](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/setup/gpu-correctness.log) | 3,468 |
| verification (unassigned) | [benchmarks/results/barrier_profile/verification/verification.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/barrier_profile/verification/verification.json) | 436 |
| bulk512_measured_experimental_code, verification (unassigned) | [benchmarks/results/coordinate_kernel/verification.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/coordinate_kernel/verification.json) | 911 |
| environment (unassigned) | [benchmarks/results/continuation_20261108/ENVIRONMENT_AND_PROTOCOL.md](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/continuation_20261108/ENVIRONMENT_AND_PROTOCOL.md) | 3,084 |
| environment (unassigned) | [benchmarks/results/setup/h100-environment.txt](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/setup/h100-environment.txt) | 6,696 |
| environment (unassigned) | [benchmarks/results/setup/binary.sha256](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/setup/binary.sha256) | 141 |
| environment (unassigned) | [benchmarks/results/setup/source.patch](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/setup/source.patch) | 6,694 |
| final_task1 (unassigned) | [benchmarks/results/closeout/task1/summary.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task1/summary.json) | 9,073 |
| final_task1 (unassigned) | [benchmarks/results/closeout/task1/TASK1_VERIFICATION.md](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task1/TASK1_VERIFICATION.md) | 2,962 |
| final_task1 (unassigned) | [benchmarks/results/closeout/task1/iJO_parity.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task1/iJO_parity.json) | 1,073 |
| final_task1 (unassigned) | [benchmarks/results/closeout/task1/f64_precision_retention_probe.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task1/f64_precision_retention_probe.json) | 750 |
| final_task1 (unassigned) | [benchmarks/results/closeout/task1/final_compatibility_default/summary.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task1/final_compatibility_default/summary.json) | 1,680 |
| final_task1 (unassigned) | [benchmarks/results/closeout/task1/final_compatibility_default/FINAL_GATE.md](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task1/final_compatibility_default/FINAL_GATE.md) | 1,961 |
| final_task1 (unassigned) | [benchmarks/results/closeout/task1/final_compatibility_default/gpu-correctness-tests.log](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task1/final_compatibility_default/gpu-correctness-tests.log) | 962 |
| final_task1 (unassigned) | [benchmarks/results/closeout/task1/RECON1_reduced_f64/original_bounds.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task1/RECON1_reduced_f64/original_bounds.json) | 1,760 |
| final_task1 (unassigned) | [benchmarks/results/closeout/task1/RECON1_reaction_f64/original_bounds.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task1/RECON1_reaction_f64/original_bounds.json) | 1,761 |
| final_task2 (unassigned) | [benchmarks/results/closeout/task2_3/final-release/TASK2_BARRIER.md](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task2_3/final-release/TASK2_BARRIER.md) | 1,533 |
| Table2_verified_barrier_column_data_component | [benchmarks/results/closeout/task2_3/final-release/task2_barrier_summary.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task2_3/final-release/task2_barrier_summary.csv) | 1,016 |
| Table2_verified_barrier_column_data_component | [benchmarks/results/closeout/task2_3/final-release/task2_per_seed.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task2_3/final-release/task2_per_seed.csv) | 16,525 |
| Table2_verified_barrier_column_data_component | [benchmarks/results/closeout/task2_3/final-release/task2_barrier_summary.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task2_3/final-release/task2_barrier_summary.json) | 34,036 |
| final_task2 (unassigned) | [benchmarks/results/closeout/task2_3/final-release/task2_barrier_dimension_scaling.svg](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task2_3/final-release/task2_barrier_dimension_scaling.svg) | 47,124 |
| final_task2 (unassigned) | [benchmarks/results/closeout/task2_3/final-release/task2_barrier_dimension_scaling.pdf](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task2_3/final-release/task2_barrier_dimension_scaling.pdf) | 17,150 |
| Table2_verified_barrier_column_data_component | [benchmarks/results/closeout/task2_3/final-release/manifest.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task2_3/final-release/manifest.json) | 1,277 |
| final_task3 (unassigned) | [benchmarks/results/closeout/task2_3/final-release/TASK3_ROTATION_OFF.md](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task2_3/final-release/TASK3_ROTATION_OFF.md) | 1,520 |
| final_task3 (unassigned) | [benchmarks/results/closeout/task2_3/final-release/task3_per_seed.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task2_3/final-release/task3_per_seed.csv) | 3,192 |
| final_task3 (unassigned) | [benchmarks/results/closeout/task2_3/final-release/task3_off_summary.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task2_3/final-release/task3_off_summary.json) | 8,048 |
| final_task3 (unassigned) | [benchmarks/results/closeout/task2_3/final-release/task3_FCC_OFF_parity.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task2_3/final-release/task3_FCC_OFF_parity.json) | 907 |
| final_task3 (unassigned) | [benchmarks/results/closeout/task2_3/final-release/resource_accounting.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task2_3/final-release/resource_accounting.json) | 225 |
| Figure2_verified_byte_data_component | [benchmarks/results/closeout/task4/final_release/summary.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task4/final_release/summary.json) | 5,249 |
| Figure2_verified_byte_data_component | [benchmarks/results/closeout/task4/final_release/base/metrics.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task4/final_release/base/metrics.csv) | 13,407 |
| Figure2_verified_byte_data_component | [benchmarks/results/closeout/task4/final_release/base/report.ncu-rep](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task4/final_release/base/report.ncu-rep) | 1,049,170 |
| Figure2_verified_byte_data_component | [benchmarks/results/closeout/task4/final_release/exact_PGI_KO/metrics.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task4/final_release/exact_PGI_KO/metrics.csv) | 13,407 |
| Figure2_verified_byte_data_component | [benchmarks/results/closeout/task4/final_release/exact_PGI_KO/report.ncu-rep](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/task4/final_release/exact_PGI_KO/report.ncu-rep) | 1,049,220 |
| bounded_Recon3D (unassigned) | [benchmarks/results/closeout/recon3d_bounded/protocol.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/recon3d_bounded/protocol.json) | 2,539 |
| bounded_Recon3D (unassigned) | [benchmarks/results/closeout/recon3d_bounded/pilot_decision.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/recon3d_bounded/pilot_decision.json) | 901 |
| bounded_Recon3D (unassigned) | [benchmarks/results/closeout/recon3d_bounded/capture.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/recon3d_bounded/capture.json) | 1,354 |
| bounded_Recon3D (unassigned) | [benchmarks/results/closeout/recon3d_bounded/jobs.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/recon3d_bounded/jobs.json) | 639 |
| bounded_Recon3D (unassigned) | [benchmarks/results/closeout/recon3d_bounded/RECON3D_ATTEMPT.md](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/recon3d_bounded/RECON3D_ATTEMPT.md) | 2,285 |
| bounded_Recon3D (unassigned) | [benchmarks/results/closeout/recon3d_bounded/naja-fcc66fe-source.tar.gz](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/recon3d_bounded/naja-fcc66fe-source.tar.gz) | 251,138 |
| bounded_Recon3D (unassigned) | [benchmarks/scripts/recon3d_bounded.py](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/recon3d_bounded.py) | 12,357 |
| bounded_Recon3D (unassigned) | [benchmarks/scripts/recon3d_bounded_capture.sbatch](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/recon3d_bounded_capture.sbatch) | 582 |
| bounded_Recon3D (unassigned) | [benchmarks/scripts/recon3d_bounded_metrics.sbatch](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/recon3d_bounded_metrics.sbatch) | 563 |
| bounded_Recon3D (unassigned) | [benchmarks/scripts/recon3d_bounded_report.py](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/recon3d_bounded_report.py) | 4,714 |
| bulk512_measured_experimental_code (unassigned) | [benchmarks/results/coordinate_kernel/source.tar.gz](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/coordinate_kernel/source.tar.gz) | 564,310 |
| bulk512_measured_experimental_code (unassigned) | [benchmarks/results/coordinate_kernel/kernel.patch](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/coordinate_kernel/kernel.patch) | 4,998 |
| bulk512_measured_experimental_code (unassigned) | [benchmarks/results/coordinate_kernel/compiler_resources.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/coordinate_kernel/compiler_resources.json) | 2,134 |
| bulk512_measured_experimental_code (unassigned) | [benchmarks/scripts/build_coordinate_kernel.sh](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/build_coordinate_kernel.sh) | 491 |
| bulk512_measured_experimental_code (unassigned) | [benchmarks/build-h100-coordinate/CMakeCache.txt](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/build-h100-coordinate/CMakeCache.txt) | 33,645 |
| bulk512_original_inputs (unassigned) | [benchmarks/data/models/iJO1366/rounding/polytope.npz](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/data/models/iJO1366/rounding/polytope.npz) | 15,524,304 |
| bulk512_original_inputs (unassigned) | [benchmarks/data/models/iJO1366/rounding/manifest.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/data/models/iJO1366/rounding/manifest.json) | 123 |
| bulk512_original_inputs (unassigned) | [benchmarks/data/models/iJO1366/meta.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/data/models/iJO1366/meta.json) | 1,644 |
| bulk512_original_inputs (unassigned) | [benchmarks/data/models/iJO1366/gem/A_eq.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/data/models/iJO1366/gem/A_eq.csv) | 116,562,936 |
| bulk512_original_inputs (unassigned) | [benchmarks/data/models/iJO1366/gem/b_eq.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/data/models/iJO1366/gem/b_eq.csv) | 45,125 |
| bulk512_original_inputs (unassigned) | [benchmarks/data/models/iJO1366/gem/l_bounds.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/data/models/iJO1366/gem/l_bounds.csv) | 65,211 |
| bulk512_original_inputs (unassigned) | [benchmarks/data/models/iJO1366/gem/u_bounds.csv](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/data/models/iJO1366/gem/u_bounds.csv) | 64,575 |
| bulk512_original_inputs (unassigned) | [benchmarks/data/models/iJO1366/gem/reaction_ids.txt](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/data/models/iJO1366/gem/reaction_ids.txt) | 21,999 |
| bulk512_original_inputs (unassigned) | [benchmarks/data/raw/iJO1366.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/data/raw/iJO1366.json) | 2,948,407 |
| bulk512_original_inputs (unassigned) | [benchmarks/data/raw/manifest.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/data/raw/manifest.json) | 3,075 |
| bulk512_assay_proofs (unassigned) | [benchmarks/results/high_throughput/iJO1366/cached_target_audit/model/active_reactions.npy](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/high_throughput/iJO1366/cached_target_audit/model/active_reactions.npy) | 2,711 |
| bulk512_assay_proofs (unassigned) | [benchmarks/results/high_throughput/iJO1366/cached_target_audit/model/diagnostic_mask.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/high_throughput/iJO1366/cached_target_audit/model/diagnostic_mask.json) | 435,958 |
| bulk512_assay_proofs (unassigned) | [benchmarks/results/high_throughput/iJO1366/cached_target_audit/reaggregation_validation.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/high_throughput/iJO1366/cached_target_audit/reaggregation_validation.json) | 2,699 |
| bulk512_assay_proofs (unassigned) | [benchmarks/results/high_throughput/iJO1366/cached_target_audit/array_hash_verification.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/high_throughput/iJO1366/cached_target_audit/array_hash_verification.json) | 1,451 |
| bulk512_assay_proofs (unassigned) | [benchmarks/scripts/reaggregate_cached_metrics.py](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/reaggregate_cached_metrics.py) | 6,353 |
| bulk512_assay_proofs (unassigned) | [benchmarks/scripts/validate_full64_reaggregation.py](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/validate_full64_reaggregation.py) | 2,155 |
| bulk512_assay_proofs (unassigned) | [benchmarks/scripts/audit_full64_comparison.py](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/audit_full64_comparison.py) | 22,106 |
| bulk512_assay_proofs (unassigned) | [benchmarks/scripts/cached_target_mask.py](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/cached_target_mask.py) | 4,621 |
| bulk512_assay_proofs (unassigned) | [benchmarks/scripts/cached_target_mask_parallel.py](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/cached_target_mask_parallel.py) | 2,204 |
| bulk512_drivers (unassigned) | [benchmarks/naja_runner.py](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/naja_runner.py) | 11,268 |
| bulk512_drivers (unassigned) | [benchmarks/cpu_baseline.py](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/cpu_baseline.py) | 16,150 |
| bulk512_drivers (unassigned) | [benchmarks/metrics.py](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/metrics.py) | 18,425 |
| bulk512_drivers (unassigned) | [benchmarks/scripts/high_throughput_sample.py](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/high_throughput_sample.py) | 5,907 |
| bulk512_drivers (unassigned) | [benchmarks/scripts/high_throughput_cpu.sbatch](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/high_throughput_cpu.sbatch) | 660 |
| bulk512_drivers (unassigned) | [benchmarks/scripts/coordinate512_worker.py](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/coordinate512_worker.py) | 2,932 |
| bulk512_drivers (unassigned) | [benchmarks/scripts/coordinate512.sbatch](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/coordinate512.sbatch) | 518 |
| bulk512_drivers (unassigned) | [benchmarks/scripts/benchmark_coordinate_kernel.py](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/benchmark_coordinate_kernel.py) | 3,153 |
| bulk512_drivers (unassigned) | [benchmarks/scripts/benchmark_coordinate_kernel.sbatch](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/benchmark_coordinate_kernel.sbatch) | 404 |
| bulk512_drivers (unassigned) | [benchmarks/scripts/timed_cpu_baseline.py](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/timed_cpu_baseline.py) | 1,383 |
| bulk512_drivers (unassigned) | [benchmarks/scripts/hopsy_worker_timing.py](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/hopsy_worker_timing.py) | 5,378 |
| bulk512_drivers (unassigned) | [benchmarks/scripts/quality_diagnostics.py](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/quality_diagnostics.py) | 7,856 |
| bulk512_native_CPU_build (unassigned) | [benchmarks/audit/hopsy_native_build.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/audit/hopsy_native_build.json) | 934 |
| bulk512_native_CPU_build (unassigned) | [benchmarks/audit/hopsy_native_compiler_target.txt](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/audit/hopsy_native_compiler_target.txt) | 11,138 |
| bulk512_native_CPU_build (unassigned) | [benchmarks/source/hopsy-1.7.0.tar.gz](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/source/hopsy-1.7.0.tar.gz) | 17,545,782 |
| bulk512_native_CPU_build (unassigned) | [benchmarks/wheels-native/hopsy-1.7.0-cp312-cp312-linux_x86_64.whl](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/wheels-native/hopsy-1.7.0-cp312-cp312-linux_x86_64.whl) | 1,115,516 |
| bulk512_native_CPU_build (unassigned) | [benchmarks/scripts/record_native_build.py](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/record_native_build.py) | 2,050 |
| bulk512_native_CPU_build (unassigned) | [benchmarks/audit/hopsy-worker-timing-verification/iJO1366-real-parity.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/audit/hopsy-worker-timing-verification/iJO1366-real-parity.json) | 1,154 |
| bulk512_native_CPU_build (unassigned) | [benchmarks/audit/metrics-parallel-verification.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/audit/metrics-parallel-verification.json) | 1,120 |
| bulk512_native_CPU_build (unassigned) | [benchmarks/audit/high-throughput/cpu-selected-budget.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/audit/high-throughput/cpu-selected-budget.json) | 2,271 |
| bulk512_native_CPU_build (unassigned) | [benchmarks/audit/high-throughput/full64-state.json](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/audit/high-throughput/full64-state.json) | 9,976 |
| final_release_source (unassigned) | [benchmarks/results/closeout/naja-preprint-closeout-20261108.tar.gz](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/naja-preprint-closeout-20261108.tar.gz) | 251,524 |
| bulk512_retained_array (unassigned) | [benchmarks/results/high_throughput/iJO1366/cpu_native512_physical64_thin2000/seed101001/samples.npy](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/high_throughput/iJO1366/cpu_native512_physical64_thin2000/seed101001/samples.npy) | 28,557,312,128 |
| bulk512_retained_array (unassigned) | [benchmarks/results/high_throughput/iJO1366/cpu_native512_physical64_thin2000/seed202001/budget12000/samples.npy](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/high_throughput/iJO1366/cpu_native512_physical64_thin2000/seed202001/budget12000/samples.npy) | 28,557,312,128 |
| bulk512_retained_array (unassigned) | [benchmarks/results/high_throughput/iJO1366/cpu_native512_physical64_thin2000/seed303001/budget12000/samples.npy](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/high_throughput/iJO1366/cpu_native512_physical64_thin2000/seed303001/budget12000/samples.npy) | 28,557,312,128 |
| bulk512_retained_array (unassigned) | [benchmarks/results/coordinate_kernel/paired/chains512/coordinate/samples.npy](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/coordinate_kernel/paired/chains512/coordinate/samples.npy) | 5,949,440,128 |
| bulk512_retained_array (unassigned) | [benchmarks/results/high_throughput/iJO1366/naja512_coordinate_tpb128_thin500_prefault8/seed202001/samples.npy](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/high_throughput/iJO1366/naja512_coordinate_tpb128_thin500_prefault8/seed202001/samples.npy) | 5,949,440,128 |
| bulk512_retained_array (unassigned) | [benchmarks/results/high_throughput/iJO1366/naja512_coordinate_tpb128_thin500_prefault8/seed303001/samples.npy](/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/high_throughput/iJO1366/naja512_coordinate_tpb128_thin500_prefault8/seed303001/samples.npy) | 5,949,440,128 |

<!-- sized-inventory:end -->
