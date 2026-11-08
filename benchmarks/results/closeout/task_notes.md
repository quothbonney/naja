# Closeout task notes: CLI, measured extensions and data availability

Snapshot 2026-11-08T20:56:50.976860+00:00. Tasks5/7/9 use existing data or syntax checks; task8 is the separately authorized one-hour-bounded GPU attempt.

## Task 5 — exact prepare/bulk commands

The following real fixture uses the already projected exact b4025 target. Preparation completed successfully; the merged bulk command with explicit seed and F64 output passed its parser/contract dry-run without GPU sampling. Source commit: `c9e5db8b8ed02c692b36e74e14b71eedcd871476`; validated release binary SHA256: `2fc4654b47d9a2f8a3fb46df6fd5a70dc1fe1bbbb48f758d816a2b9a565b10b1`.

```bash
/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/enter.sh /mnt/home/jordan+123/history-native-20261015-r5/benchmarks/build-closeout-release/naja sample prepare --models-root /mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/cli_fixture/models --model-list /mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/cli_fixture/all.txt --out-model-list /mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/cli_fixture/prepared.txt
/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/enter.sh /mnt/home/jordan+123/history-native-20261015-r5/benchmarks/build-closeout-release/naja sample bulk --models-root /mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/cli_fixture/models --model-list /mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/cli_fixture/prepared.txt --out-root /mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/closeout/cli_fixture/runs --name closeout_syntax_check --gpus 0 --seed 101 --sample-dtype float64 --n-chains 16 --n-samples 11000 --tpb 512 --thinning 1000 --start-policy file --extra-constraints auto --constraint-eps 0 --barrier-rotate --barrier-rotate-backend gpu --backmap --bounds-policy filter --bounds-eps 1e-7 --write-npy --dry-run
```

Remove `--dry-run` from the second command to sample this fixture later. It produces 11,000 native draws/chain; discarding 1,000 warmup draws/chain remains an external benchmark step. This is not a new corpus run.

For an externally supplied conditioned library with a pre-rounded base, the supported prepare form is:

```bash
naja sample prepare --models-root MODEL_ROOT --model-list ALL_MODELS.txt \
  --base-model-dir BASE_MODEL --mode symlink --out-model-list PREPARED.txt
naja sample bulk --models-root MODEL_ROOT --model-list PREPARED.txt \
  --out-root OUT --name RUN_NAME --gpus 0 --seed 101 --sample-dtype float64 \
  --n-chains 16 --n-samples 11000 --tpb 512 --thinning 1000 \
  --start-policy file --extra-constraints auto --constraint-eps 0 \
  --barrier-rotate --barrier-rotate-backend gpu --backmap \
  --bounds-policy filter --bounds-eps 1e-7 --write-npy
```

The capitalized paths in the second example are placeholders. Exact zero-width KO targets require affine-hull projection before inheritance can have a strict interior; native prepare does not perform that reduction.

Merged bulk supports `--seed` and `--sample-dtype`; the legacy C7 check is preserved separately. The per-model run manifests contain the exact benchmark commands. Prepare currently reports feasible-start failures without excluding those names from the written list. Shared `sample --help` is valid; `sample prepare --help` returns an unknown-flag error. See [CLI validation and saved help](cli_validation.json).

## Task 6 — waived

Waived explicitly by the user. No further corpus lookup or missing-input request.

## Task 7 — actual fresh-seed extension timings

| Method | Seed | Retained/chain | Worker draw s | Inclusive sampler s | Actual process wall s |
|---|---:|---:|---:|---:|---:|
| cpu | 303001 | 15,000 | 21.034593 | 26.812905 | 31.528857 |
| gpu | 101001 | 3,000 | — | 10.400100 | 12.052567 |
| gpu | 303001 | 3,000 | — | 10.458100 | 12.014802 |

All three extensions pass the historical 878-mask quality gate and unmasked 1e-7 bounds. These are actual fresh processes, not interpolated prefix times. GPU sampler time includes device allocation/slack/RNG setup; CPU worker time excludes IPC/output. Secondary process walls include output and relevant wrapper/normalization overhead. Original fixed-budget failures remain failures.

[Full timing/source/hash CSV](fresh_seed_extension_timings.csv). The separately completed new 16-chain protocol uses different seeds and fixed larger budgets; see [its report](../fresh16_conservative/iJO1366/VALIDATION.md).

## Task 8 — one bounded Recon3D attempt

Capture state: **native_capture_complete**. Immutable FCC source/binary, explicit GPU rotation backend and whitening; one H100 allocation capped59minutes, pilot+one main run only. CPU normalization/diagnostics use a separate job after GPU release.

The cached target has n10600,d4861,m18477. PolyRound-library equality elimination costs128.8417s, iterative MVE0s; total cached preprocessing287.7203s includes original-target FVA134.2945s. The2723 pruned items are strictly inactive inequality rows, not reactions. All10600 original reaction FVA widths are at least0.2; this assay uses no LP exclusions.

Pilot:512chains,64warmup+64retained,thinning810. Main:512chains,1000warmup+800retained,F64,TPB128; thinning follows the frozen pilot-lag/runtime rule. No additional attempt is permitted after OOM, timeout or failed quality. This equality-only study does not establish a matched-quality CPU sampling speedup or globally optimized time-to-quality.

[Frozen protocol](recon3d_bounded/protocol.json) · [Current capture](recon3d_bounded/capture.json) · [Slurm jobs](recon3d_bounded/jobs.json).

Main-run quality remains unmeasured until CPU diagnostics finish; no convergence claim is assigned yet.

## Task 9 — data availability and numbering

The user explicitly assigns the Task2 barrier-column data to Table2 and Task4 byte data to Figure2. Their exact source paths and sizes are recorded in [paper_item_mapping.csv](paper_item_mapping.csv). Only those components are verified; other columns/captions and the remaining numbered items are unassigned without manuscript source. Both measured Figure2-byte cases have m750; no m1500 conditional measurement is supplied.

| Paper component | Verified source file | Bytes |
|---|---|---:|
| Table2 | [task2_barrier_summary.csv](task2_3/final-release/task2_barrier_summary.csv) | 1016 |
| Figure2 | [summary.json](task4/final_release/summary.json) | 5249 |

[Content-organized artifact paths and exact byte sizes](data_availability_candidates.csv) · [Machine-readable inventory](data_availability_inventory.json). This includes available CSV/JSON tables, SVG/PDF figures, profiling source CSVs, hardware/correctness evidence and six 512-chain retained arrays. No large data were copied.

The512-chain measured GPU row uses experimental binary SHA51854688… and its own archived source.tar.gz/kernel.patch/verification records. The final generic C9release is a separate code artifact. The inventory now includes the original iJO1366 bundle (A,b,start,T,shift), GEM S/bounds/reaction IDs, raw-model metadata, sampler/diagnostic/instrumentation scripts, the conservative mask/proofs, native hopsy source/wheel/build manifest and timer verification. The start point is embedded as start.npy inside polytope.npz; no standalone start file is needed. Reproduction-asset refresh uses stat sizes only.
