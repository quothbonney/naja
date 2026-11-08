# Naja benchmark implementation audit

Source audited: `naja-test-jack/naja`, current checkout. No source edits or jobs made by this audit.

## Immediate matched prototype

Use identical PolyRound arrays for both methods: `A z <= b`, strictly feasible `start`, affine reaction map `x = T z + shift`. Preferred Naja bundle:

```
models/ID/rounding/polytope.npz  # float64 C-contiguous A,b,start,T,shift
models/ID/gem/reaction_ids.txt  # column order of T
models/ID/gem/l_bounds.csv
models/ID/gem/u_bounds.csv
```

Vectors are one dimensional. Optional `rounding/extra.npz` has `extra_A,extra_b`; optional `start.npy` overrides shared start. Naja needs the `gem` directory even for reduced-space output. Build manifest should record n = reactions, d = A.shape[1], m = A.shape[0] and SHA256 of all arrays.

Example existing CLI (no new code required):

```bash
naja sample run --model-dir "$MODEL" --out-root "$OUT" --gpu 0 \
  --n-chains 16 --n-samples 10000 --tpb 128 --thinning 0 \
  --barrier-rotate --constraint-eps 0 --extra-constraint-eps 0 \
  --write-npy --verbose
```

Reduced-space outputs should be mapped externally to reaction space for diagnostics. Avoid `--backmap` in matched move-mixture experiments because it silently disables Dikin moves. `--backmap --bounds-policy filter --write-samples-valid` is supported when bounds filtering is required, but changes the kernel path. Save all raw samples and validity masks: filtering can alter chain alignment, so do not compute R-hat on an unstructured filtered matrix.

## Critical limitations for the proposed paper protocol

1. **No sampling `--seed`.** `job_sampling.cu:388` derives an integer seed from `std::hash(MODEL_NAME) ^ (hash(MODEL_DIR)<<1) ^ GPU_DEVICE`. Identical invocations repeat the same random stream. Minimal reproducible remedy is an explicit seed override + config/profile recording. A code-free workaround is three distinct model parent directories sharing identical array files; freeze those exact absolute paths and compiler version because std::hash is implementation-dependent. Do not call repeated identical commands three seeds.
2. **Dikin is not a CLI parameter.** Directions are generated automatically when extra constraints exist or `--pair-schedule-method barrier` is used. Probability is hardcoded 0.15 when directions exist (`job_sampling.cu:409`). `CoordinateHitAndRunBackmap` explicitly calls sampler with probability 0, and affine-hull output also uses that path. To test 0/0.15/0.30 requires a small explicit parameter and propagation or must be marked unsupported.
3. **Streaming is unreachable from CLI.** `CoordinateHitAndRunStreamed` exists in `gpu/chr.cu` but has no caller; no stream/chunk flag. Current backmap routine allocates full reduced and full mapped sample matrices. A 20M reaction-space run could exhaust H100 memory. Do not describe ordinary backmap runs as streaming.
4. **No burn-in in main run.** Every chain begins at exactly the same start. Initial retained samples enter diagnostics unless the external harness removes a predeclared warmup. Report warmup sampling cost and use identical discard conventions.
5. **Plain split-R-hat and conservative ESS.** `src/validate/mcmc.cpp` implements classical split-R-hat (not rank-normalized/folded). Its ESS per reaction is the *minimum single-chain ESS*, not pooled multi-chain ESS. Constant dimensions get R-hat 1 and ESS 1. Exclude fixed reactions consistently from aggregate medians and record their count. Use one external diagnostic implementation for Naja and hopsy; report its exact definition.
6. **`validate --model-dir` assumes reduced-space samples.** It applies T again in its bounds checker, so feeding already backmapped output can dimension-mismatch or double transform. Its JSON has per-dimension arrays from which p95 can be calculated; no native p95 summary.
7. **Auto thinning is geometry-based heuristic, not online adaptation.** With barrier rotate: kappa <=1e4 gives max(d/6,50); <=1e8 gives 200; otherwise 500. Without rotation: max(d/6,1). Explicit `--thinning N` overrides. Actual thinning is written into profile.
8. **Do not infer uniformity from R-hat or bound compliance.** Analytic tests + external matched-reference Wasserstein/correlation checks are required. Exact KO bounds can lower affine dimension. Relaxing constraints changes target; automatic hull reduction uses tightness at start, not a globally proved affine hull. Preprocess true equalities with PolyRound for the quality reference and flag failures.

## Supported controls

| Requested control | Existing interface / caveat |
|---|---|
| chains, retained draws | `--n-chains`, `--n-samples` (per chain) |
| barrier off / rotate / whiten | omit flags / `--barrier-rotate` / `--barrier-whiten` (implies rotate) |
| coordinate/fixed-pair mixture | `--pair-prob P` with no schedule |
| barrier Jacobi schedule | `--pair-schedule-method barrier`; nonpositive pair probability is forcibly changed to 0.5 |
| static pair schedule | `--pair-schedule CSV` (`--pair-schedule-file` alias) |
| runtime pair schedule | `--iter-rounding-passes N --iter-rounding-warmup N` |
| sparse moves | `--ksparse-prob P --ksparse-k K` |
| resync | `--resync-interval N` |
| threads/block | `--tpb 64/128/256/512` all supported (also 32,1024) |
| exact base / relaxed delta | `--constraint-eps 0 --extra-constraint-eps .01` in **run** |
| feasible-start LP | `--start-policy cube_center` (requires licensed Gurobi); default `file` avoids LP |
| bulk GPU list | `sample bulk --gpus 0,1,... --models-root ... --model-list ... --name ...` |
| bulk host concurrency | exactly one worker thread per listed GPU; no worker-count option |
| resume | bulk `--skip-existing`, `--resume-from` |
| bulk output staging | `--flat-output DIR --stage-dir DIR`; external syncer needed |

Bulk CLI does not expose seed, extra-constraint-eps, resync, sparse moves, runtime pair controls. Do not assume all run options work in bulk. Duplicate GPU IDs may create multiple workers, but must pilot for shared-context contention and CUDA thread safety; independent SLURM tasks are clearer.

## Timing, metadata and correctness

`profile.json` already records `load_polytope_s`, `cube_center_lp_s`, `barrier_rotation_s`, `upload_to_gpu_s`, `gpu_sampling_s`, `download_from_gpu_s`, `backtransform_s`, `write_output_s`, `total_s`; shape/thinning/tpb and sampling throughput. It omits seed/kappa; kappa appears with `--verbose`. `barrier_rotation_s` includes rotations/whitening and CPU work, not isolated Hessian/eigendecomposition. GPU-sampling phase includes backmap for backmapped runs. Some preparation phases are outside named timing bins; phase sums need not equal total. Add external wall time (process startup + output) for fair end-to-end speedup; retain kernel-only throughput separately. `--verbose` additionally computes Q'HQ on CPU, so use it consistently or minimally instrument structured kappa reporting.

There is no meaningful `naja --version`; record `git rev-parse HEAD`, dirty diff, compiler/CMake/CUDA/driver and build flags.

Configure H100 build with:

```bash
cmake -S naja-test-jack/naja -B benchmarks/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DNAJA_EIGEN_ROOT="$EIGEN" -DNAJA_GUROBI_ROOT="$GUROBI" \
  -DNAJA_ENABLE_GPU_TESTS=ON
cmake --build benchmarks/build -j 8
srun --partition=h100 --gres=gpu:1 --cpus-per-task=8 \
  ctest --test-dir benchmarks/build -V -R '^test_gpu_'
```

Five registered tests: `test_gpu_sampling_feasibility`, `test_gpu_sampling_cube_moments`, `test_gpu_sampling_simplex_moments`, `test_gpu_backmap_correctness`, `test_gpu_thin_polytope`. Tests use small synthetic polytopes and no Gurobi license. They do not exercise all barrier/mixture/streamed paths; pass means those specific tests passed, not all algorithms validated. Save CTest verbose output and hardware snapshot in same job.

## Sensible 12-hour prioritization

Complete H100 correctness first; then matched pilot core + iJO1366 and any other successfully preprocessed models, seed-separated runs, shared external reaction-space metrics and end-to-end timing table. Run CPU and GPU jobs concurrently with fixed nonoverlapping CPU allocations, avoiding BLAS oversubscription. Sixteen chains naturally support sixteen independent chain workers; allocating all 128 CPUs alone does not make sixteen sequential chains use 128 CPUs effectively. Measure rather than assume baseline parallel speedup. Expand seeds/models only after pilot validates target agreement, chain layout, metrics, and wall estimates. Mark incomplete models/time caps honestly; do not promise full 8,074-variant rebuild or factorial ablations in 12 hours.

## Localized campaign additions

After the initial audit, `sample run --seed` was added with zero preserving legacy hashing; resolved seed is written to profile/config snapshots. The exact diff is preserved at `benchmarks/results/source.patch`. `benchmarks/naja_runner.py` records the complete command and binary hash, draws 11,000 samples/chain by default, discards 1,000 retained warmup draws, and block-copies native draw-major output into an mmap `.npy` with shape `(chains,10000,d)` for shared external diagnostics. Warmup cost remains in sampling and wall timings. It does not enable verbose Q'HQ computation or GPU backmap. `scripts/verify_seed.py` tests repeatability and changed streams on a synthetic cube; `scripts/scaling.py` provides sequential chain/TPB sweeps. No Dikin, streaming, or diagnostic algorithms were changed.

GPU correctness sources use explicit failure tests, not `assert`, so their checks remain active in Release/NDEBUG builds.
