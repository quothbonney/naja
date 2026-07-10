# Naja: Technical Fact Sheet

## What Naja Is

Naja is a GPU-accelerated uniform sampler for convex polytopes, written in C++17/CUDA and targeting NVIDIA A100 GPUs. It implements Coordinate Hit-and-Run (CHR) as a CUDA kernel, with the core innovation being a collection of techniques that make CHR practical at scale: barrier-Hessian-based rounding that replaces expensive iterative preconditioning, a multi-GPU bulk execution engine that can process thousands of models in a single invocation, and a streaming architecture that decouples GPU memory from sample count.

The primary application domain is metabolic flux sampling: given a genome-scale metabolic model expressed as a polytope `{x : Ax <= b}` in a pre-rounded coordinate system, naja draws uniform samples from the feasible flux space. It is designed to handle the "many models" regime where you have hundreds or thousands of conditioned variants of a base model (e.g., one per tissue sample or experimental condition) and need to sample all of them.

The codebase is approximately 9,500 lines of source across ~50 files, organized into six subsystems: CLI, Engine, GPU, Rounding, Pipeline, and Conditioning.

---

## Architecture Overview

```
main.cu  (entry point: dispatches to sample | condition subcommands)
  |
  +-- CLI Layer           (command parsing, flag validation, dispatch tables)
  |     cmd_run           single-model sampling
  |     cmd_bulk          multi-GPU batch mode
  |     cmd_prepare       batch model setup
  |     cmd_verify        model file validation
  |     cmd_eval_rounding rounding quality diagnostics
  |     cmd_calibrate_rounding  cross-model schedule calibration
  |     cmd_inherit_rounding    copy/symlink rounding between models
  |
  +-- Engine Layer        (orchestration)
  |     job_sampling.cu   single-model pipeline: load -> round -> upload -> sample -> download
  |     job_bulk.cpp      multi-GPU work-stealing dispatcher
  |     bounds_filter     post-sampling GEM validity check
  |
  +-- GPU Layer           (CUDA kernels)
  |     chr.cu/cuh        CHR kernel (core sampler)
  |     backmap.cu/cuh    fused back-transformation kernel
  |     dmatrix/dvector   RAII device memory wrappers
  |     device_utils      GPU enumeration and context switching
  |
  +-- Rounding Subsystem  (preconditioning)
  |     barrier_rotation  eigenbasis rotation from barrier Hessian
  |     barrier_schedule  analytical Jacobi pair construction from Hessian
  |     runtime_warmup    empirical pair schedule refinement
  |     jacobi            2x2 Jacobi diagonalizer
  |     plan              rounding plan orchestrator
  |     schedule_io       pair schedule CSV persistence
  |     inherit           rounding inheritance across models
  |
  +-- Pipeline Subsystem  (model I/O)
  |     model_contract    directory structure validation
  |     extra_constraints constraint augmentation (KO models)
  |     feasible_start_lp Gurobi LP for inscribed-cube starting point
  |     run_manifest      JSON provenance tracking
  |
  +-- Conditioning        (E-Flux/E-Flux2 bound tightening)
        eflux/eflux2      expression-based constraint tightening
        gpr               Gene-Protein-Reaction boolean parser
```

---

## The Single-Job Sampling Pipeline

When naja processes a single model, the following sequence executes:

1. **Load polytope data** from CSV: constraint matrix `A` (m x n), right-hand side `b` (m), starting point `x0` (n). These come from a pre-rounded representation where the original model `{x : A_orig * x <= b_orig}` has already been transformed via PolyRound's maximum volume ellipsoid procedure into a reduced coordinate system.

2. **Augment extra constraints** if present. For conditioned models (knockouts, tissue-specific bounds), additional constraint rows `(A_extra, b_extra)` are appended to the base polytope. This is the mechanism that distinguishes one conditioned model from another: same base rounding, different extra rows.

3. **Global constraint relaxation**: optionally inflate all bounds by a small epsilon (`b += eps`). This prevents numerical infeasibility at tight constraint boundaries, which matters when the starting point sits near a facet after extra constraints are added.

4. **Feasibility check**: verify that the starting point satisfies `Ax <= b` to within tolerance 1e-9. This is a hard gate -- if the point is infeasible, the job aborts.

5. **Starting point policy**: either use the file-provided start point, or solve a Gurobi LP to find the center of the largest axis-aligned inscribed cube inside the polytope. The cube-center policy gives a more interior starting point, which improves early-chain behavior.

6. **Affine hull reduction**: if enabled, detect tight (active) constraints at the start point, compute the null space of the corresponding equality sub-matrix via `Eigen::FullPivLU`, and project the polytope into the lower-dimensional space where sampling has full volume. This eliminates degenerate dimensions that would otherwise produce identical coordinates across all samples. The back-transformation `x_original = hull_basis * x_reduced + hull_shift` is composed into the backmap matrix.

7. **Barrier rotation**: rotate the polytope into the eigenbasis of the log-barrier Hessian (detailed in its own section below).

8. **Barrier whitening**: optionally rescale each rotated axis by `1/sqrt(eigenvalue)` to make the local barrier metric approximately identity (detailed below).

9. **Upload to GPU**: transfer `A`, `b`, and replicated `X0` (one copy per chain) to device memory as `DMatrix<double>` / `DVector<double>` objects.

10. **Build rounding plan**: construct the Jacobi pair schedule that will be used during sampling (detailed in its own section below).

11. **GPU sampling**: launch the CHR kernel with all chains running in parallel. Depending on configuration, this is either:
    - `CoordinateHitAndRun()`: sample in reduced space, download, then undo whitening+rotation on the host.
    - `CoordinateHitAndRunBackmap()`: sample in reduced space, then apply the composed `T*x + shift` transformation on the GPU before download. This is used when the user wants samples in the original (pre-rounding) flux space.

12. **Download samples** to host as an Eigen matrix.

13. **Write output**: save samples as float32 `.npy` file (half the size of double, contiguous sample rows). Optionally run bounds filtering against the GEM's original flux bounds.

14. **Write provenance**: profile.json (timing breakdown), config_used.txt (full configuration snapshot), run manifest.

---

## GPU Coordinate Hit-and-Run: The Core Kernel

### Algorithm

The CHR kernel (`chrKernel` in `chr.cu`) implements a parallelized version of Coordinate Hit-and-Run for uniform sampling from a polytope `{x : Ax <= b}`. The fundamental MCMC loop is:

```
For each step t = 0 to (n_samples * thinning - 1):
    1. Choose a direction d (coordinate, pair, k-sparse, or dense)
    2. Compute the chord: find the interval [alpha_min, alpha_max] such that
       x + alpha * d remains feasible for all alpha in [alpha_min, alpha_max]
    3. Sample alpha uniformly from [alpha_min, alpha_max]
    4. Update: x <- x + alpha * d
    5. Update slack: s <- s - alpha * (A * d)
    6. If t is a thinning step, record x as a sample
```

The key to making this efficient on GPU is that step 2 (the chord computation) parallelizes across constraints: each thread processes a subset of the m constraint rows, computing the projected direction `ae = a_i . d` and the ratio `ae / slack_i` for its assigned rows. A block-wide CUB reduction then finds the global min and max across all threads.

### Kernel Architecture

- **One CUDA block per Markov chain.** Block index `bid` identifies the chain. All chains execute independently and in parallel.
- **Current position `x` lives in shared memory** (size = `cols * sizeof(double)`). This is the hot data that every thread reads during constraint evaluation.
- **Slack vector `s = b - Ax` lives in global memory**, one column per chain in a `DMatrix`. Slack is maintained incrementally: after each step, `s -= alpha * (A * d)`. This avoids recomputing the full matrix-vector product at each step.
- **Direction selection runs on thread 0 only**, because it requires sequential RNG calls. The chosen direction parameters are broadcast to all threads via shared memory and a `__syncthreads()`.
- **Constraint evaluation is fully parallel**: each thread handles rows `tid, tid + blockDim, tid + 2*blockDim, ...` of the constraint matrix. For standard coordinate moves, this is a single column read from A. For pair moves, it is two column reads combined with the rotation coefficients.

### Block-Level Reduction

The chord endpoints are computed via CUB's `BlockReduce`:

```cuda
Real aggregate_max = BlockReduce(temp_storage_max).Reduce(partial_max, cub::Max());
Real aggregate_min = BlockReduce(temp_storage_min).Reduce(partial_min, cub::Min());
```

This is a logarithmic-depth tree reduction within a single block. Each thread holds its partial min/max across its assigned constraint rows; the reduction finds the global extremes in `O(log(TPB))` steps.

The step size is then computed on thread 0:
```cuda
alpha = (1/aggregate_min) + u * ((1/aggregate_max) - (1/aggregate_min));
```
where `u ~ Uniform(0,1)` from cuRAND.

### Template Instantiation and Launch Configuration

The kernel is templated on `ThreadsPerBlock` (TPB) to allow compile-time specialization of the CUB reduction:

```cuda
template __global__ void chrKernel<double, curandState, 32>(...);
template __global__ void chrKernel<double, curandState, 64>(...);
template __global__ void chrKernel<double, curandState, 128>(...);
template __global__ void chrKernel<double, curandState, 256>(...);
template __global__ void chrKernel<double, curandState, 512>(...);
template __global__ void chrKernel<double, curandState, 1024>(...);
```

At runtime, `launchChrKernel` dispatches to the appropriate instantiation. The default is 128 threads per block; the user can override via `TPB_SS`. The number of blocks equals the number of chains (typically 16-64).

### PRNG Management

Each chain has its own cuRAND MRG32k3a state, stored in global memory. Thread 0 reads the state at the start of each step, makes all RNG calls (direction selection + step size), then writes the state back. This avoids race conditions while keeping PRNG quality. States are initialized via a dedicated kernel (`initPrngStatesCHR`) seeded with a hash of the model name, directory path, and GPU device ID -- ensuring reproducible but distinct streams across models and GPUs.

### Slack Resynchronization

Over many steps, incremental slack updates accumulate floating-point error. The optional `resync_interval` parameter triggers a full recomputation every N steps:

```cuda
if (resync_interval > 0 && (t % resync_interval) == 0) {
    for (int r = tid; r < rows; r += threadstride) {
        Real acc = 0.0;
        for (int c = 0; c < cols; ++c)
            acc += A[IDX2C(r, c, rows)] * x_s[c];
        slack[IDX2C(r, bid, rows)] = b[r] - acc;
    }
}
```

This is expensive (a full matrix-vector product per chain) but prevents drift-induced infeasibility in very long runs. In practice, it is rarely needed when the polytope is well-conditioned.

---

## Direction Selection: The Four Move Types

The CHR kernel supports four types of sampling directions, selected hierarchically at each step via a single uniform draw `u ~ U(0,1)`. The hierarchy is:

1. **Dense directions** (probability `dikin_prob`, default 0 or 0.15 when extra constraints are present)
2. **K-sparse directions** (probability `ksparse_prob`, default 0)
3. **Pair directions** (probability `pair_prob`, default 0.5 in barrier mode)
4. **Standard coordinate** (remaining probability)

### Standard Coordinate Moves

The simplest direction: pick a random axis `e_i` uniformly from `{0, ..., d-1}`. The projected direction for constraint row `r` is just `A[r, i]` -- a single element read from the constraint matrix. Position update is `x[i] += alpha`.

This is the default CHR direction. It has minimal computational overhead but can mix slowly when the polytope has strong coupling between coordinates (i.e., when moving along one axis forces you into a narrow corridor along another).

### Pair Moves (2-Sparse Rotated Directions)

Pair moves operate on two coordinates simultaneously, using a direction that is a linear combination of two coordinate vectors. There are two sub-modes:

**Mode 1 (Fixed pairs):** Direction is `(e_i - e_j) / sqrt(2)` where `i` is chosen uniformly and `j` is chosen uniformly from the remaining indices. The coefficients are `ci = 1/sqrt(2)`, `cj = -1/sqrt(2)`. This explores the off-diagonal coupling between any two coordinates.

**Mode 2 (Jacobi-rotated pairs):** Direction is drawn from a precomputed rotation schedule. For pair `k`, the two orthonormal directions in the `(e_i, e_j)` plane are:
```
u1 = c * e_i + s * e_j
u2 = -s * e_i + c * e_j
```
where `(c, s) = (cos(theta), sin(theta))` and `theta` is the Jacobi angle that diagonalizes the 2x2 block of the barrier Hessian (or empirical covariance) for coordinates `(i, j)`. One of the two directions is chosen uniformly, then sign-symmetrized.

The critical insight: after barrier rotation, the remaining coupling between pairs of coordinates is captured by the off-diagonal elements of the covariance in the rotated frame. The Jacobi angle for each pair eliminates this residual coupling. So pair moves with Jacobi angles are doing "last mile" decorrelation that coordinate moves alone would take many steps to achieve.

Pair moves cost about 2x per constraint row (two column reads from A instead of one) but can dramatically improve mixing when the polytope has strong coordinate coupling.

### K-Sparse Directions

A direction with `k` nonzero entries, each `+/- 1/sqrt(k)` with random sign:
```cuda
for t in 0..k:
    idx = random coordinate
    sign = random +/-1
    direction[idx] = sign / sqrt(k)
```

Indices are sampled with replacement (duplicates just reduce effective sparsity; this is acceptable for an "escape" move). The default `k` is 8, capped at 32 to fit in shared memory.

K-sparse directions help escape from locally constrained regions where all coordinate and pair directions are pinched. They explore subspaces that are not aligned with any single pair of axes.

### Dense Directions (Barrier Hessian Eigenvectors)

When extra constraints (KO/conditioning) are added to a base model, they can create new pinched directions that don't align with any coordinate axis. The dense direction mechanism addresses this by precomputing a set of "escape directions" from the eigenvectors of the barrier Hessian restricted to the extra constraints.

The implementation:
1. On the CPU, compute the delta-Hessian from only the extra constraint rows: `H_delta = sum_{extra rows} (1/delta_i^2) * n_i * n_i'`, where `delta_i` is the Euclidean distance to facet `i` and `n_i` is its unit normal.
2. Eigen-decompose `H_delta` and select the top-k eigenvectors where the eigenvalue exceeds `10x` the median (i.e., significantly above noise).
3. Precompute `A * V` (the projection of all constraint rows onto the escape directions) on the CPU.
4. Upload both `V` (d x k) and `A*V` (m x k) to the GPU as dense matrices.

During sampling, when a dense direction is selected:
- Thread parallelism is over constraint rows (same as coordinate), reading from the precomputed `A*V` matrix instead of a single column of `A`.
- Thread 0 updates all `d` coordinates of `x` using the corresponding column of `V`.

Default probability is 15% when extra constraints are present. Typical k is 1-32 directions.

---

## Barrier Rotation and Whitening

This is the central preconditioning innovation. Traditional CHR relies on expensive iterative rounding (e.g., PolyRound's maximum volume ellipsoid) to make the polytope "round" before sampling. Naja replaces this with an analytical transformation derived from the log-barrier Hessian, which can be computed in a single pass.

### The Log-Barrier Hessian

For a polytope `{x : Ax <= b}` with interior point `x_c`, the log-barrier function is `phi(x) = -sum_i log(b_i - a_i' x)`. Its Hessian at `x_c` is:

```
H(x_c) = A' diag(1/s^2) A
```

where `s = b - A*x_c` is the slack vector. This matrix encodes the local geometry of the polytope as seen from `x_c`: each constraint contributes a rank-1 term weighted by `1/slack^2`. Tight constraints (small slack) dominate the Hessian; loose constraints barely contribute.

The eigendecomposition `H = Q * Lambda * Q'` reveals the principal directions of constraint-induced coupling. The largest eigenvalue corresponds to the direction where constraints are tightest (most "pinched"); the smallest eigenvalue corresponds to the most open direction.

### Computing the Rotation

Implementation in `barrier_rotation.cpp`:

1. **Slack with clamping**: Compute `s = b - A*x_c`. To prevent one near-tight facet from dominating the entire Hessian, clamp all slacks to at least `tau = 1e-6 * median(positive slacks)`. Track the number of clamped entries.

2. **Hessian assembly**: Form `B = diag(1/s) * A` (shape m x d), then `H = B' * B` (shape d x d). This is an efficient rank-d factored form that avoids building the full m x m outer product.

3. **Ridge regularization**: Add `lambda * (trace(H)/d) * I` with `lambda = 1e-10`. This ensures the eigendecomposition is numerically stable even when the Hessian is nearly singular.

4. **Eigendecomposition**: `Eigen::SelfAdjointEigenSolver` gives guaranteed real eigenvalues (ascending) and an orthogonal eigenvector matrix `Q`.

5. **Apply rotation**: Transform the polytope into the eigenbasis:
   ```
   A_rotated = A * Q
   x_rotated = Q' * x_c
   ```
   If an affine hull basis exists, it is also rotated: `hull_basis = hull_basis * Q`.

Because `Q` is orthogonal, this is a pure rotation -- chord lengths and volumes are preserved. No direction is stretched or compressed; the coordinates are simply reoriented to align with the principal axes of the barrier metric.

### Whitening (Diagonal Rescaling)

After rotation, the eigenvalues of `H` in the rotated frame are `Lambda = diag(lambda_1, ..., lambda_d)`. The barrier metric in the rotated frame is `diag(lambda_i)`, which means coordinate `i` is "pinched" proportionally to `lambda_i`.

Whitening rescales each axis to equalize this:
```
scale_i = 1 / sqrt(lambda_i)
A_whitened = A_rotated * diag(scale)
x_whitened = diag(1/scale) * x_rotated
```

After whitening, the local barrier metric at `x_c` is approximately identity -- the polytope looks locally "round" from the perspective of the starting point.

**Eigenvalue clamping**: To prevent extreme scales from near-zero eigenvalues, a floor is applied: `ev_floor = max(median_eigenvalue * 1e-6, 1e-30)`. Any eigenvalue below this floor is clamped up. The number of clamped eigenvalues is logged.

**Condition number**: The barrier condition number `lambda_max / lambda_min` (before whitening) determines how anisotropic the geometry is. This directly drives the adaptive thinning logic:
- Condition <= 1e4: thin by `max(n/6, 50)`
- Condition <= 1e8: thin by 200
- Condition > 1e8: thin by 500

This replaces a fixed thinning parameter with one that adapts to the actual geometry.

### Why This Matters (vs. Traditional Rounding)

In the traditional pipeline (hopsy/PolyRound), rounding is done via iterative maximum volume ellipsoid (MVE) computation:
1. Find Chebyshev center of polytope (LP solve)
2. Solve convex optimization for maximum volume inscribed ellipsoid
3. Apply affine transformation to make polytope round
4. Repeat up to 20 iterations until eigenvalue ratio < 6

This is expensive: each iteration involves solving a convex program, and convergence is not guaranteed (the code has fallback logic for when MVE fails). It also operates as a black box -- you don't get any diagnostic information about which directions are problematic.

Naja's barrier rotation + whitening:
- Is a single-pass computation (one eigendecomposition)
- Uses ALL constraint rows (base + extra), so it captures KO-induced coupling that PolyRound's rounding (computed on the base model only) misses
- Produces a diagnostic condition number that directly informs thinning
- Integrates with the constraint epsilon (`b += eps`) which helps even out Hessian weights so no single tight constraint dominates
- Composes cleanly with the affine hull reduction and backmap transformation

The key insight is that you don't need the polytope to be globally round. You need the local geometry at the starting point to be well-conditioned, because that's where the Markov chain begins and the barrier Hessian captures exactly this local structure.

---

## Jacobi Pair Schedule Construction

The pair schedule determines which coordinate pairs are sampled together during pair moves, and at what rotation angle. Naja offers three approaches to constructing this schedule, each representing a different tradeoff between cost and quality.

### Approach 1: Barrier-Hessian Schedule (Analytical)

Implementation in `barrier_schedule.cpp`. This constructs the schedule directly from the barrier Hessian without any sampling:

1. **Build the barrier Hessian** `G = A' diag(1/s^2) A` at the starting point. This is the same matrix used for barrier rotation, but computed independently (the schedule construction operates in whatever coordinate system is current, which may already be barrier-rotated).

2. **Compute pairwise correlations**: For each pair `(i, j)`, compute the barrier-metric correlation:
   ```
   rho_ij = |G_ij| / sqrt(G_ii * G_jj)
   ```
   This is the absolute normalized off-diagonal element of the Hessian -- a measure of how strongly coupled coordinates `i` and `j` are in the barrier metric. High `rho` means these coordinates share tight constraints and will mix slowly unless sampled together.

3. **Greedy maximum-weight matching**: Sort all `d*(d-1)/2` candidate pairs by descending `rho`. Greedily match pairs: take the highest-correlation pair, mark both indices as used, repeat. This ensures the most strongly coupled pairs are prioritized.

4. **Compute Jacobi angles**: For each matched pair `(i, j)`, extract the 2x2 block `[[G_ii, G_ij], [G_ij, G_jj]]` and compute the Jacobi rotation angle:
   ```
   theta = 0.5 * atan2(2 * G_ij, G_ii - G_jj)
   (c, s) = (cos(theta), sin(theta))
   ```
   This angle diagonalizes the 2x2 block, meaning a pair move along `(c*e_i + s*e_j)` or `(-s*e_i + c*e_j)` will explore the two decorrelated principal directions in that subspace.

5. **Handle unpaired coordinates**: If `d` is odd or some coordinates had zero diagonal in `G`, remaining unpaired indices are matched arbitrarily and given their Jacobi angle from whatever coupling exists.

The result is a `PairSchedule` with vectors `i`, `j`, `c`, `s` of length `d/2`, uploaded to the GPU as `DVector<int>` and `DVector<double>`.

**Auto-enablement**: When the barrier schedule method is selected (`PAIR_SCHEDULE_METHOD=barrier`), pair probability is automatically set to 0.5 if the user hasn't specified it. This means half of all MCMC steps use rotated pair moves and half use standard coordinate moves.

### Approach 2: Runtime Warmup Schedule (Empirical)

Implementation in `runtime_warmup.cpp`. This runs short warmup chains to empirically discover the covariance structure:

1. **Initialize random pairing**: Shuffle coordinate indices and pair them sequentially. Initial rotation angles are identity (c=1, s=0).

2. **Iterative refinement** over `iter_rounding_passes` passes:
   - Run a single-chain CHR warmup on the GPU using the current schedule (pass 0 uses fixed pairs `pair_mode=1`; subsequent passes use Jacobi-rotated pairs `pair_mode=2`).
   - Download the warmup samples.
   - For each pair `(i, j)`, compute the empirical variance and covariance from the warmup samples:
     ```
     var_i = mean((x_i - mean(x_i))^2)
     var_j = mean((x_j - mean(x_j))^2)
     cov_ij = mean((x_i - mean(x_i)) * (x_j - mean(x_j)))
     ```
   - Update the Jacobi angle for each pair using `jacobi_rotation_cs(var_i, var_j, cov_ij)`.

3. **Return the refined schedule**.

This approach is more expensive than the analytical barrier schedule (it requires actual GPU sampling during warmup) but captures the true global covariance structure rather than just the local barrier geometry. It is typically used with 2 passes and a few thousand warmup steps.

### Approach 3: Pre-Computed Schedule (File)

A previously computed schedule can be loaded from a CSV file (`schedule_io.cpp`). This is used for:
- Rounding inheritance: a schedule computed on a base model can be reused for conditioned variants.
- Cross-model calibration: a schedule calibrated across multiple models (via `cmd_calibrate_rounding`) can be shared.

The CSV format has columns `i, j, c, s` with one row per pair.

### Schedule Selection Logic

In `plan.cpp`, the `build_rounding_plan()` function implements the fallback chain:

1. If `PAIR_SCHEDULE` file path is provided -> load from CSV (mode 2).
2. Else if `ITER_ROUNDING_PASSES > 0` and `ITER_ROUNDING_WARMUP > 0` and `dim >= 2` -> run runtime warmup (mode 2).
3. Else -> no schedule, `pair_mode = 1` or 0 (coordinate moves only, or fixed pairs).

The barrier schedule (`PAIR_SCHEDULE_METHOD=barrier`) is handled separately in `job_sampling.cu` before the plan builder is called, because it operates on the host-side Hessian rather than GPU warmup.

---

## Fused Back-Transformation

When samples are needed in the original (pre-rounding) flux space, naja performs the coordinate back-transformation on the GPU rather than downloading reduced-space samples and transforming on the CPU.

The back-transformation is `x_original = T * x_reduced + shift`, where `T` is the composed transformation matrix and `shift` is the composed shift vector. The composition handles multiple chained transforms:

1. **Affine hull reduction**: `T_hull = hull_basis`, `shift_hull = hull_shift`
2. **Barrier rotation**: `T_rot = Q` (orthogonal eigenvector matrix)
3. **Barrier whitening**: `T_whiten = diag(scale)` (diagonal scaling)
4. **PolyRound transform**: `T_polyround` (from the original rounding CSV)

These are composed into a single `T` and `shift`:
```cpp
if (hull_enabled) {
    T = T_polyround * hull_basis;  // hull_basis already includes rotation_Q and whiten_scale
    shift = T_polyround * hull_shift + shift_polyround;
} else if (barrier_rotated) {
    T = T_polyround * rotation_Q;
    if (barrier_whitened) T = T * whiten_scale.asDiagonal();
    shift = shift_polyround;
}
```

The fused backmap (`backmap.cu`) then applies this in two steps:
1. **cuBLAS GeMM**: `output = T * samples` (matrix-matrix multiply, the heavy operation)
2. **Custom kernel**: `output[row][col] += shift[row]` (add shift to every column)

This avoids downloading `n_chains * n_samples * reduced_dim` doubles to the host, transforming, and then storing `n_chains * n_samples * original_dim` doubles. Instead, only the final original-space samples are transferred.

---

## Double-Buffered Streaming

For very large sample counts that don't fit in GPU memory at once, `CoordinateHitAndRunStreamed()` implements a producer-consumer pipeline with double buffering:

```
Device buffers:    dev_chunk[0]    dev_chunk[1]
Pinned host bufs:  host_chunk[0]   host_chunk[1]
CUDA streams:      compute_stream  copy_stream

Chunk 0: compute on dev_chunk[0]                    (compute_stream)
Chunk 1: compute on dev_chunk[1]                    (compute_stream)
          + copy dev_chunk[0] -> host_chunk[0]      (copy_stream)
Chunk 2: compute on dev_chunk[0]                    (compute_stream)
          + copy dev_chunk[1] -> host_chunk[1]      (copy_stream)
...
```

Key details:
- **Two device buffers** (`DMatrix<double>`) of size `N x (nchains * chunk_nspc)` hold alternating chunks of samples.
- **Two pinned host buffers** (`PinnedHostBuffer`, allocated via `cudaHostAlloc`) receive the device-to-host copies. Pinned memory is required for `cudaMemcpyAsync` to actually be asynchronous.
- **CUDA events** synchronize between streams: the copy stream waits for the compute event before copying, ensuring samples are fully computed before transfer begins.
- **`persist_state = true`**: Between chunks, the kernel writes the final position of each chain back to `X0` in device memory, so the next chunk starts from where the previous chunk ended. This is critical for chain continuity.
- **Host sink callback**: After each chunk's copy completes, a user-provided callback processes the host-side data (e.g., writing to disk). This allows truly unbounded sample counts with bounded GPU memory.

---

## Multi-GPU Bulk Execution

The bulk mode (`job_bulk.cpp`) is designed for the "many models" regime: hundreds or thousands of conditioned models that each need independent sampling. This is the production use case for metabolic modeling, where each tissue sample or experimental condition produces a different set of extra constraints.

### Work-Stealing Architecture

```cpp
std::atomic<size_t> next_index{0};
std::vector<std::thread> workers;

for (int device_id : gpu_ids) {
    workers.emplace_back(bulk_worker, device_id, ...);
}

void bulk_worker(int device_id, ...) {
    naja::gpu::set_device(device_id);
    while (true) {
        size_t idx = next_index.fetch_add(1);  // atomic increment
        if (idx >= jobs.size()) break;
        // ... set up and run this model ...
        run_sampling_job(job_cfg, false, false);
    }
}
```

Each GPU gets one worker thread. Workers atomically claim the next unprocessed model from a shared counter. This naturally load-balances across GPUs: if one model takes longer (e.g., higher dimension, more constraints), other GPUs don't sit idle waiting.

### GPU Context Switching ("Hotswapping")

When a worker finishes one model and starts the next, the full GPU context from the previous job is already cleaned up by RAII destructors:
- `DMatrix` destructor calls `cudaFree`
- `DVector` destructor calls `cudaFree`
- `PRNGState` destructor frees the curand state array
- `CUBLASHandle` destructor destroys the cuBLAS context

The new model then allocates fresh GPU memory for its own `A`, `b`, `X0`, slack, and samples. There is no explicit "GPU clear" step -- the C++ destructor chain handles everything.

The `set_device(device_id)` call at the start of each worker thread binds the CUDA context to that thread for its lifetime. All subsequent CUDA operations (allocations, kernel launches, memcpy) automatically target the correct GPU.

### Skip-Existing Logic

In long bulk runs that may be interrupted and resumed, the `SKIP_EXISTING` flag enables intelligent resumption:
- **Nested output mode**: Check if `<out_dir>/<model_name>/*/samples.npy` and `profile.json` exist. Find the most recently modified completed run directory.
- **Flat output mode**: Check if `<flat_dir>/<model_name>.npy` is a non-empty file.

If a completed run is found, the model is marked as `SKIP` with zero elapsed time and the worker immediately moves to the next job.

### Flat Output Mode

For downstream pipelines that want a simple directory of `.npy` files (one per model), the `FLAT_OUTPUT_DIR` option writes samples directly as `<dir>/<model>.npy`. Metadata (profile, config snapshots) goes to a separate `.meta/<model>/` directory.

To prevent partial files from being visible to downstream consumers, writing uses **atomic rename**: samples are first written to a temporary file `<model>.npy.tmp.<pid>.<gpu_id>`, then atomically renamed to the final path via `rename(2)`. On failure, the temp file is cleaned up.

### Summary Output

After all workers complete, a `bulk_summary.csv` is written with columns:
```
model, status, elapsed_s, device_id, message, output_dir, inherited_from_base
```

This provides a complete audit trail of which models succeeded, failed, or were skipped, how long each took, which GPU processed it, and whether its rounding was inherited from a base model.

---

## Rounding Inheritance and Model Preparation

In the "many models" workflow, all conditioned models share a common base model. The base model's PolyRound rounding (the `T`, `shift`, reduced `A`, `b`, `x0` matrices) can be reused for conditioned variants, with only the extra constraint rows differing.

### Inheritance Mechanism

`inherit.cpp` implements `inherit_rounding_impl()`:

1. **Copy or symlink** the base model's rounding directory into the conditioned model's directory. Symlinks save disk space when hundreds of models share the same base.
2. **Normalize extra constraints**: If the conditioned model has its own extra constraints, these must be expressed in the base model's rounded coordinate system. The normalization computes `A_extra_rounded = A_extra_original * T_inv` and adjusts `b_extra` accordingly.
3. **Recompute starting point**: If extra constraints are present and the base model's starting point is no longer feasible, solve a new inscribed-cube LP (`feasible_start_lp.cpp`) using Gurobi to find a feasible interior point for the augmented polytope.
4. **Write provenance**: An `INHERITED_FROM.txt` file records which base model the rounding came from.

### Batch Preparation

`cmd_prepare.cpp` automates the preparation of many models at once:
- For each model in a list, verify the directory structure matches the model contract.
- Inherit rounding from the base model.
- Ensure a feasible starting point exists.
- Optionally pre-compute pair schedules.

This is typically run once before a bulk sampling job.

### Cross-Model Schedule Calibration

`cmd_calibrate_rounding.cpp` computes a single pair schedule that works well across multiple models:
1. For each model in the calibration set, run short warmup chains.
2. Pool the per-pair covariance estimates across models.
3. Compute Jacobi angles from the pooled covariance.
4. Write the shared schedule CSV.

This is useful when all conditioned models have similar constraint structure (same base, different extra rows) and you want to avoid per-model schedule computation.

---

## Adaptive Thinning

Thinning (the number of MCMC steps between recorded samples) controls the tradeoff between sample quality and throughput. Too little thinning produces correlated samples; too much wastes GPU cycles.

Naja supports three thinning modes:

1. **Explicit**: User sets `THINNING=N` directly.
2. **Dimension-based default**: `thinning = max(n/6, 1)` where `n` is the reduced dimension. This is a simple heuristic that assumes mixing time scales linearly with dimension.
3. **Barrier-adaptive**: When barrier rotation is enabled, the barrier condition number drives the thinning:
   ```cpp
   if (barrier_condition <= 1e4)       thinning = max(n / 6, 50);
   else if (barrier_condition <= 1e8)  thinning = 200;
   else                                thinning = 500;
   ```

The barrier-adaptive mode is the recommended default. The condition number is a direct measure of how anisotropic the geometry is: a well-conditioned polytope (condition ~ 1e2) needs minimal thinning, while a poorly conditioned one (condition ~ 1e10) needs aggressive thinning even after whitening.

---

## Post-Sampling Validation: Bounds Filtering

For metabolic models, samples must satisfy the original GEM (genome-scale metabolic model) flux bounds, not just the relaxed polytope constraints. The bounds filter (`bounds_filter.cpp`) checks each sample against the model's lower and upper flux bounds and produces:

- `valid_mask.npy`: boolean mask of which samples pass.
- `valid_fraction.txt`: scalar fraction of valid samples.
- `bounds_report.json`: per-reaction violation statistics.
- Optionally `samples_valid.npy`: only the passing samples.

The bounds tolerance is configurable via `BOUNDS_EPS` (default 1e-6).

---

## Comparison with Traditional CHRR (Hopsy)

Hopsy implements the traditional Coordinate Hit-and-Run with Rounding (CHRR) pipeline for polytope sampling. Key differences:

### Rounding Strategy

| Aspect | Hopsy | Naja |
|--------|-------|------|
| Method | Iterative MVE (up to 20 iterations) | Single-pass barrier Hessian eigendecomposition |
| Scope | Base model only (before conditioning) | Full augmented polytope (base + extra constraints) |
| Failure mode | Falls back to identity transform | Always produces a rotation (ridge-stabilized) |
| Diagnostic | Eigenvalue ratio check (< 6 to converge) | Condition number drives adaptive thinning |
| Cost | Multiple LP + convex optimization solves | One matrix-matrix product + eigendecomposition |

### Sampling Architecture

| Aspect | Hopsy | Naja |
|--------|-------|------|
| Compute | CPU, single-threaded per chain | GPU, all chains in parallel |
| Parallelism | Python multiprocessing (one process per chain) | One CUDA block per chain, threads parallel over constraints |
| Direction | Full Hit-and-Run (dense random direction) | Coordinate + pair + k-sparse + dense escape directions |
| Constraint eval | Sequential dot product (O(m*d) per step) | Parallel over constraints (O(m/TPB * d) per step per thread) |
| Thinning | Fixed by user | Adaptive from barrier condition number |
| Memory model | Each process has full copy of polytope | Single GPU copy, shared across all chains |

### Multi-Model Execution

| Aspect | Hopsy | Naja |
|--------|-------|------|
| Batch mode | External scripting (for loop) | Built-in bulk mode with work-stealing |
| Multi-GPU | Not supported | Native: one worker thread per GPU, atomic job dispatch |
| Skip logic | External | Built-in skip-existing |
| Provenance | External | Built-in manifests and summary CSV |

### Direction Diversity

Hopsy's Hit-and-Run uses dense random directions (sampled from a standard normal), which explore the full d-dimensional space at each step. This has the advantage of being isotropic but the disadvantage of being expensive per step (O(m*d) for each constraint evaluation).

Naja's CHR uses sparse directions (1 or 2 nonzero entries for coordinate/pair moves) which are much cheaper per step (O(m) for coordinate, O(2m) for pair) but explore a lower-dimensional subspace at each step. The tradeoff is compensated by:
- Much higher throughput (more steps per second on GPU)
- Pair moves with Jacobi angles that target the specific coupling structure
- K-sparse and dense escape directions for occasional full-space exploration

### The Net Effect

In practice, naja achieves orders-of-magnitude higher throughput than hopsy for the same sample quality, primarily because:
1. GPU parallelism over chains and constraints
2. Sparse directions enable much cheaper per-step cost
3. Barrier rotation + whitening eliminates the most expensive part of the traditional pipeline (iterative rounding)
4. The bulk execution engine eliminates per-model overhead

---

## Configuration Reference

All parameters are set via a key=value config file:

### Core Sampling Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_CHAINS` | 16 | Number of parallel Markov chains |
| `N_SAMPLES` | 10000 | Samples per chain |
| `THINNING` | 0 (auto) | Steps between saved samples. 0 = adaptive |
| `TPB_SS` | 128 | CUDA threads per block for CHR kernel |
| `GPU_DEVICE` | 0 | GPU device index for single-model mode |
| `BACK_TRANSFORM` | true | Apply T*x+shift on GPU to get original-space samples |

### Direction Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `PAIR_PROB` | 0.0 | Probability of pair moves per step (auto-set to 0.5 in barrier mode) |
| `KSPARSE_PROB` | 0.0 | Probability of k-sparse moves per step |
| `KSPARSE_K` | 8 | Number of nonzeros in k-sparse directions |
| `PAIR_SCHEDULE_METHOD` | "" | "barrier" for analytical Hessian schedule |
| `PAIR_SCHEDULE` | "" | Path to pre-computed pair schedule CSV |

### Rounding Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `BARRIER_ROTATE` | false | Rotate to eigenbasis of barrier Hessian |
| `BARRIER_WHITEN` | false | Diagonal rescale by 1/sqrt(eigenvalue) after rotation |
| `ITER_ROUNDING_PASSES` | 0 | Number of warmup passes for empirical schedule |
| `ITER_ROUNDING_WARMUP` | 0 | Warmup steps per pass |
| `RESYNC_INTERVAL` | 0 | Steps between slack resynchronization (0 = disabled) |

### Polytope Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONSTRAINT_EPS` | 0.0 | Global bound inflation: b += eps |
| `EXTRA_CONSTRAINT_EPS` | 0.0 | Extra constraint relaxation |
| `EXTRA_CONSTRAINTS` | "auto" | Mode: auto/ignore/require |
| `AFFINE_HULL_TOL` | 0.0 | Tolerance for detecting tight (equality) constraints |
| `START_POLICY` | "file" | "file" or "cube_center" (Gurobi LP) |

### Bulk Mode Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `GPU_LIST` | "" | Comma-separated GPU IDs for bulk mode |
| `BULK_MODEL_LIST` | "" | Path to file listing model names (one per line) |
| `SKIP_EXISTING` | false | Skip models with completed outputs |
| `FLAT_OUTPUT_DIR` | "" | Write samples as flat <model>.npy files |

### Output Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `WRITE_DATA` | true | Write samples.npy |
| `BOUNDS_POLICY` | "ignore" | "filter" to check GEM bounds post-sampling |
| `BOUNDS_EPS` | 1e-6 | Tolerance for bounds filtering |
| `WRITE_SAMPLES_VALID` | false | Also write filtered samples_valid.npy |

---

## Build System

**Languages**: C++17 (host **and** device — device code was C++14; raised to
C++17 for CUDA 13, which no longer accepts C++14).
**GPU target**: portable via `NAJA_CUDA_ARCH` (`CMAKE_CUDA_ARCHITECTURES`),
default `native`. Verified: A100 sm_80, H100 sm_90, B300 sm_100/103. Requires
CUDA ≥12.8 for Blackwell; **CUDA 13 dropped Volta sm_70.** CMake ≥3.30.
**Compiler flags**: `--expt-relaxed-constexpr --use_fast_math -O3`

**Dependencies** (vendored under `extern/`: eigen3, kissfft, gurobi):
- **Eigen3** (3.4, header-only): host dense linear algebra / eigendecomposition
- **Gurobi 13.0**: LP solver (inscribed-cube starting point); license needed at
  runtime only (`GRB_LICENSE_FILE`), not to build
- **kissfft**: FFT (built from vendored C sources)
- **CUB** (CCCL): block-level reductions in the CHR kernel. Note: CUDA 13 removed
  the `cub::Max`/`cub::Min` functors — naja uses local Max/Min ops instead.
- **cuBLAS / cuSOLVER / cuRAND**: GPU GEMM/backmap, dense eigensolver, MRG32k3a RNG
- **zlib**: `.npz` I/O
- (Arrow/volesti/indicators were vendored but unused, and have been removed.)

See `docker/README.md` for the containerized, cross-hardware build.

**Build targets**:
- Main executable: `naja` (~50 source files)
- 13 unit test executables (CPU-side logic)
- 5 GPU test executables (require CUDA device, gated by `NAJA_ENABLE_GPU_TESTS`)

---

## Model Directory Convention

Each model lives in a directory under `DATA_DIR`:

```
<DATA_DIR>/<MODEL_NAME>/
    rounding/
        <MODEL_NAME>_rounding_A.csv       # Reduced constraint matrix (m x n)
        <MODEL_NAME>_rounding_b.csv       # Reduced RHS vector (m)
        <MODEL_NAME>_rounding_start.csv   # Feasible starting point (n)
        <MODEL_NAME>_rounding_T.csv       # Back-transformation matrix (n_orig x n)
        <MODEL_NAME>_rounding_shift.csv   # Back-transformation shift (n_orig)
        <MODEL_NAME>_rounding_extra_A.csv # Optional extra constraint rows
        <MODEL_NAME>_rounding_extra_b.csv # Optional extra constraint RHS
        pair_schedule.csv                  # Optional pre-computed pair schedule
        INHERITED_FROM.txt                 # Optional provenance from base model
    gem/
        l_bounds.csv                       # Original GEM lower bounds (for bounds filter)
        u_bounds.csv                       # Original GEM upper bounds
```

The `model_contract.cpp` module validates this structure before sampling, ensuring all required files exist and have consistent dimensions.

---

## Test Suite

19 test files covering unit and integration testing:

**CPU unit tests** (13 files):
- CSV loader, E-Flux smoke test, GPR expression parsing, E-Flux2 bounds
- Start feasibility, CHR pair step logic, Jacobi rotation
- Pair schedule I/O, extra constraint epsilon/mode
- Gurobi LP solver (requires Gurobi license)

**GPU tests** (5 files, require CUDA device):
- Sampling feasibility: all samples satisfy `Ax <= b`
- Unit cube moments: verify uniform distribution on `[0,1]^d`
- Simplex moments: verify uniform distribution on standard simplex
- Backmap correctness: verify `T*x + shift` transformation
- Thin polytope: verify sampling on highly constrained geometry

**Integration test** (1 file):
- CLI dry-run: verify command parsing and model validation without GPU

---

## Output Files

Each sampling run produces:

| File | Format | Contents |
|------|--------|----------|
| `samples.npy` | NumPy float32 | Shape (n_chains * n_samples, dim). Samples in original space (if BACK_TRANSFORM) or reduced space. |
| `profile.json` | JSON | Timing: load_time, upload_time, sampling_time, download_time, write_time, total_time. Also: throughput (samples/s), dimensions, chain/sample counts. |
| `config_used.txt` | Text | Full config snapshot with derived paths and metadata. |
| `valid_mask.npy` | NumPy bool (optional) | Per-sample GEM bounds validity. |
| `valid_fraction.txt` | Text (optional) | Scalar fraction of valid samples. |
| `bounds_report.json` | JSON (optional) | Per-reaction violation statistics. |

---

## Observed Performance

The following numbers are from a production run on a genome-scale metabolic model (RECON-family) with 8,074 conditioned variants, sampled on 8 NVIDIA A100 GPUs.

### Per-Condition Timing (50,000 samples, 582 dimensions)

| Phase | Median | Mean |
|-------|--------|------|
| GPU sampling | 14.5s | 15.3s |
| NFS write | 1.8s | 2.5s |
| Total wall time | 28.0s | 27.4s |

Throughput: ~3,400 samples/sec per GPU.

Time budget per condition: 56% GPU compute, 9% NFS write, 35% overhead (polytope load, barrier rotation, Gurobi LP for start point, GPU upload/download).

The overhead fraction is dominated by host-side operations that run once per model: loading the CSV constraint matrices from NFS, computing the barrier Hessian eigendecomposition (a d x d dense eigensolve), optionally solving a Gurobi LP for the inscribed-cube starting point, and uploading the matrices to the GPU. For a 582-dimensional model with ~3,000 constraints, this totals roughly 10 seconds. The GPU sampling phase itself -- the actual MCMC kernel execution -- takes about 14.5 seconds for 50K samples with adaptive thinning.

### Full Run (8,074 conditions across 8 GPUs)

| Metric | Value |
|--------|-------|
| Wall clock | ~7.7 hours |
| GPU-hours | 34.2 |
| Total samples | 403.7 million |
| Total data | 940 GB |
| Effective rate | ~1,000 conditions/hr |
| Effective rate | ~14.6 million samples/hr |

The near-linear scaling across 8 GPUs (7.7 hours wall = 61.6 GPU-hours theoretical maximum, vs 34.2 GPU-hours actual = 55% GPU utilization) reflects the overhead-dominated time budget: GPUs spend roughly half their wall time waiting for host-side preparation. The work-stealing scheduler keeps all GPUs busy despite variable per-model execution times -- faster models finish and immediately claim the next job.

At 940 GB of output for 8,074 models, the average per-model output is ~117 MB (50,000 samples x 582 dimensions x 4 bytes/float32). NFS write bandwidth is the secondary bottleneck at scale.

---

## Key Design Decisions and Their Rationale

### Why Coordinate Hit-and-Run (not full Hit-and-Run)?

Full Hit-and-Run samples a dense random direction at each step, which costs O(m*d) per constraint evaluation. CHR samples along a coordinate axis, costing O(m) per evaluation. For d=2000 (typical for genome-scale models), CHR is ~2000x cheaper per step. The mixing time difference is compensated by running many more steps (enabled by GPU throughput) and by using pair moves to address the most problematic coupling.

### Why barrier Hessian (not covariance)?

The empirical covariance requires running a Markov chain to estimate, which is a chicken-and-egg problem: you need good mixing to estimate covariance, but you need covariance to achieve good mixing. The barrier Hessian is available analytically from the constraint matrix and starting point -- no sampling required. It captures the local geometry of the constraint set, which is exactly what determines mixing behavior near the starting point.

### Why Jacobi pairs (not full rotation)?

A full d x d rotation would turn coordinate moves into dense directions, losing the O(m) per-step advantage. Jacobi pairs maintain 2-sparsity while targeting the specific pairs of coordinates with the strongest coupling. This preserves the computational advantage of CHR while addressing its main weakness (slow mixing on coupled coordinates).

### Why one block per chain (not one thread per chain)?

The constraint evaluation for a single MCMC step requires a reduction over m constraint rows. With one block per chain, this reduction is a fast intra-block CUB operation. With one thread per chain, you would need atomic operations or multi-pass global reductions, which are much slower. The block-per-chain design also naturally partitions shared memory and avoids inter-block synchronization.

### Why RAII for GPU memory (not manual management)?

Each sampling job allocates and frees many GPU buffers (A, b, X0, slack, samples, PRNG states, pair schedule vectors). RAII ensures cleanup happens automatically when the `DMatrix`/`DVector` objects go out of scope, even on exception paths. This is critical for the bulk worker loop, where a failed job must not leak GPU memory that would corrupt subsequent jobs on the same GPU.

### Why atomic rename for flat output?

In bulk mode with flat output, downstream consumers may be monitoring the output directory for new `.npy` files. Writing directly to the final path would expose partial files. Atomic rename via `rename(2)` ensures the file transitions from nonexistent to complete in a single filesystem operation -- no partial state is ever visible.

### Why float32 output (not float64)?

Sampling is done in double precision throughout the kernel. But for storage, samples are downcast to float32, halving the output file size. For metabolic flux analysis, float32 precision (~7 significant digits) is more than sufficient for downstream statistical analysis. The downcast happens at write time, after all computation is complete.
