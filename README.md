# Naja

GPU-accelerated polytope sampling for genome-scale metabolic models.

Naja implements coordinate hit-and-run (CHR) on CUDA with built-in support for model conditioning, polytope rounding, multi-chain sampling, and quality diagnostics. It is designed for large-scale metabolic flux sampling where CPU-based approaches are intractable.

## Quick Start

## Building

**See [`BUILDING.md`](BUILDING.md) for the full guide** (prerequisites, native &
containerized builds, the GPU/CUDA support matrix, verification, and porting
notes). In short — naja builds portably on A100 / H100 / B300; the GPU target is
auto-detected (`NAJA_CUDA_ARCH=native`, the default), and Eigen/kissfft/Gurobi
are vendored under `extern/`:

```bash
./scripts/build.sh                                   # native (auto-detect) -> build/naja
NAJA_CUDA_ARCH='80;90;100;103' ./scripts/build.sh    # portable fat binary (CUDA 13)
```

Containerized / cross-cluster builds (recommended) are in
[`docker/README.md`](docker/README.md). Requires CUDA **≥12.8** for Blackwell
(CUDA 13 dropped Volta `sm_70`) and CMake ≥3.30. Gurobi's LP feasible-start needs
a valid license (`GRB_LICENSE_FILE`) only at **runtime**, not to build.

---

## Quick Start

Build naja, then sample one model and validate the result:

```bash
./build/naja sample run \
  --model-dir models/ko_b0026 \
  --out-root out/ \
  --gpu 0 --n-chains 4 --n-samples 12500 --write-npy

./build/naja validate --samples out/ko_b0026_20260221_001/samples.npy --n-chains 4
```

Output: `samples.npy` as `(n_samples, dim)` float32, C-contiguous. Each row is one flux sample vector.

## Commands

### `naja sample run` — Sample one model

```bash
naja sample run \
  --model-dir models/ko_b0026 \
  --out-root out/ \
  --gpu 0 --n-chains 4 --n-samples 12500
```

Key flags (most have sensible defaults):

| Flag | Default | Description |
|------|---------|-------------|
| `--model-dir` | required | Path to model directory (must have `rounding/` and `gem/`) |
| `--out-root` | required | Output directory root |
| `--gpu` | `0` | GPU device index |
| `--n-chains` | required | Number of independent Markov chains |
| `--n-samples` | required | Samples per chain |
| `--thinning` | `dim/6` | Steps between saved samples |
| `--pair-prob` | `0.0` | Fraction of pair-direction steps (0.3 recommended) |
| `--start-policy` | `file` | Start point: `file` (from rounding) or `cube_center` (LP) |
| `--constraint-eps` | `0.0` | Global constraint relaxation |
| `--extra-constraints` | `auto` | How to handle extra constraint files: `auto`, `ignore`, `require` |
| `--backmap` | off | Back-transform to original reaction space via T matrix |
| `--write-npy` | off | Write samples.npy |
| `--bounds-policy` | `ignore` | `ignore` or `filter` (check against gem bounds) |
| `--verbose` | off | Print detailed config and timing |
| `--quiet` | off | Suppress all status output |
| `--dry-run` | off | Validate setup without sampling |

### `naja sample prepare` — Prepare models for sampling

Inherits rounding from a base model, builds extra constraints from conditioned bounds, and computes feasible start points (parallel LP, fast-path skip).

```bash
naja sample prepare \
  --models-root models/ \
  --model-list jobs.txt \
  --base-model-dir base_model/ \
  --mode symlink
```

### `naja sample bulk` — Multi-GPU batch sampling

```bash
naja sample bulk \
  --models-root models/ \
  --model-list jobs.txt \
  --out-root out/ \
  --name my_run \
  --gpus 0,1,2,3 \
  --n-chains 4 --n-samples 12500 --write-npy
```

### `naja condition eflux` — E-Flux expression conditioning

```bash
naja condition eflux \
  --base-model-dir base_model/ \
  --out-model-dir conditioned/ko_arca/ \
  --row-id arca_glucose \
  --reaction-scores scores.csv
```

### `naja validate` — Sample quality diagnostics

Computes ESS (effective sample size), split-R-hat, bounds compliance, and chord lengths. Outputs JSON to stdout, one-line verdict to stderr.

```bash
naja validate \
  --samples out/ko_b0026.npy \
  --model-dir models/ko_b0026 \
  --n-chains 4
```

```
[WARN] ESS min=4 p10=9 med=162  R-hat med=1.095 max=3.065 >1.1=279 >1.2=106  bounds=0/2583000
```

## Model Directory Layout

Each model directory follows this contract:

```
models/ko_b0026/
  rounding/
    ko_b0026_rounding_A.csv        # inequality matrix (m x d)
    ko_b0026_rounding_b.csv        # inequality RHS (m x 1)
    ko_b0026_rounding_start.csv    # feasible start point (d x 1)
    ko_b0026_rounding_T.csv        # backmap matrix (n x d)
    ko_b0026_rounding_shift.csv    # backmap shift (n x 1)
    ko_b0026_rounding_extra_A.csv  # optional: extra constraints from conditioning
    ko_b0026_rounding_extra_b.csv  # optional: extra constraint RHS
  gem/
    reaction_ids.txt               # reaction names (n lines)
    l_bounds.csv                   # lower bounds (n x 1)
    u_bounds.csv                   # upper bounds (n x 1)
```

The `rounding/` files define the reduced-space polytope `Ay <= b` and the backmap `v = Ty + shift` to original reaction space. Base rounding files can be symlinked from a shared base model; only the extra constraints and gem bounds differ per condition. The preferred compact layout uses `rounding/polytope.npz`; both it and the legacy CSV layout are supported. See [the model-format reference](docs/model-format.md) for the complete contract and conversion workflow.

## Architecture

```
src/
  engine/       — core sampling orchestration (job_sampling.cu, job_bulk.cpp)
  rounding/     — Jacobi pair rotations, schedule IO, Dikin preconditioner
  pipeline/     — model contracts, extra constraints, feasible start LP
  conditioning/ — E-Flux / E-Flux2 expression conditioning
  validate/     — ESS, split-R-hat, bounds compliance, chord lengths
  gpu/          — CUDA CHR kernels, DMatrix/DVector, backmap
  cli/          — command dispatch and argument parsing
  util/         — shared utilities (filesystem, status logging, CSV/NPY IO)
```

See `ARCHITECTURE.md` for a comprehensive Mermaid diagram of the full system.

## Key Features

- **GPU CHR sampling** with pair-direction moves for better mixing in tilted polytopes
- **Dikin preconditioner** for analytic corrective rounding when extra constraints pinch the base polytope
- **Parallel model preparation** with fast-path LP skip (16 concurrent Gurobi workers)
- **Float32 contiguous output** — `(n_samples, dim)` layout for efficient downstream access
- **Built-in validation** via `naja validate` with ESS, split-R-hat, and bounds checking
- **Degenerate polytope detection** — warns when tight constraints span the full dimension

## Dependencies

- CUDA Toolkit (CUDA 13.2 verified; CUDA 12.8+ needed for Blackwell)
- Eigen3 headers (install separately)
- Gurobi 13 development files (for feasible-start LP; install separately)
- kissfft (vendored in `extern/kissfft/`, for ESS autocovariance)

See [BUILDING.md](BUILDING.md) for supported GPU architectures and exact dependency discovery settings.

## Tests

```bash
# CPU tests (always available)
ctest --test-dir build --output-on-failure

# GPU correctness tests (requires CUDA device)
cmake -S . -B build -DNAJA_ENABLE_GPU_TESTS=ON ...
ctest --test-dir build -R test_gpu
```

GPU tests verify sampling feasibility, moment correctness (hypercube, simplex), backmap accuracy, and thin-polytope handling.
