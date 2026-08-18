# Building naja

naja is a CUDA C++ project (a GPU CHRR polytope sampler). This guide covers
building it **portably across NVIDIA GPUs** — A100, H100, and B300 (Blackwell) —
either natively or in a container. Historically naja built on only one cluster;
the steps below are hardware- and cluster-agnostic.

> **Verified:** builds and passes all GPU correctness tests on **B300 (sm_103) /
> CUDA 13.2** via the container flow below, and the CUDA architecture is now
> auto-detected per host, so A100 (sm_80) and H100 (sm_90) work with no edits.

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| **CUDA Toolkit** (nvcc + `cudart`, `curand`, `cublas`, `cusolver`) | **≥ 12.8** for Blackwell (B300). **CUDA 13 removed Volta `sm_70`.** |
| **CMake ≥ 3.30** | needed to emit Blackwell `sm_100/103`; older CMake won't know the arch |
| **Ninja** (optional) | faster builds; `build.sh` uses it if present |
| **zlib** (`zlib1g-dev`) | `.npz` I/O |
| C++17 host compiler | e.g. gcc within your CUDA version's supported range |
| Eigen 3.4 headers | install with your package manager, or set `EIGEN3_ROOT` |
| Gurobi 13.0 development files | proprietary; install separately and set `GUROBI_HOME` |

**kissfft** is vendored under `extern/` and built from source. Eigen and Gurobi
are intentionally not distributed in this repository. CMake looks in the
repository's legacy `extern/` locations first, then in `EIGEN3_ROOT` and
`GUROBI_HOME`. You can override either location explicitly with
`-DNAJA_EIGEN_ROOT=/path/to/eigen3` and `-DNAJA_GUROBI_ROOT=/path/to/gurobi`.

## 2. Choosing the GPU architecture

The target is set by `NAJA_CUDA_ARCH` (a `CMAKE_CUDA_ARCHITECTURES` value):

| GPU | `NAJA_CUDA_ARCH` |
|---|---|
| A100 | `80` |
| H100 | `90` |
| B300 (Blackwell) | `100` or `103` |
| **auto-detect the build host's GPU** (default) | `native` |
| **portable fat binary** (runs on all of the above, needs CUDA 13) | `80;90;100;103` |

`native` is the default and is correct when you build on the same GPU you'll run
on. Use the explicit list for a single binary that runs on mixed hardware.

## 3. Native build (on a machine with the CUDA toolkit + a GPU)

```bash
./scripts/build.sh                                   # native (auto-detect), -> build/naja
NAJA_CUDA_ARCH='80;90;100;103' ./scripts/build.sh    # portable fat binary
./scripts/build.sh clean                             # wipe build/ first
```

## 4. Containerized build (recommended; cross-cluster)

The container is the reproducible, self-documenting build. See
[`docker/README.md`](../docker/README.md) for full detail. Two flows:

**a) Dedicated image (public artifact — builds a portable fat binary, no GPU
needed to build):**
```bash
docker build -f docker/Dockerfile.naja -t naja:latest .
enroot import -o naja.sqsh dockerd://naja:latest        # for Slurm+enroot clusters
```

**b) Build inside an existing CUDA-13 container** (e.g. an NGC image already on
your cluster). This is what the Slurm helpers do — they install cmake/ninja/zlib
then run `native`:
```bash
sbatch scripts/run_b300_naja_build.sbatch               # Slurm + enroot, B300
# or interactively inside any CUDA-devel container:
NAJA_CUDA_ARCH=native ./scripts/build_container.sh
```

## 5. Running & the Gurobi license

The LP feasible-start (`src/pipeline/feasible_start_lp.cpp`) requires a valid
Gurobi 13.x license **at runtime**. Point `GRB_LICENSE_FILE` at it (or place
`~/gurobi.lic`). There is no hardcoded license path. The Gurobi development
libraries are required to build the current executable, but a license is not.

```bash
export GRB_LICENSE_FILE=/path/to/gurobi.lic
./build/naja sample run --model-dir <dir> --out-root <dir> --gpu 0 --n-chains 4 --n-samples 5000
```

## 6. Verifying the build

```bash
# CLI + IO, no GPU/license needed:
bash tests/test_sample_cli_dry_run.sh                   # -> exercises sample run/bulk --dry-run

# GPU kernel correctness (on-device generation + moment checks) — the real
# per-hardware proof; build with the tests enabled and run them:
sbatch scripts/run_b300_naja_gputest.sbatch             # ctest -R gpu  (5 tests)
```
All five GPU tests (feasibility, cube/simplex moments, backmap, thin polytope)
pass on B300. Because `--use_fast_math` changes FP reciprocal/FMA paths per
architecture, re-run these when moving to a new GPU and expect any golden/
reference samples to be re-baselined per hardware.

## 7. Troubleshooting / porting notes

- **`namespace "cub" has no member "Max"/"Min"`** — CUDA 13's CCCL removed those
  functors; naja now uses local Max/Min ops (`src/gpu/chr.cu`). If you see it,
  you're on an older tree.
- **`Unsupported gpu architecture 'compute_70'`** — CUDA 13 dropped Volta; don't
  target `sm_70`. Use `native` or `80;90;100;103`.
- **CMake rejects `sm_103`** — CMake too old; install `cmake>=3.30` (the
  container does this via pip).
- **Gurobi not found** — set `GUROBI_HOME` to the Gurobi installation root, or
  pass `-DNAJA_GUROBI_ROOT=/path/to/gurobi`. The build expects Gurobi 13's
  `libgurobi130` and `gurobi_c++` development libraries.
