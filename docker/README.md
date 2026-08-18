# Building & running naja in a container (portable across GPUs)

naja is a CUDA C++ project. Historically it built only on one CSAIL cluster;
this container recipe makes it build and run on **any** modern NVIDIA GPU —
A100, H100, B300 — with a single source of truth for the toolchain.

## What naja actually needs (audited)

| Dependency | Source | Notes |
|---|---|---|
| CUDA toolkit (nvcc, cudart, curand, cublas, cusolver) | CUDA `-devel` base image | **CUDA ≥ 12.8** for Blackwell (B300). CUDA 13 **dropped Volta sm_70.** |
| CMake ≥ 3.30 | pip (in image) | needed to emit Blackwell `sm_100/103` |
| Ninja | pip/apt | faster CUDA builds |
| zlib (`zlib1g-dev`) | apt | `.npz` I/O (`src/npz.cpp`) |
| Eigen 3.4 | `libeigen3-dev` in the image | header-only |
| kissfft | vendored `extern/kissfft` | built from C sources |
| Gurobi 13.0 | stage at `extern/gurobi/linux64` before the image build | proprietary development files; a license is needed only to *run* the LP feasible-start |

No BLAS/LAPACK or Arrow dependencies are needed.

Before building the image, copy a locally installed Gurobi 13 distribution into
the ignored `extern/gurobi/linux64` path. This keeps proprietary files out of
Git while making them available to Docker's build context:

```bash
mkdir -p extern/gurobi
cp -a "$GUROBI_HOME" extern/gurobi/linux64
docker build -f docker/Dockerfile.naja -t naja:latest .
```

## GPU / arch support matrix

`naja` selects its GPU target via `NAJA_CUDA_ARCH` (a `CMAKE_CUDA_ARCHITECTURES`
value):

| GPU | compute cap | `NAJA_CUDA_ARCH` | min CUDA |
|---|---|---|---|
| A100 | sm_80 | `80` | 11.0 |
| H100 | sm_90 | `90` | 12.0 |
| B300 (Blackwell) | sm_100 / sm_103 | `100` or `103` | 12.8 / 13.0 |
| **portable fat binary** | all of the above | `80;90;100;103` | 13.0 |
| **auto-detect (build on target)** | build host's GPU | `native` | needs a visible GPU at configure time |

## Option A — dedicated image (public-release artifact)

Builds a **portable fat binary** (runs on A100/H100/B300), no GPU needed to build:

```bash
docker build -f docker/Dockerfile.naja -t naja:latest .
# single-arch (smaller): --build-arg NAJA_CUDA_ARCH=90
# older cluster: --build-arg CUDA_TAG=12.6.3-devel-ubuntu24.04 --build-arg NAJA_CUDA_ARCH="80;90"
```

Run on a Slurm+enroot cluster by importing the image to a squashfs:

```bash
# from a machine with docker + enroot:
enroot import -o naja.sqsh dockerd://naja:latest
# then on the cluster:
srun --gres=gpu:1 --container-image=/path/to/naja.sqsh \
     --container-mounts=$DATA:$DATA -- naja sample run --model-dir ... 
```

## Option B — build in an existing CUDA container (fastest right now)

If you already have a CUDA-13 image on the cluster (e.g. an NGC PyTorch sqsh),
build in place with `native` (auto-detects the node's GPU — this is what
`scripts/build_container.sh` + `scripts/run_b300_naja_build.sbatch` do):

```bash
sbatch scripts/run_b300_naja_build.sbatch    # -> build/naja for this cluster's GPU
```

or interactively inside any CUDA-devel container:

```bash
NAJA_CUDA_ARCH=native ./scripts/build_container.sh   # installs cmake/ninja/zlib, then builds
```

## Running: Gurobi license

The LP feasible-start (`src/pipeline/feasible_start_lp.cpp`) needs a valid
Gurobi 13.x license **at runtime only**. Provide it with `GRB_LICENSE_FILE`
(or mount `~/gurobi.lic`); building and non-LP code paths need no license.
