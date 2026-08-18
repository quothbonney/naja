# Naja: GPU-Accelerated Convex Polytope Sampling

> **Naja** *(noun)* — a genus of elapid snakes, including the true cobras.
> From Sanskrit *nāga*: a divine serpent, capable of exerting a rather
> disproportionate effect.

<p align="center">
  <img src="data/petitprince.jpg" alt="A boa constrictor digesting an elephant — or perhaps a hat" width="420">
</p>

Naja is a CUDA program for drawing many samples from large, high-dimensional
convex polytopes. Its first home is genome-scale metabolic flux sampling, but
the sampler itself is not especially interested in where your inequalities came
from. Give it a well-formed polytope, a GPU, and a little patience.

It implements coordinate hit-and-run with GPU execution, model conditioning,
rounding, multi-chain runs, and diagnostics for the question every MCMC result
eventually has to answer: “should I believe this?”

## The short version

Naja needs a CUDA-capable Linux machine. Build it there; the build guide has the
unpleasant but useful details about CUDA versions, Eigen, Gurobi, and containers.

```bash
./scripts/build.sh

./build/naja sample run \
  --model-dir /path/to/models/MODEL_X \
  --out-root /path/to/out \
  --gpu 0 --n-chains 4 --n-samples 5000 --write-npy

./build/naja validate \
  --samples /path/to/out/MODEL_X_YYYYMMDD_001/samples.npy \
  --n-chains 4
```

For a cautious first run, add `--dry-run`. For flux-space output and bounds
checking, add `--backmap --bounds-policy filter`.

## The longer version, elsewhere

- [Building Naja](docs/building.md) — prerequisites, portable CUDA builds, containers, and tests.
- [Model format reference](docs/model-format.md) — the input contract, legacy CSV support, and the preferred `.npz` bundles.
- [GPU CHR mathematics](docs/gpu-chrr-mathematics.md) — the sampling algorithm and its CUDA realization.
- [Architecture](docs/architecture.md) — how the snake is assembled internally.
- [Technical summary](docs/technical-summary.md) — LLM-oriented repository context and implementation notes.
- [Slurm guide](slurm/GUIDE.md) — running a small herd of jobs without losing track of them.
- [Contributing](CONTRIBUTING.md) — tests, reproducibility expectations, and what must stay out of Git.
- [Citation metadata](CITATION.cff) — for work that grows out of this one.

## Commands worth knowing

```text
naja sample run       sample one model
naja sample bulk      distribute many models across one or more GPUs
naja sample prepare   inherit rounding and prepare conditioned models
naja condition eflux  condition a model from expression data
naja validate         compute ESS, split-R-hat, bounds checks, and chord lengths
```

Ask the executable when in doubt:

```bash
./build/naja --help
./build/naja sample run --help
```

Naja writes run manifests and generated configurations alongside output. Keep
them with results: a sample without its command line is just an interestingly
shaped pile of numbers.

## A small warning label

This repository does not distribute Eigen or Gurobi. Gurobi 13 development files
are required to build the current executable, and a valid license is required
when using the LP feasible-start path. The precise setup is in [docs/building.md](docs/building.md).

The GPU correctness tests are meant to be run on each new hardware class. A100,
H100, and B300/CUDA 13.2 are covered by the portable-build guidance; please do
not mistake an untested GPU for a snake that has already eaten the elephant.
