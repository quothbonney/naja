# Naja: GPU-Accelerated Convex Polytope Sampling

> **Naja** *(noun)*  
> A genus of elapid snakes, including the true cobras.  
> From Sanskrit *nāga*: divine serpent, capable of exerting disproportionate effect.

Naja is a GPU-first library for high-throughput sampling of large, high-dimensional convex polytopes, designed to serve as a standalone computational core for constraint-based modeling and related feasibility problems when CPU-based approaches become intractable.

---

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

Single model run (writes `samples.npy` and, with bounds filtering enabled, `valid_mask.npy` + `bounds_report.json`):
```bash
./build/naja sample run \
  --model-dir /path/to/models/MODEL_X \
  --out-root /path/to/out \
  --gpu 0 --n-chains 4 --n-samples 5000 --tpb 128 \
  --backmap --write-npy \
  --bounds-policy filter --bounds-eps 1e-6
```

Bulk run (many models by name; writes `bulk_summary.csv` under the run directory):
```bash
./build/naja sample bulk \
  --models-root /path/to/models \
  --model-list /path/to/jobs.txt \
  --out-root /path/to/out \
  --name RUN_NAME \
  --gpus 0,1,2,3 \
  --n-chains 4 --n-samples 5000 --tpb 128 \
  --backmap --write-npy \
  --bounds-policy filter --bounds-eps 1e-6
```

Bulk prepare many models (optimally inheriting rounding from a base model):
```bash
./build/naja sample prepare \
  --models-root /path/to/models \
  --model-list /path/to/jobs.txt \
  --base-model-dir /path/to/models/base_model/ \
  --mode symlink \
  --out-model-list /tmp/model_list.txt
```

---

<p align="center">
  <img src="./data/petitprince.jpg" alt="petitprince" width="50%">
</p>
