# Naja: GPU-Accelerated Convex Polytope Sampling

> **Naja** *(noun)*  
> A genus of elapid snakes, including the true cobras.  
> From Sanskrit *nāga*: divine serpent, capable of exerting disproportionate effect.

Naja is a GPU-first library for high-throughput sampling of large, high-dimensional convex polytopes, designed to serve as a standalone computational core for constraint-based modeling and related feasibility problems when CPU-based approaches become intractable.

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

<br>



<p align="center">
  <img src="./data/petitprince.jpg" alt="petitprince" width="40%">
</p>
