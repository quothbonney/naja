# Bounded parallel common diagnostics

`metrics.py` preserves the schema-2 scientific definitions and now factors each
reaction chunk into a picklable worker. Spawned workers open the same readonly
sample mmap and affine backmap once, use BLAS threads=1, and return indexed
Rhat/ESS/range/count arrays plus two per-sample violation masks. The parent
assigns each reaction chunk by index and ORs the masks. No hopsy chain or model
object is forked/pickled.

Automatic parallelism requires all of:

- A successful verification artifact whose estimator source SHA matches the
  currently imported `metrics.py` source SHA.
- At least 20,000,000 chain × draw × reaction observations (a workload heuristic).
- More than one allocated CPU. Worker count is bounded by four and
  `SLURM_CPUS_PER_TASK`.

Small workloads remain serial because spawn setup costs exceed their diagnostic
work. `--workers 1` forces the serial path. Explicit `--workers 4` permits
verification runs but still obeys the Slurm CPU cap. Worker BLAS thread limits
are set before spawn imports NumPy, and the parent's original environment is
restored afterward. No baseline environment is modified.

Reproduce the independent scientific/performance comparison:

```bash
# From the workspace root, after the actual input sample runs exist:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  srun --partition=all --cpus-per-task=4 --mem=16G --time=00:20:00 \
  benchmarks/env-cpu/bin/python benchmarks/audit/test_metrics_parallel.py
```

The test uses the archived pre-refactor estimator, refactored serial estimator,
and four spawned workers. It checks every scientific JSON field to 1e-12 and
requires per-reaction CSVs to match **bitwise** on actual e_coli_core data and an
actual iJO1366 50,000-draw trajectory projected onto 128 real reaction rows
(64 model-active and 64 solver-masked). The projection is an audit fixture, not
a new benchmark model. No duplicate large sample file is created.

The current proof is `metrics-parallel-verification.json`, tied to source SHA
`854e524cade5cecc6a9c5541cb15957e0336c3461434c2363d9ef2a2b701947c`.
The archived serial reference is `metrics.serial-before-parallel.py`. A source
edit invalidates the automatic-parallel verification gate until the test is
rerun successfully. Proof writing is atomic.

Measured on the reserved H100-node CPU allocation:

| Case | Refactored serial | Four workers | Observed speedup |
|---|---:|---:|---:|
| Core, 16 × 500 draws, 95 reactions | 0.240 s | 3.092 s | 0.078× |
| iJO1366 projection, 16 × 50,000 draws, 128 reactions | 36.046 s | 17.848 s | 2.020× |

The archived serial estimator took 34.228 s on the heavy case, so four workers
were about 1.918× faster than that reference in this shared-node measurement.
These are diagnostics-only pilot timings, not Naja/CPU sampling speedups.
Repeated timing noise and concurrent node load are not hidden by claiming an
ideal fourfold gain.

New metrics record the import-time estimator code SHA, actual worker count,
allocated CPUs, worker BLAS limit, verification status and elapsed diagnostic
time. Old schema-2 numerical results remain scientifically valid; strict
cross-run reuse must preserve the **producing module's import-time SHA**, not
reread a possibly edited file after computation. Results from older modules
without an import-time SHA must not be relabeled as produced by a newer source
for strict cache proofs. The quality-cache helper owns that conservative
eligibility check.
