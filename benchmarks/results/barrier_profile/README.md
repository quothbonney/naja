# Barrier CPU/GPU attribution

This isolated experiment changes only the dense constraint rotation `A * Q`
from host Eigen multiplication to cuBLAS DGEMM, retaining its required host/device
copies. The executable is identical in CPU and GPU modes; only environment
variable `NAJA_BARRIER_GPU_ROTATE=1` selects the GPU multiplication. The CPU
mode removes that environment variable entirely. Frozen canonical source,
binaries and benchmark results are unchanged.

| Model | CPU-path barrier(s) | GPU-path barrier(s) | Barrier speedup |
|---|---:|---:|---:|
|iJO1366|0.1491|0.1051|1.42×|
|RECON1|0.5078|0.1938|2.62×|
|Recon3D, exploratory no-MVE|93.8597|1.4728|63.73×|

On large Recon, the existing CPU `A*Q` consumes92.9324s, about99.01% of
the barrier phase. The profiled CPU rotation NVTX range contains zero CUDA
kernel or memcpy activity. GPU rotation costs0.49388s including all copies;
its DGEMM kernel takes0.02033s. Full-matrix validation against the CPU
reference gives relative Frobenius error2.396e-15. These are single-pass
measurements of this Naja implementation, with validation costs excluded.

Frozen experimental executable: `build-h100-barrier-profile/naja`, SHA256
`8a9387cb0339b83132191a43b9e04d86a28d9381287cdc22c8dac1bb2214b54f`.
Source snapshot/patch and the five analytic GPU tests plus CPU-mode bitwise
parity checks are retained in this folder. Root owns the experimental source.
The orchestration scripts are `scripts/barrier_profile_runs.py` and
`scripts/barrier_profile.sbatch`; `commands.sh` records exact executed commands.

Each target: seed101,16 chains,8 total samples per chain, fixed thinning1,
TPB128, zero base/delta constraint epsilon, float64 output, host prefault8 for
sample download. iJO1366 and RECON1 use barrier rotation. The large
`Recon3D_nullspace_only` geometry uses rotation+whitening and **no iterative
MVE rounding**; it is an explicitly exploratory equality-eliminated target,
not the canonical nine-model convergence comparison. These short runs test
profiling/algebra/feasibility and do not establish mixing or uniformity.

## Measurements

All runs enable `NAJA_BARRIER_TRACE=1`, producing nested NVTX ranges and
`NAJA_BARRIER_STAGE` JSON in stderr. Each model/mode has an unprofiled target
and an Nsight Systems target tracing CUDA,NVTX,OSRT. CPU sampling and context
switch collection are disabled. Target wall time comes from the same subprocess
timer in both cases, excluding Nsight report-processing time. Single-pair
profiler overhead is reported; tiny subsecond targets have appreciable overhead.
Primary CPU/GPU phase comparisons use unprofiled native profiler/chrono times.

`barrier.total` encloses the full barrier block. The pure-CPU `barrier.cpu_rotate_A`
range encloses `A_host = A_host * rotation_Q`. Its dense work is approximately
`2*m*d*d` FLOPs. NVTX traces distinguish this host work from the cuBLAS Hessian
SYRK and cuSOLVER eigensolve kernels. All nested stage values are saved; do not
add a parent range to its children.

For profiled targets, `trace.sqlite`, `kernel_trace.csv` and `nvtx_stages.csv`
retain the complete evidence. GPU math sum is the sum of CUDA **kernel** interval
durations clipped to `barrier.total`. GPU math union merges overlapping intervals;
these are separately labeled, and memory copies are recorded separately. Neither
measure substitutes a coarse GPU utilization percentage. Native barrier time and
the enclosing NVTX range can differ slightly because end-of-scope cleanup follows
the native timer reading.

The GPU rotation range includes upload of A and Q, allocation, handle setup,
DGEMM, synchronization, download and cleanup. Its leaf range `rotation.upload_A`
does not include the separate Q upload; use the outer `barrier.gpu_rotate_A`
for the complete cost. The helper currently downloads A with its default pageable
host allocation, so `--host-prefault-threads 8` affects sampling output, not this
A round trip. No transfer/allocation cost is hidden from the rotation comparison.

## Validation and scope

Separate JO and large-Recon validation targets set
`NAJA_BARRIER_VALIDATE_ROTATION=1`. They compute the full CPU reference `A*Q`
and compare the full GPU matrix against it, requiring finite output and relative
Frobenius error≤1e-10. These targets include expensive CPU reference work and
are explicitly excluded from performance comparisons. Their JSON is retained
under `gpu/validation/result.json`. Root's generic GPU test/parity evidence is
under `verification/`.

Every completed short target checks all128 float64 samples against every original
reaction bound through the saved affine map, and against the original reduced
constraints. No samples or reactions are filtered or clipped. JO and exploratory
Recon3D use1e-7 bounds tolerance. RECON1 retains its documented1e-6 geometry
acceptance tolerance; its raw1e-7 and exact-zero fractions are also preserved.
These checks are not a convergence test.

`summary.json` contains individual `runs` and per-model `comparisons`;
`comparison.csv` is the compact table. Per-run `result.json` includes the exact
command, binary hash, native profile, chronological stage timings, numerical
validation and target wall. Comparisons use four completed timing runs per model;
large-Recon algebra validation follows the first performance pair to obtain the
requested attribution quickly. A missing validation is pending evidence rather
than a successful test; RECON1's null algebra field means a separate full-matrix
reference was not requested there (the generic helper is checked on JO and
large Recon, and RECON1 has its own sample bound checks). Job515 was intentionally stopped after saving all small
model pairs to prioritize the large target; job520 resumes those completed cases.
Its interrupted validation directory is preserved and excluded from timing rows.

Run from the workspace root with an allocated Slurm GPU:

```bash
sbatch --chdir="$PWD/benchmarks" benchmarks/scripts/barrier_profile.sbatch
```

That resumes saved completed cases. For a fresh independent repetition:

```bash
NAJA_BARRIER_PROFILE_OUT="$PWD/benchmarks/results/barrier_profile_repeat" \
  sbatch --chdir="$PWD/benchmarks" benchmarks/scripts/barrier_profile.sbatch
```
