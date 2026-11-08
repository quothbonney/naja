# H100 utilization timeline

`naja_h100_utilization_timeline.svg` and `.pdf` are the requested three-row,
shared-seconds figure. `.png` is a preview. No convergence claim is made.
Canonical benchmark results and binaries were not changed.

The visual reference is [FlashTP Figure 6](https://proceedings.mlr.press/v267/lee25l.html).
This plot uses actual NVIDIA Nsight Systems 2025.3.2 GPU hardware counters,
GH100 general metric set `gh100`, at **10,000 Hz**. The source is the exported
SQLite `GPU_METRICS` table, joined to `TARGET_INFO_GPU_METRICS` by metric ID.
It does not use nvidia-smi/DCGM utilization. nvidia-smi is used only to record
static hardware identity. `RmProfilingAdminOnly: 0`; counters were accessible.

The allocated device was Slurm GPU4, exposed as local device0, PCI
`0000:9b:00.0`, UUID `884d19b4-3b1e-5246-0d07-a1a68b778800`. The probe's
visible hardware identity matches the exported `TARGET_INFO_GPU` table for
each full timeline. Slurm allocated a dedicated GPU; other workloads could
still share host CPUs, memory and storage. These are throughput/profiling runs.

## Protocol and counter meanings

All three runs: iJO1366, seed101, barrier rotation, automatic thinning
(resolved to96), TPB128,1,000 warmup plus10,000 retained draws per chain;
the native command generates11,000 draws and writes all of them. Row1 uses
16 chains, row2 uses512, row3 uses512 and `--host-prefault-threads 8`.
The first two rows use the verified wide-index `build-h100-safe/naja`;
the third uses `build-h100-prefault/naja`, also with wide indexing.
Manifests record the exact binary SHA256, Slurm allocation and commands.
No external sample-layout normalization is included in the timeline.
The two512-chain profiled native arrays are bitwise identical, including
the NPY file bytes; `same_seed_output_hashes.json` records both SHA256 hashes.

The plotted blue line is **SM active**, a fraction of peak SM activity;
it is not arithmetic SM throughput or achieved warp occupancy. Orange sums
Nsight's DRAM read/write throughput percentages. PCIe RX/TX use Nsight's
own peak-sustained-throughput percentages, including protocol bytes; they
are not estimated from PCIe generation/link width. Peak denominators are
the NVIDIA metric definitions, not a locally fitted maximum. GH100 general
metrics do not expose L2 bandwidth, so it is measured separately with Nsight
Compute. Raw counter percentages are integer-quantized by this export.
`counter_coverage.json` verifies100µs median intervals in all three captures,
with no gaps above150µs. Nsight skipped unsupported CTC metrics, which are
unrelated to the four plotted series; every requested plotted metric is present.

`timeline_source.csv` preserves the unsmoothed100µs observations, including
separate DRAM read/write columns. `timeline_plotted.csv` preserves the four
plotted series: a centered **5ms moving average**, sampled every1ms for
rendering. Edge windows use only available observations, not zero padding.
Data end when the target process exits; shorter runs are not padded with
idle measurements out to the shared axis limit.

## Phase extraction and cross-checks

Naja emits no NVTX ranges in these binaries. `kernel_trace.csv` in each run
retains every kernel and its time. Brackets come from the CUDA trace:

- Prepare: process launch to `chrKernel` start; includes load, CUDA startup,
  barrier SYRK/eigendecomposition, allocation and upload.
- Sample: the actual `chrKernel` interval.
- Map + D2H: sampling end through the last large device-to-host copy. This
  includes the DGEMM undoing the581-dimensional barrier rotation, any host
  allocation/first-touch wait, and the copy.
- Write / exit: the trailing GPU-idle interval through target process exit.
  This includes output writing and cleanup; it is deliberately not labeled
  as a pure write-system-call timer.

The native path exports reduced coordinates. Its DGEMM undoes the barrier
rotation; it does **not** apply the full reaction-space affine backmap.
`back_transform=false` in native profile JSON refers to that latter option.
Naja's `backtransform_s` encloses its download timer, so those two fields
must not be added together. `timeline_summary.json` preserves trace/kernel
durations beside all native profiler times. In particular, a prefault download
includes CPU page touches that are absent from the device memcpy interval.

Each target wall is measured by the same small subprocess timer inside and
outside Nsight. Its epoch timestamps align the SQLite session epoch to process
launch/exit; the x-axis starts at launch. `profiler_overhead.csv` compares each
profiled target wall against the midpoint of adjacent unprofiled runs, on the
same allocated GPU. It separately records total Nsight command wall including
trace processing/export, which is not target runtime.

| Run | Unprofiled midpoint(s) | Profiled target(s) | Target overhead |
|---|---:|---:|---:|
|16 chains|5.941|6.273|5.60%|
|512 chains|28.388|29.153|2.70%|
|512 + prefault|19.288|19.462|0.90%|

These are single profiled runs bracketed by two unprofiled runs, not uncertainty
estimates. In particular, the two unprofiled prefault walls differ by1.18s.

## Commands, raw data and reproduction

`commands.sh` records exact executed commands. Each `timeline.nsys-rep` and
`timeline.sqlite` is retained, together with target output, stderr/stdout,
native profiles, binary hashes and timings. The initial counter probe and its
metric-set enumeration are in `probe/`. Tools were downloaded from NVIDIA's
official CUDA package repository, SHA256-checked against its package index,
and extracted locally; `tool-downloads.json` records URLs and hashes. No system
packages, driver settings or global profiler permissions were changed.

From the workspace root:

```bash
sbatch --chdir="$PWD/benchmarks" benchmarks/scripts/nsight_timeline.sbatch
benchmarks/env-plot/bin/python benchmarks/scripts/plot_nsight_timeline.py
sbatch --chdir="$PWD/benchmarks" benchmarks/scripts/nsight_compute.sbatch
benchmarks/env-cpu/bin/python benchmarks/scripts/summarize_nsight_compute.py
sbatch --chdir="$PWD/benchmarks" benchmarks/scripts/nsight_l2_recheck.sbatch
sbatch --chdir="$PWD/benchmarks" benchmarks/scripts/nsight_l2_counts.sbatch
```

The collection script preserves successful targets when resumed; use a fresh
artifact directory for an independent complete repetition. Saved commands use
an unchanged tiny timed launcher around Naja so profiler postprocessing is
excluded from target wall time. The underlying Naja arguments are recorded
verbatim. Nsight traces the target child process, verified by the actual CUDA
kernel/memcpy records. Launch these scripts only with an allocated GPU.

## Nsight Compute and roofline inputs

`ncu/commands.sh` records full-set, one-kernel profiles of CHRR at16,128 and512
chains with200 retained plus1,000 warmup draws. These short runs use the same
automatic thinning and TPB128. Additional profiles cover representative SYRK,
eigendecomposition (`sytrd4_gpu`, `steqr_ker`, `laed4_par`) and the rotation-undo
DGEMM. The latter uses the full16-chain output shape so the kernel name and
invocation match the timeline. These are representative kernels, not an
aggregate eigensolver roofline. Missing kernel matches must be treated as
missing evidence, not zero FLOPs.

Nsight Compute uses `--set full`, explicit requested bytes/occupancy/FLOP
counters, kernel replay, its default cache policy, and `--clock-control none`
to avoid changing device clocks. Full instrumentation/replay costs can be
orders of magnitude greater than ordinary target time and are not benchmark
timings. Reports retain warnings about unavailable unrelated CTC counters.

`ncu/kernel_summary.csv` and `ncu/summary.json` contain L2 requested-byte
throughput as percent of peak and hit rate, DRAM byte throughput, SM throughput,
achieved occupancy, launch resource limits, measured bytes and executed FP64
work for roofline inputs. Executed FP64 FLOPs are DADD + DMUL +2×DFMA thread
instructions plus NVIDIA's FP64 tensor-operation count. These include floating
operations used inside lowered division and are not just an algorithmic FLOP
estimate. `ncu/warp_stalls.csv` ranks NVIDIA's stalled-warps-per-issue-active
ratios; those ratios are not percentages. All raw `.ncu-rep` and CSV exports
are retained, so alternate roofline definitions can be recomputed.

NVIDIA's derived L2 hit ratio is unstable across replay passes: the128-chain
full-set export reports an impossible100.591%, while narrow repeats also
produce out-of-range16/512 values. All are retained verbatim, and the full-set
invalid value is flagged by `l2_hit_rate_in_physical_range=false`; none is
clipped. `ncu_l2_recheck/` preserves the narrow derived-ratio repeat.
`ncu_l2_counts/` separately collects only hit and miss sector counts. Even
that collection requires three passes, so we do not claim simultaneous
single-pass counters. The explicit estimate100×hit/(hit+miss) is99.837%,
99.990%,99.975% at16,128,512 chains. It is a count-normalized estimate,
distinct from NVIDIA's unstable derived ratio. Both definitions and the raw
counts are in `ncu/l2_hit_rate_validation.csv`/`.json`. NVIDIA documents
[out-of-range metrics caused by replay variation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#range-and-precision).
All other requested full-set metrics and all eight kernel profiles completed.

| CHRR chains | SM throughput(%peak) | L2 bytes(%peak) | DRAM bytes(%peak) | Achieved occupancy |
|---|---:|---:|---:|---:|
|16|1.335|0.487|0.00581|6.250%|
|128|10.773|3.895|0.04727|6.250%|
|512|38.519|14.380|0.17094|23.861%|

All three original CHRR launches use104 registers/thread and6,216B shared
memory/block, with four resident blocks/SM allowed by registers. Theoretical
warp occupancy is25%. At512 chains, the kernel reaches0.97 resident waves
and nearly that occupancy ceiling. The top three warp stall categories are
wait, long scoreboard, and barrier at every chain count. This identifies a
register/latency/issue problem rather than saturation of HBM bandwidth;
it does not attribute every stall to a particular source line.
