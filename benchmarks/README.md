# Naja H100 benchmark prototype

Campaign started 2026-11-07 23:15 UTC. User deadline: 12 hours.

All added scripts, environments, inputs, logs and outputs live here. The Naja
source is `../naja-test-jack/naja`; seed support, aligned GPU-test flags, and a
separate wide-index fix are recorded alongside source and binary hashes. The
canonical benchmark binary remains frozen; large-matrix experiments use a
separately verified build. No credentials are saved.

The first deliverable is a measured matched Naja/hopsy table on the successfully
preprocessed BiGG models, not a reconstruction of unavailable historical results.
The original nine-model list and paper are absent. Production corpus work is
outside this prototype's scope. The nine
public models requested by the user are downloaded by `scripts/fetch_models.py`.

Protocol: shared PolyRound geometry, 16 independent chains, 1,000 retained-draw
warmup followed by 10,000 retained draws per chain, three explicit seeds when
time permits. Record wall time including warmup and writing; preserve phase times
and actual thinning. Use common external reaction-space diagnostics and report
bound violations before any filtering. Exclude constant reactions from aggregate
R-hat/ESS, retaining their count. Automatic Naja thinning is a geometry heuristic.

CPU chain parallelism uses 16 workers for 16 chains; 128 worker processes would
not create additional independent chain work. CPU and GPU jobs use separate Slurm
allocations and constrained BLAS threads. CPU baseline defaults and optimized
thinning will be stated explicitly, with no unsupported literature attribution.

GPU correctness must pass before benchmark sampling. Existing Gurobi WLS license
was reported expired by Gurobi; use license-free CPU preprocessing and Naja's
file-start path. MATLAB was not found in the compute-node PATH.

From the workspace root, setup commands are `bash benchmarks/scripts/setup_cpu.sh`
and `bash benchmarks/scripts/setup_gpu.sh`. Exact package locks are under this
directory and `results/`. GPU checks, from this directory:
`sbatch --output=logs/correctness-%j.log scripts/correctness.sbatch`.

The live report is [results/BENCHMARK_TABLE.md](results/BENCHMARK_TABLE.md), with
machine-readable tables in `results/summary/`, actual passing-budget records in
`results/quality/`, and exported figures in `results/figures/`. Hardware and build
details are in [results/HARDWARE.md](results/HARDWARE.md); interpretation and
independent checks are in `audit/`. Every run retains its full command, input
model, seed, sampler settings, wall time and diagnostics. Use fresh output paths
when reproducing; existing canonical outputs are retained.

Examples from this directory (inside an appropriate Slurm allocation):

```bash
env-cpu/bin/python cpu_baseline.py sample --model data/models/e_coli_core \
  --out results/reproduction/hopsy-core --seed 101 --thinning jadebeck --workers 16
scripts/enter.sh python naja_runner.py --exe build-h100/naja \
  --model data/models/e_coli_core --out results/reproduction/naja-core --seed 101
env-cpu/bin/python metrics.py --samples results/reproduction/naja-core/samples.npy \
  --model-dir data/models/e_coli_core
```

`scripts/campaign.py` submits bounded Slurm arrays when model preprocessing is
ready. Its deadline is an explicit campaign setting; change `--deadline-utc` for
a future campaign. `scripts/refresh_results.py` updates reports and figures every
minute and stops remaining campaign jobs at 11:05 UTC on 2026-11-08, before the
user's 11:15 deadline. PID files are `results/campaign.pid`, `results/refresh.pid`
and the quality controller's metadata. These are local processes and Slurm jobs;
no system service or cron entry is installed.
