# CPU baseline protocol

The pinned CPU environment is `benchmarks/env-cpu`, recreated by
`bash benchmarks/scripts/setup_cpu.sh`. Exact package versions are in
`benchmarks/cpu-requirements.lock.txt`; the initial environment has hopsy 1.7.0,
PolyRound 0.5.0, COBRApy 0.32.1, NumPy 2.5.4 and SciPy 1.18.1. GLPK 5.0 is
provided by swiglpk 5.0.13. Optional HiGHS/OSQP packages were subsequently added
for the RECON1 numerical retry; the initial environment is preserved in
`cpu-requirements.base.lock.txt`, and metadata records optional solver versions.
Gurobi is explicitly avoided because the supplied
license has expired. No MATLAB executable was visible on the compute node;
the prototype CPU reference is hopsy CHRR.

`cpu_baseline.py prep` loads the archived BiGG JSON without imposing an
objective-optimality restriction, constructs mass-balance equalities and original
reaction bounds, removes redundant and effectively fixed constraints, eliminates
equalities, and rounds with PolyRound's maximum-volume-ellipsoid algorithm.
These are one-time shared costs. Separate timings are recorded for simplification,
equality elimination, rounding, and writes in each model's `meta.json`.
Infinite bounds cause an error instead of silently being clipped. PolyRound's
default threshold of 1e-7 for effectively fixed directions is recorded.

The shared bundle is `data/models/ID/rounding/polytope.npz`, containing contiguous
float64 `A,b,start,T,shift`, with flux backmap `x = T @ y + shift`. The rounded
origin is strictly interior. Original bounds, equalities and reaction IDs are
stored under `gem/`. The original source file SHA256 is in `meta.json`.

Sampling uses `hopsy.UniformCoordinateHitAndRunProposal`, 16 independent RNGs,
and 16 worker processes. Requesting 128 workers does not run 16 chains faster:
the hopsy multiprocessing pool is capped by the number of chains (see installed
`hopsy/misc.py`, `sample` docstring). BLAS/OpenMP are limited to one thread per
sampling worker. The 1,000 retained warmup draws are generated and discarded;
10,000 subsequent draws per chain are saved. All chains start at the same
interior origin, with distinct seeds; this does not provide overdispersed starts.

The saved `samples.npy` has shape `(chains, draws, reduced_dimension)` and uses
float64. Sampling excludes affine backmapping; external common diagnostics map
both samplers to reactions. Therefore times describe generation in the reduced
space plus reduced output writing, and must not be advertised as a per-sample
backtransform benchmark. The runner records load, chain initialization, sampling
(including warmup), write and internal wall times in `run.json`. The separate
`process_wall_seconds` starts before NumPy/hopsy imports and includes module
startup; this is preferred for comparison to Naja subprocess wall time. Slurm
queue time is excluded. The whole hopsy API call,
including process setup and result transfer, is inside sampling time.

The thinning labels have distinct meanings:

- `default`: hopsy.sample's actual API default, thinning=1; this is unthinned.
- `jadebeck`: `max(1, ceil(d²/6))`, the GEM guideline from Jadebeck, Wiechert and
  Nöh, PLOS Computational Biology 2023, equation 14.
- `legacy`: `8*d²`, the earlier CHRR guideline, not hopsy's API default.
- An integer: explicit fixed thinning, recorded verbatim numerically.

The optimized-thinning result is the primary CPU comparison. Comparing only
against unthinned CHRR can inflate apparent speedups. The Jadebeck guideline
was derived for per-sample backtransformation; since this prototype saves
reduced samples, it is a documented reference setting rather than a claim of
optimality for this batching scheme.

Sources: [hopsy documentation](https://modsim.github.io/hopsy/),
[Jadebeck et al. 2023](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011378).

Example commands (inside a Slurm allocation; use `/usr/bin/env`, since the login
PATH contains an `env` shim inaccessible on compute nodes):

```bash
benchmarks/env-cpu/bin/python benchmarks/cpu_baseline.py prep \
  --model benchmarks/data/raw/e_coli_core.json \
  --out benchmarks/data/models/e_coli_core
benchmarks/env-cpu/bin/python benchmarks/cpu_baseline.py sample \
  --model benchmarks/data/models/e_coli_core \
  --out benchmarks/results/hopsy/e_coli_core/jadebeck/seed101 \
  --seed 101 --chains 16 --draws 10000 --warmup 1000 \
  --thinning jadebeck --workers 16 --chunk 11000
```

Single-call sampling is the default. Smaller chunks with one worker checkpoint
retained output and `completed_draws` to disk; diagnostics must ignore the
unwritten tail if interrupted. An independent audit found a hopsy 1.7 repeated
parallel-call pickling failure (`logs/hopsy-state-verification.log`), so the runner
explicitly rejects multiple chunks with more than one worker. All benchmark
parallel runs used a single call. Serial chunk state continuity was checked
exactly on a box; all six complete core runs were also bitwise identical across
one and sixteen workers. Interrupted runs are not silently resumed or treated
as complete.

Pilot observations: e_coli_core has n=95, d=24 and m=36 after preprocessing;
iAB_RBC_283 has n=469, d=130 and m=188. The 16-worker core pilot required about
3.9 seconds internally at thinning96. Running the same 16 chains serially
required about 1.3 seconds internally, because process-pool overhead dominates
this small problem. The raw samples were verified bitwise identical across
worker counts for all three seeds and both thinning settings. Consequently,
the report must retain worker count and compare against the fastest measured
CPU configuration, rather than assuming that more workers always improve
throughput. Module startup is additional and is reported separately as above.
