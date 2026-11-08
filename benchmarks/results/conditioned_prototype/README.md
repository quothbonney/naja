# Conditional rounding-reuse prototype

Target: iJO1366 with PGI constrained to [-0.1, 0.1], an explicitly labeled
capacity knockdown rather than an exact knockout. All other original bounds,
medium and stoichiometry remain unchanged; no growth objective is imposed.

[Implementation and exact source interfaces](../../audit/conditioned_reuse_assessment.md)
· [Initial Slurm jobs and priority hold](jobs.json)
· [Live saved selections](summary.json).

Seed101 is the initial experiment. Fresh full PolyRound + native CHRR, inherited
base geometry + native CHRR, and inherited geometry + Naja barrier rotation use
the same conditional target. An independently labeled Naja rotation-off control
is included. CPU uses 16 physical cores with 32 logical CPUs reserved; Naja uses
one H100 and 16 chains. All are separate from canonical benchmark families.

Budgets are 10,000 then 50,000 retained draws/chain, with 1,000 discarded warmup
draws. The declared exploratory criterion is median split-Rhat <= 1.05 and
bulk ESS >= 100, plus separately required conditioned/original bounds and mass
balance at 1e-7. Every selected budget is actually executed; no failed cap is
called converged. Preparation and correction costs are included or shown
explicitly; one-time base setup and external diagnostics are separately reported.

No winner or three-seed completion is assumed. Sampling starts only after the
higher-priority 96-logical-CPU 512-chain benchmark. Fresh condition-specific LP
masks, exact delta arrays, hashes, strict interior certificates and validation
logs accompany the geometry when preparation completes.


The primary performance interpretation separates conditioning setup/correction
from sampling. Report measured selected-budget sampling-phase time and median
bulk ESS per sampling-phase second, with process/output handling secondary.
Current CPU phase wraps `hopsy.sample`, including multiprocessing and IPC;
current GPU phase wraps `CoordinateHitAndRun`, including device sample-buffer
allocation, slack GEMM and RNG initialization. Both exclude file output and
external adapters, but neither is silently equated to a pure compute timer.
Validated worker-compute and CUDA-kernel instrumentation will be added when
available. Full PolyRound setup versus inherited assembly/interior-LP plus local
barrier correction is a separate workflow setup comparison, not a GPU CHR-kernel
speedup. All complete-process and output costs remain secondary evidence.

CPU runs now use the opt-in worker timer validated byte-identical on BOX and
real iJO1366: [real parity proof](../../audit/hopsy-worker-timing-verification/iJO1366-real-parity.json).
The frozen instrumentation module SHA256 is
`a01b0bfd384c37ad483abdc0840fc28611ce34132d654e91e1e794a09e0732ba`.
Primary CPU sampling time is the maximum accumulated `MarkovChain.draw` time
across workers, excluding startup, IPC and output-list assembly; Python binding
and state-copy costs remain included. The original inclusive `hopsy.sample`
phase and full wrapper process remain secondary fields. GPU still has an
explicitly scoped sampler-phase proxy until a condition-specific pure-kernel
profile is attached. These timer scopes must accompany any comparison.

Shared CPU and Naja start from the same strict conditional interior point;
fresh PolyRound CPU starts from its own rounded origin. These are not
overdispersed initializations. The first unnormalized-LP preparation attempt
477 failed its strict-interior gate before sampling; positive LP row
normalization fixed solver scaling while preserving every saved A/b halfspace.
The failure log and both code snapshots remain preserved.
