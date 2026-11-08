# Nine-model barrier phase timing

Slurm job880, one H100, three independent-process timing repeats per mode and model.
Same frozen experimental binary in CPU/GPU-rotation modes (SHA256 in summary.json).
Seed101,16chains,8short samples,thinning1,TPB128,F64 output. These are timing repeats,
not independent convergence trials. All128short samples per run are checked.
Declared bound tolerance is1e-7 except historical RECON1 geometry at1e-6.
One-time builds, source snapshots and full rotation-algebra validation are linked
from ../barrier_profile/README.md; this extension reuses that binary unchanged.

| Model | d | Original barrier median s | GPU-rotation median s | Phase ratio |
|---|---:|---:|---:|---:|
| e_coli_core | 24 | 0.0376 | 0.0707 | 0.532 |
| iAB_RBC_283 | 130 | 0.0798 | 0.0776 | 1.03 |
| iJR904 | 232 | 0.0791 | 0.0762 | 1.04 |
| iAF1260 | 524 | 0.1356 | 0.0949 | 1.43 |
| iJO1366 | 581 | 0.157 | 0.1099 | 1.43 |
| iML1515 | 635 | 0.1709 | 0.0993 | 1.72 |
| iMM904 | 288 | 0.0891 | 0.0809 | 1.1 |
| RECON1 | 931 | 0.5135 | 0.1167 | 4.4 |
| Recon3D_nullspace_only | 4861 | 94.618 | 1.4359 | 65.9 |

The figure also shows Hessian+eigensolve stage wall times (including host waits,
lazy library initialization and synchronization) and historical one-time PolyRound
MVE wall times from each model's meta.json. Those are different mathematical
operations: the Naja barrier is evaluated on cached geometry after preprocessing,
while PolyRound constructs the global rounding map. Their vertical separation
is phase-cost context and does not establish interchangeable rounding quality.
Recon3D_nullspace_only has no MVE preprocessing result and is explicitly exploratory;
its short feasibility checks do not establish convergence. Cold small-model GPU
rotation can cost more than CPU rotation because of initialization overhead.

[Figure SVG](barrier_dimension_scaling.svg) · [PDF](barrier_dimension_scaling.pdf) ·
[All repeat source CSV](source.csv) · [Median CSV](medians.csv) · [Exact stage records](summary.json).
Exact native command lines are in repeat1/commands.sh, repeat2/commands.sh and
repeat3/commands.sh; rerun with `sbatch benchmarks/scripts/barrier_dimension_scaling.sbatch`
from the workspace root. Plot using the local env-plot environment and
`scripts/barrier_dimension_scaling.py --plot`.
