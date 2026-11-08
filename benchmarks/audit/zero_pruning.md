# Certified-zero preprocessing variant

`scripts/zero_prune_fva.py` runs full COBRApy/GLPK flux variability analysis
with a zero objective, fraction of optimum zero, original exchange bounds,
no loopless restriction, and feasibility tolerance 1e-9. Only reactions with
finite minimum and maximum both within absolute 1e-9 of zero are selected.
The complete ranges, source SHA256, solver settings and measured FVA time are
cached before any PolyRound work. No original reaction bounds are edited.

`cpu_baseline.py prep --zero-prune-fva CACHE` removes those reactions only in
the working model, runs the existing PolyRound pipeline, then restores exact
zero rows in the full reaction-space T and shift arrays. Original reaction
ordering, full bounds, full stoichiometric equalities and original reaction
count are exported. This preserves the target up to the explicitly declared
1e-9 zero-flux identification precision. FVA time is separate; the reported
`total_preprocessing_seconds` includes FVA plus the selected preprocessing run.

The e_coli_core smoke test certified eight zeros, reduced working reactions
from 95 to 87, and retained the original final d=24 and m=36. Independent checks
against the original JSON passed at 1e-7, including bounds, mass balances,
reaction order and exact zero embedding. Original-space pilot bound violations
were zero. Its artifacts are private, leaving the canonical benchmark untouched.

Recon3D FVA is job 363. Two private preprocessing candidates depend on its
successful completion: default GLPK job 367 and GLPK presolve job 368, each
16 CPUs/96 GB with a three-hour cap. The original job 150_6 continues. The
first independently validated candidate may be published only after stopping
all other canonical writers; the canonical validation marker is written last.
Once published, its geometry remains frozen. These scripts reproduce the jobs:

```bash
sbatch benchmarks/scripts/zero_prune_candidate.sbatch fva Recon3D
sbatch --dependency=afterok:FVA_JOB benchmarks/scripts/zero_prune_candidate.sbatch prep Recon3D default
sbatch --dependency=afterok:FVA_JOB benchmarks/scripts/zero_prune_candidate.sbatch prep Recon3D presolve
```

Recon3D result: full FVA completed in 134.29 seconds with all 10,600 ranges
finite and **no zero reactions**. The smallest maximum absolute feasible flux
is 0.1; no reaction qualifies even at 1e-6. Consequently this optimization
cannot reduce this model's variables or constraints. The two newly started
private duplicate preprocessing jobs were cancelled, the original preprocessing
continues, and no canonical files were changed. The decision is recorded in
`data/preprocessing/Recon3D_zero_fva/decision.json`. The full FVA table remains
useful for common diagnostic masks.

The same FVA results certify a useful second variant: every original reaction
has positive width (minimum 0.2), and 2,723 bound faces are strictly inactive by
more than 1e-3. `prep --fva-inactive-faces` retains the other 18,477 faces,
normalizes the system, and directly runs PolyRound equality elimination and MVE
rounding, omitting serial simplification. The original full-space bounds remain
in the exported GEM files and in independent validation.

Why joint removal is valid in exact arithmetic: take a point strictly inside
all original coordinate bounds relative to S. If removing inactive faces added
a new feasible point, the segment from the interior point to it would first
cross one removed face while satisfying every original constraint, contradicting
that face's strict inactivity. Positive FVA width for each coordinate ensures
such a relative-interior point for a box intersected with S: average points
strictly inside each coordinate's bounds. Numerically, the 1e-3 inactivity
margin is a conservative guard compared with the 1e-9 LP feasibility setting,
not an exact-arithmetic certificate. Full original-bound/equality validation
is still required before promotion.

The e_coli_core face-variant smoke test retained 43 faces (standard PolyRound
retains 36), kept d=24, restored eight zero rows, and passed independent
original-space validation at 1e-7 with zero bound violations. Recon3D's private
face-variant job is 386 (8 CPUs, 96 GB, three-hour cap); original job 150_6 runs
in parallel. Its total preprocessing accounting includes the 134.29-second
full FVA cost rather than treating the cached work as free.
