# Diagnostic assay scope

The original reaction masks certify min/max ranges on the original COBRA model.
They do not by themselves certify ranges on the exact cached A/b after finite-precision
equality elimination and rounding. Original data/masks/results are preserved.

The extension results initially use that same historical assay, for comparability.
They must not be presented as conservative cached-target quality validation.
Separate exact-cached-target GLPK certificates are being built for iML1515 and
iAF1260 under cached_models/, and saved samples will be reassayed with those masks.
Unresolved/capped LPs remain active. No output filtering, bound relaxation, target
change or sampler change is involved.

Bounds are checked independently on every original reaction in all assays.
The iMM904 violations (~2e-7) occur in both double-precision CPU and GPU output,
especially PLBP1I_SCe and PLBPC_SCe. They are a common cached-geometry limitation;
float64 export alone is not a demonstrated repair.
