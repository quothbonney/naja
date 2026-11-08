# RECON1 preprocessing investigation

The initial PolyRound/GLPK preprocessing failed during its redundant-constraint
removal loop, while maximizing a constraint with its upper bound relaxed by one.
The original log is preserved as `logs/prep-150_5.log`; the initial command used
GLPK without presolve and feasibility tolerance 1e-9.

Independent feasibility checks in `scripts/diagnose_recon1.py` demonstrate that
the original archived model (3,741 reactions, 2,766 metabolites) is feasible.
Both SciPy HiGHS dual simplex and interior point succeed at requested primal
and dual tolerance 1e-9. GLPK succeeds with presolve both disabled and enabled,
with the returned flux solutions having zero measured equality residual and
zero bound violation. Results, timings and compute hostname are persisted in
`recon1_feasibility.json` and `logs/recon1-feasibility.log`.

The failure is therefore a numerical issue encountered during simplification,
not evidence that the original biological model is infeasible. A retry using
the same model, original bounds, same feasibility tolerance, and GLPK presolve
enabled is running with log `logs/recon1-presolve.log`. No biological bounds
were changed. The new runner CLI options `prep --presolve` and
`--feasibility-tol` make this adjustment explicit and record it in the resulting
model metadata. The initial failure remains part of the benchmark audit.

An independent retry uses PolyRound's same algorithms with optlang's optional
`hybrid` interface, which selects HiGHS for these linear programs. A complete
box simplification/rounding smoke test passed before submitting RECON1. That
trial writes to separate model directory `RECON1_hybrid` and log
`logs/recon1-hybrid.log`, with the GLPK trial capped at 30 minutes and the
HiGHS trial allowed two hours. Slurm denied extending the initial running
30-minute HiGHS allocation, so that attempt was restarted after about two
minutes; its output is preserved as `logs/recon1-hybrid-attempt30min.log`. Canonical
`RECON1` artifacts will only be populated from a successful, validated retry.

To enable this solver, highspy 1.15.1, osqp 1.1.3, and their three new
dependencies were added to the local CPU environment; no existing package was
upgraded. `cpu-requirements.base.lock.txt` records the initial environment and
`cpu-requirements.lock.txt` records the expanded one. Sampling itself continues
to use the same hopsy C++ proposal and specified starts, without solver calls.
Import startup may change because optlang can now discover more interfaces;
the optional solver versions are recorded in subsequent metadata.

The HiGHS retry failed with optlang status `undefined` / native HiGHS status
`Unknown`; PolyRound's generic error then says that reset is unsupported for
this backend. A snapshot of that exact intermediate LP was independently
re-solved by SciPy HiGHS successfully, with maximum inequality/equality
residuals below 6.2e-10. A GLPK LP reported as `infeasible` also re-solves
successfully through independent HiGHS, with residuals below 5.9e-10. Full LP
JSON snapshots and diagnoses are saved as `recon1_failed_lp*.json` in this audit
directory. These checks establish numerical solver/interface failures rather
than an infeasible target; no failed LP has been accepted or ignored.

A conservative PolyRound retry sets `reduce=False`. This bypasses the
redundancy-testing perturbation LPs, retains redundant facets, and still
identifies effectively fixed directions, eliminates equalities and rounds.
It therefore preserves the original target while potentially increasing the
number of retained constraint rows and the cost of subsequent sampling. This
trial uses GLPK with the same 1e-9 feasibility tolerance, no presolve, and
separate output directory `RECON1_conservative`. Its status and all attempts
are in `results/setup/RECON1_preprocessing_attempts.json`.

The normal GLPK-presolve retry completed in 26 minutes with n=3,741, d=931,
m=2,120 and PolyRound time 1,549.53 seconds. The conservative retry timed out
after 20 minutes during rounding; its redundant loose facets kept PolyRound's
internal `delta_s` distortion guard large, so it was not substituted.

Independent validation of the completed normal bundle failed the strict 1e-7
bound gate narrowly: the center has maximum violation 1.14054e-7 and a small
pilot has maximum violation 1.52823e-7. Equality checks pass at that tolerance.
The strict result is preserved as `data/models/RECON1/validation.strict1e-7.json`.
The benchmark coordinator explicitly approved a 1e-6 **preprocessing** gate;
the recheck passed, and canonical `validation.json` records
`strict_1e_minus_7_passed=false` and the precision caveat. No bounds or arrays
were changed. Common sample diagnostics still use 1e-7 and zero tolerance,
so reports must show resulting violations rather than claiming strict zero
violations for this model. Only the validation marker was atomically published
after confirming the preprocessing process had completed.
