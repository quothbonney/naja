# Supplementary cached reaction-range LP cross-check

Job873 independently revisited all470 unresolved candidate reaction ranges
from the GLPK cached-target base certificate. HiGHS used positive row/objective
scaling,5seconds of additional cumulative solver time per LP, and retained basis
state. Every candidate was attempted under the predeclared policy.

298 candidates passed both optimal LPs plus original/scaled primal feasibility,
full-matrix dual stationarity/sign and primal-dual-gap gates, with estimated
reaction range at most1e-7. All other candidates remain unresolved; a small
estimated range alone is insufficient. Solver-small-coefficient truncation and
all residuals are explicit in summary.json. These are floating-point numerical
checks, not exact rational certificates.

This supplementary result has NOT changed any benchmark diagnostic mask.
The principal512-chain comparison uses the preexisting478-exclusion certificate
and separately reports the original878-exclusion assay. This avoids selecting
a reaction mask for its resulting performance statistics.

These are reaction-range LPs for convergence aggregation, not the starting-point
LP benchmark. Starting-point HiGHS persistent-state timing is a separate study.

First attempt861 used an incorrectly fixed cumulative HiGHS time limit; it was
cancelled and preserved in ../highs_range_audit/, including its source.py.
Attempt2 updates the cumulative limit before each objective, without relaxing
any validity gate. No original model or mask was modified.

[Every attempted candidate and residual](summary.json) · [Per-candidate log](progress.jsonl).
Source: benchmarks/scripts/cached_mask_highs_audit.py; its SHA is recorded in summary.json.
