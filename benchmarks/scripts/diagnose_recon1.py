#!/usr/bin/env python3
"""Independent feasibility checks of original RECON1 bounds/equalities."""
import json
from pathlib import Path
import platform
import time
import cobra
import numpy as np
from scipy.optimize import linprog

root = Path(__file__).resolve().parents[1]
cobra.Configuration().solver = "glpk"
model = cobra.io.load_json_model(str(root / "data/raw/RECON1.json"))
model.solver = "glpk"
S = cobra.util.array.create_stoichiometric_matrix(model, array_type="lil").tocsr()
lb = np.array([r.lower_bound for r in model.reactions])
ub = np.array([r.upper_bound for r in model.reactions])
report = dict(hostname=platform.node(), n=len(lb), metabolites=S.shape[0], checks=[])
for method in ("highs-ds", "highs-ipm"):
    t = time.perf_counter()
    result = linprog(np.zeros(len(lb)), A_eq=S, b_eq=np.zeros(S.shape[0]),
                     bounds=list(zip(lb, ub)), method=method,
                     options={"primal_feasibility_tolerance": 1e-9,
                              "dual_feasibility_tolerance": 1e-9})
    row = dict(method=method, success=result.success, status=result.status,
               message=result.message, wall_seconds=time.perf_counter()-t)
    if result.x is not None:
        row.update(max_equality_residual=float(np.max(np.abs(S @ result.x))),
                   max_bound_violation=float(max(0, np.max(lb-result.x), np.max(result.x-ub))))
    report["checks"].append(row)
for presolve in (False, True):
    model.solver.configuration.presolve = presolve
    model.solver.configuration.tolerances.feasibility = 1e-9
    t = time.perf_counter()
    result = model.optimize()
    row = dict(method="glpk", presolve=presolve, status=result.status,
               wall_seconds=time.perf_counter()-t)
    if result.status == "optimal":
        x=result.fluxes.to_numpy()
        row.update(max_equality_residual=float(np.max(np.abs(S @ x))),
                   max_bound_violation=float(max(0, np.max(lb-x), np.max(x-ub))))
    report["checks"].append(row)
print(json.dumps(report, indent=2))
(root / "audit/recon1_feasibility.json").write_text(json.dumps(report, indent=2)+"\n")
