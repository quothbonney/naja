#!/usr/bin/env python3
"""Capture the first failing simplification LP and independently check feasibility.

Pass the usual cpu_baseline.py prep arguments. This observes failures without
changing the algorithm or accepting solver errors.
"""
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from PolyRound.static_classes.lp_interfacing import OptlangInterfacer

root=Path(__file__).resolve().parents[1]
original=OptlangInterfacer.get_opt
captured=False


def observed_get_opt(model,settings):
    global captured
    if model.status!='optimal' and not captured:
        captured=True
        report=dict(optlang_status=model.status,backend=settings.backend,
                    native_status=str(getattr(model.problem,'status',None)),
                    variables=len(model.variables),constraints=len(model.constraints))
        prefix=root/'audit'/f'recon1_failed_lp_{settings.backend}'
        prefix.with_suffix('.json').write_text(json.dumps(model.to_json()))
        poly=OptlangInterfacer.optlang_to_polytope(model)
        A=sparse.csr_matrix(poly.A.values)
        b=poly.b.values
        S=sparse.csr_matrix(poly.S.values) if poly.S is not None else None
        h=poly.h.values if poly.h is not None else None
        t=time.perf_counter()
        check=linprog(np.zeros(A.shape[1]),A_ub=A,b_ub=b,A_eq=S,b_eq=h,
                      bounds=(None,None),method='highs-ds',
                      options={'primal_feasibility_tolerance':1e-9,
                               'dual_feasibility_tolerance':1e-9})
        report.update(independent_status=check.status,independent_success=check.success,
                      independent_message=check.message,
                      independent_wall_seconds=time.perf_counter()-t)
        if check.x is not None:
            report['max_independent_inequality_violation']=float(max(0,np.max(A@check.x-b)))
            report['max_independent_equality_residual']=float(np.max(np.abs(S@check.x-h))) if S is not None else 0.
        print('FAILED_LP_DIAGNOSIS '+json.dumps(report),flush=True)
        prefix.with_name(prefix.name+'_diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
    return original(model,settings)


OptlangInterfacer.get_opt=staticmethod(observed_get_opt)
spec=importlib.util.spec_from_file_location('cpu_baseline',root/'cpu_baseline.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.main()
