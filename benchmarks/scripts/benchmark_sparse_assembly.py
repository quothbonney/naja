#!/usr/bin/env python3
"""Time coefficient assembly and verify identical inequality/equality LP matrices."""
import json,platform,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from PolyRound.static_classes.lp_interfacing import OptlangInterfacer,solvers
from fast_sparse import bulk_constraint_system

original=OptlangInterfacer.make_sparse_constraint_system
def build(function,A,b,equality=False):
    interface=solvers['glpk'];model=interface.Model()
    variables=np.array([interface.Variable(f'x{i}',lb=None) for i in range(A.shape[1])])
    model.add(variables);model.update()
    start=time.perf_counter();function(model,A,b,variables,'glpk',equality)
    return model,time.perf_counter()-start

rng=np.random.default_rng(823)
small=rng.normal(size=(50,20));small[np.abs(small)<.2]=0
b=pd.Series(np.ones(50),index=[f'c{i}' for i in range(50)])
checks=[]
for eq in (False,True):
    old,_=build(original,small,b,eq);new,_=build(bulk_constraint_system,small,b,eq)
    sense='=' if eq else '<'
    a0,b0=OptlangInterfacer.constraints_as_mat(old,sense)
    a1,b1=OptlangInterfacer.constraints_as_mat(new,sense)
    checks.append(bool(np.array_equal(a0.to_numpy(),a1.to_numpy()) and np.array_equal(b0.to_numpy(),b1.to_numpy())))
dense=rng.normal(size=(1200,400));rhs=pd.Series(np.ones(1200),index=[f'r{i}' for i in range(1200)])
old,slow=build(original,dense,rhs);del old
new,fast=build(bulk_constraint_system,dense,rhs);del new
result=dict(status='passed' if all(checks) else 'failed',hostname=platform.node(),
            small_inequality_equality_exact=checks,shape=list(dense.shape),
            original_seconds=slow,bulk_seconds=fast,speedup=slow/fast,
            note='Coefficient assembly only; solver settings and coefficient values unchanged.')
(root/'audit/fast_sparse_assembly.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2));raise SystemExit(not all(checks))
