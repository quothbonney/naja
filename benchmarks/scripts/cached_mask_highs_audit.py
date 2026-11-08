"""Independent numerical LP range audit; never modifies the source model/mask."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import highspy
import numpy as np
from scipy.sparse import csc_matrix

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--parent', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--limit', type=int)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

parent = json.loads(a.parent.read_text())
bundle = a.model/'rounding/polytope.npz'
assert parent['geometry_sha256'] == sha(bundle)
with np.load(bundle) as z:
    A, b, T = z['A'], z['b'].reshape(-1), z['T']
norm = np.linalg.norm(A, axis=1)
keep = norm > 0
assert np.all(b[~keep] >= 0)
Q, rhs = A[keep]/norm[keep, None], b[keep]/norm[keep]
sparse = csc_matrix(Q)
lp = highspy.HighsLp()
lp.num_col_, lp.num_row_ = Q.shape[1], Q.shape[0]
lp.col_cost_ = np.zeros(Q.shape[1])
lp.col_lower_ = np.full(Q.shape[1], -highspy.kHighsInf)
lp.col_upper_ = np.full(Q.shape[1], highspy.kHighsInf)
lp.row_lower_, lp.row_upper_ = np.full(Q.shape[0], -highspy.kHighsInf), rhs
lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = sparse.indptr, sparse.indices, sparse.data
h = highspy.Highs()
options = dict(output_flag=False, threads=1, solver='simplex', presolve='on',
               small_matrix_value=1e-12, primal_feasibility_tolerance=1e-9,
               dual_feasibility_tolerance=1e-9, time_limit=5.)
for name, value in options.items():
    assert h.setOptionValue(name, value) == highspy.HighsStatus.kOk
assert h.passModel(lp) in (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning)
indices = np.arange(Q.shape[1], dtype=np.int32)
def solve(c):
    # HiGHS retains cumulative solver time when objectives change in-place.
    assert h.setOptionValue('time_limit', h.getRunTime()+5.) == highspy.HighsStatus.kOk
    assert h.changeColsCost(len(c), indices, c) == highspy.HighsStatus.kOk
    start = time.perf_counter()
    status = h.run()
    solution = h.getSolution()
    x, dual = np.asarray(solution.col_value), np.asarray(solution.row_dual)
    value = float(c @ x)
    primal = float(np.maximum(A@x-b, 0).max())
    scaled_primal = float(np.maximum(Q@x-rhs, 0).max())
    stationarity = float(np.abs(Q.T@dual-c).max())
    dual_sign = float(np.maximum(dual, 0).max())
    gap = float(abs(value-rhs@dual))
    valid = (status == highspy.HighsStatus.kOk and h.getModelStatus() == highspy.HighsModelStatus.kOptimal
             and primal <= 1e-7 and scaled_primal <= 1e-8 and stationarity <= 1e-7
             and dual_sign <= 1e-9 and gap <= 1e-7*max(1., abs(value)))
    return dict(objective=value, valid=bool(valid), status=str(h.getModelStatus()),
                seconds=time.perf_counter()-start, max_original_primal_violation=primal,
                max_scaled_primal_violation=scaled_primal, max_dual_stationarity_error=stationarity,
                max_dual_sign_error=dual_sign, primal_dual_gap=gap)

rows = [r for r in parent['reactions'] if not r['nearconstant_excluded']]
if a.limit:
    rows = rows[:a.limit]
records = []
for record in rows:
    i = record['index']
    scale = float(np.abs(T[i]).max())
    assert scale > 0
    lo, hi = solve(T[i]/scale), solve(-T[i]/scale)
    width = scale*(-hi['objective']-lo['objective'])
    valid = lo['valid'] and hi['valid'] and -1e-7 <= width <= 1e-7
    row = dict(index=i, reaction=record['reaction'], objective_scale=scale, minimum=lo,
               maximum=hi, estimated_range=width, numerical_nearconstant_certificate=bool(valid))
    records.append(row)
    with (a.output/'progress.jsonl').open('a') as f:
        f.write(json.dumps(row, allow_nan=False)+'\n')
    print(i, record['reaction'], valid, width, lo['status'], hi['status'], flush=True)
    save(a.output/'summary.json', dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        geometry_sha256=sha(bundle), parent_certificate=str(a.parent), parent_sha256=sha(a.parent),
        solver_version=h.version(), options=options, code_sha256=sha(Path(__file__)),
        tiny_coefficients_below_cutoff=int(np.count_nonzero((abs(sparse.data)>0)&(abs(sparse.data)<1e-12))),
        policy='Independent numerical optimal-LP cross-check on row-normalized cached A/b; original primal and full-matrix dual residuals checked. No model/mask mutation. These are floating-point numerical certificates, not exact rational proofs.',
        candidates_planned=len(rows), candidates_completed=len(records),
        newly_numerically_certified=sum(r['numerical_nearconstant_certificate'] for r in records), records=records))
