"""Cold/steady LP and genuine HiGHS basis reuse on the existing exact slab."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import warnings
import csv
from statistics import median

B=Path(__file__).resolve().parents[1]
O=Path(os.environ.get('NAJA_LP_MICRO_OUT',str(B/'results/knockout_panel/lp_microbench'))).resolve()

def worker(case):
    started=time.perf_counter()
    import numpy as np
    from scipy.sparse import csc_matrix
    from scipy.optimize import linprog
    import highspy
    imports=time.perf_counter()-started
    t=time.perf_counter()
    source=B/'results/conditioned_prototype/iJO1366/models/shared/rounding/polytope.npz'
    with np.load(source) as p:A,b=p['A'],p['b'].reshape(-1)
    norms=np.linalg.norm(A,axis=1);keep=norms>0
    assert np.all(b[~keep]>=0)
    Q=np.column_stack([A[keep]/norms[keep,None],np.ones(keep.sum())]);rhs=b[keep]/norms[keep]
    c=np.r_[np.zeros(A.shape[1]),-1.];bounds=[(None,None)]*A.shape[1]+[(0,None)]
    assembly=time.perf_counter()-t
    report=dict(case=case,source=str(source),imports_seconds=imports,geometry_load_and_normalization_seconds=assembly,
        affinity=sorted(os.sched_getaffinity(0)),host_logical_cpus=os.cpu_count(),repetitions=[],
        solver_versions=dict(highspy=highspy.Highs().version()),scope='Existing PGI capacity slab microbenchmark; not exact gene-KO geometry.')
    def threads():return len(list(Path('/proc/self/task').iterdir()))
    if case=='scipy_default':
        for i in range(5):
            t=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                r=linprog(c,A_ub=Q,b_ub=rhs,bounds=bounds,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
            elapsed=time.perf_counter()-t;assert r.success,r.message
            original_violation=float(np.maximum(A@r.x[:-1]-b,0).max());assert original_violation<=1e-7
            report['repetitions'].append(dict(repetition=i,role='first_call' if i==0 else 'repeat_same_LP_new_model',seconds=elapsed,iterations=r.nit,radius=r.x[-1],threads_after=threads(),warnings=[str(w.message) for w in caught],max_original_Ay_minus_b_violation=original_violation))
    else:
        requested={'highs_default':0,'highs1':1,'highs4':4}[case]
        t=time.perf_counter();sparse=csc_matrix(Q)
        lp=highspy.HighsLp();lp.num_col_=Q.shape[1];lp.num_row_=Q.shape[0]
        lp.col_cost_=c;lp.col_lower_=np.r_[np.full(A.shape[1],-highspy.kHighsInf),0.];lp.col_upper_=np.full(Q.shape[1],highspy.kHighsInf)
        lp.row_lower_=np.full(Q.shape[0],-highspy.kHighsInf);lp.row_upper_=rhs
        lp.a_matrix_.format_=highspy.MatrixFormat.kColwise;lp.a_matrix_.start_=sparse.indptr;lp.a_matrix_.index_=sparse.indices;lp.a_matrix_.value_=sparse.data
        report['one_time_CSC_model_assembly_seconds']=time.perf_counter()-t
        report['solver_small_matrix_value']=1e-12
        report['normalized_nonzero_coefficients_below_solver_cutoff']=int(np.count_nonzero((np.abs(sparse.data)>0)&(np.abs(sparse.data)<1e-12)))
        model_statuses=[]
        def make():
            t=time.perf_counter();h=highspy.Highs()
            for name,value in [('output_flag',True),('log_to_console',False),('log_file',str(O/(case+'-solver.log'))),('small_matrix_value',1e-12),('solver','simplex'),('threads',requested),('primal_feasibility_tolerance',1e-9),('dual_feasibility_tolerance',1e-9)]:
                status=h.setOptionValue(name,value);assert status==highspy.HighsStatus.kOk,(name,status)
            status=h.passModel(lp);model_statuses.append(str(status))
            assert status in (highspy.HighsStatus.kOk,highspy.HighsStatus.kWarning),status
            return h,time.perf_counter()-t
        h,construction=make();report['first_model_construction_seconds']=construction
        report['requested_threads']=requested;report['accepted_threads_option']=h.getOptionValue('threads')[1]
        def solve(h,role,i,model_construction=0.,current_rhs=rhs):
            t=time.perf_counter();status=h.run();elapsed=time.perf_counter()-t
            assert status==highspy.HighsStatus.kOk and h.getModelStatus()==highspy.HighsModelStatus.kOptimal
            sol=np.asarray(h.getSolution().col_value);basis=h.getBasis();assert basis.valid
            row=dict(role=role,repetition=i,seconds=elapsed,model_construction_seconds=model_construction,
                iterations=h.getInfo().simplex_iteration_count,radius=float(sol[-1]),threads_after=threads(),
                valid_basis=True,max_scaled_primal_violation=float(np.maximum(Q@sol-current_rhs,0).max()))
            row['max_original_Ay_minus_b_violation']=float(np.maximum(A[keep]@sol[:-1]-current_rhs*norms[keep],0).max())
            assert row['max_original_Ay_minus_b_violation']<=1e-7 and row['max_scaled_primal_violation']<=1e-8,row
            report['repetitions'].append(row);return basis
        basis=solve(h,'cold_first_solver_call',0,construction)
        for i in range(4):
            assert h.getBasis().valid
            basis=solve(h,'same_model_internal_basis_without_reset',i)
        for i in range(1,5):
            assert h.setBasis(basis)==highspy.HighsStatus.kOk
            basis=solve(h,'same_model_explicit_basis_same_target',i)
        for i in range(4):
            cold,construction=make();solve(cold,'new_model_cold_basis_pool_already_initialized',i,construction)
        # Actual neighboring conditions: same matrix, PGI slab bounds change.
        # Basis objects survive bound updates; this is more than an initial guess.
        for i,capacity in enumerate([0.09,0.11,0.08,0.1]):
            changed=rhs.copy();changed[-2:]+=(capacity-0.1)/norms[keep][-2:]
            for row in [-2,-1]:
                j=len(changed)+row
                assert h.changeRowBounds(j,-highspy.kHighsInf,float(changed[j]))==highspy.HighsStatus.kOk
            assert h.setBasis(basis)==highspy.HighsStatus.kOk
            basis=solve(h,'cached_basis_changed_capacity_target',i,current_rhs=changed)
        for i,capacity in enumerate([0.09,0.11,0.08,0.1]):
            changed=rhs.copy();changed[-2:]+=(capacity-0.1)/norms[keep][-2:]
            for row in [-2,-1]:
                j=len(changed)+row
                assert h.changeRowBounds(j,-highspy.kHighsInf,float(changed[j]))==highspy.HighsStatus.kOk
            assert h.getBasis().valid
            basis=solve(h,'cached_internal_basis_changed_capacity_no_reset',i,current_rhs=changed)
        report['model_pass_statuses']=model_statuses
        report['solver_warnings']=[line for line in (O/(case+'-solver.log')).read_text().splitlines() if 'WARNING' in line]
    (O/(case+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(case,[(r['role'],round(r['seconds'],6)) for r in report['repetitions']],flush=True)

def main():
    O.mkdir(parents=True,exist_ok=True)
    if len(sys.argv)>1:worker(sys.argv[1]);return
    for case in ['scipy_default','highs_default','highs1','highs4']:
        if (O/(case+'.json')).exists():continue
        with (O/(case+'.log')).open('w') as f:subprocess.run([sys.executable,__file__,case],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
    results=[json.loads((O/(case+'.json')).read_text()) for case in ['scipy_default','highs_default','highs1','highs4']]
    (O/'summary.json').write_text(json.dumps(dict(cases=results,note='Four Slurm CPUs reserved; requested default threading is recorded rather than treated as128physicalcores. Same-target cached solves can trivially reuse an optimal solution; changed-capacity rows demonstrate actual retained-basis reoptimization.'),indent=2)+'\n')
    rows=[]
    for case in results:
        for role in dict.fromkeys(r['role'] for r in case['repetitions']):
            group=[r for r in case['repetitions'] if r['role']==role]
            rows.append(dict(case=case['case'],role=role,median_s=median(r['seconds'] for r in group),runs=len(group),iterations=[r['iterations'] for r in group],threads_after=sorted({r['threads_after'] for r in group})))
    with (O/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Completed LP microbenchmark',flush=True)

if __name__=='__main__':main()
