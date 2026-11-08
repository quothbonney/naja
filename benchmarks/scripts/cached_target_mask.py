"""LP-nearconstant mask on the exact cached reduced target, with warm GLPK basis."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import swiglpk as glp
import numpy as np
from scipy import sparse

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def build(model, candidate_tolerance=1e-6, range_tolerance=1e-7):
    model=Path(model).resolve();started=time.perf_counter()
    z=np.load(model/'rounding/polytope.npz');A=z['A'];b=z['b'].reshape(-1);T=z['T'];shift=z['shift'].reshape(-1)
    norm=np.linalg.norm(A,axis=1);nonzero=norm>0
    assert np.all(b[~nonzero]>=0)
    matrix=sparse.csr_matrix(A[nonzero]/norm[nonzero,None])
    problem=glp.glp_create_prob();glp.glp_add_rows(problem,matrix.shape[0]);glp.glp_add_cols(problem,matrix.shape[1])
    for i,value in enumerate(b[nonzero]/norm[nonzero],1):glp.glp_set_row_bnds(problem,i,glp.GLP_UP,0.,float(value))
    for j in range(1,matrix.shape[1]+1):glp.glp_set_col_bnds(problem,j,glp.GLP_FR,0.,0.)
    coo=matrix.tocoo();ia=glp.intArray(len(coo.data)+1);ja=glp.intArray(len(coo.data)+1);ar=glp.doubleArray(len(coo.data)+1)
    for k,(i,j,value) in enumerate(zip(coo.row,coo.col,coo.data),1):ia[k]=int(i)+1;ja[k]=int(j)+1;ar[k]=float(value)
    glp.glp_load_matrix(problem,len(coo.data),ia,ja,ar);glp.glp_set_obj_dir(problem,glp.GLP_MIN)
    parameters=glp.glp_smcp();glp.glp_init_smcp(parameters);parameters.msg_lev=glp.GLP_MSG_OFF
    parameters.tol_bnd=1e-9;parameters.tol_dj=1e-9;parameters.presolve=glp.GLP_OFF
    assert glp.glp_simplex(problem,parameters)==0 and glp.glp_get_status(problem)==glp.GLP_OPT
    def solve(c):
        for j,value in enumerate(c,1):glp.glp_set_obj_coef(problem,j,float(value))
        ret=glp.glp_simplex(problem,parameters);status=glp.glp_get_status(problem)
        x=np.array([glp.glp_get_col_prim(problem,j) for j in range(1,A.shape[1]+1)])
        feasible=float(np.max(A@x-b))<=1e-7
        return glp.glp_get_obj_val(problem),bool(ret==0 and status==glp.GLP_OPT and feasible),str(status)
    ids=(model/'gem/reaction_ids.txt').read_text().splitlines()
    active=np.ones(len(ids),bool);records=[];indices=np.arange(A.shape[1],dtype=np.int32)
    for i in np.flatnonzero(np.max(np.abs(T),axis=1)<=candidate_tolerance):
        c=T[i];scale=float(np.abs(c).max())
        record=dict(index=int(i),reaction=ids[i],T_row_max=scale)
        if scale==0:
            lo=hi=float(shift[i]);valid=True;record['certificate']='exact zero affine row; no LP required'
        else:
            lower,okmin,smin=solve(c/scale);upper,okmax,smax=solve(-c/scale)
            lo=lower*scale+shift[i];hi=-upper*scale+shift[i]
            valid=okmin and okmax and np.isfinite([lo,hi]).all() and hi-lo>=-range_tolerance
            record.update(minimum_status=smin,maximum_status=smax,certificate='Exact cached A/b, positive row normalization and objective scaling; warm GLPK basis reused; raw primal residual checked')
        excluded=bool(valid and hi-lo<=range_tolerance);active[i]=not excluded
        record.update(minimum=float(lo),maximum=float(hi),range=float(hi-lo),nearconstant_excluded=excluded)
        records.append(record)
    meta=json.loads((model/'meta.json').read_text());source=Path(meta['source'])
    report=dict(schema_version=1,model=model.name,source=str(source),source_sha256=sha(source),geometry_sha256=sha(model/'rounding/polytope.npz'),
        solver='GLPK on exact cached reduced polytope',target_geometry_kind='exact_cached_polytope',solver_feasibility_tolerance=1e-9,range_tolerance=range_tolerance,candidate_tolerance=candidate_tolerance,
        policy='Candidate tiny affine rows; exclude only optimal min/max range<=1e-7 on exact cached conditional A/b, or exactly zero affine rows. No target relaxation. Positive row/objective scaling only inside LP.',
        n_reactions=len(ids),candidates=len(records),active_reactions=int(active.sum()),excluded_nearconstant_reactions=int((~active).sum()),
        wall_seconds=time.perf_counter()-started,reactions=records,code_sha256=sha(Path(__file__)))
    glp.glp_delete_prob(problem)
    return active,report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();mask,report=build(a.model)
    out=a.output or a.model;out.mkdir(parents=True,exist_ok=True);np.save(out/'active_reactions.npy',mask);(out/'diagnostic_mask.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['excluded_nearconstant_reactions','candidates','wall_seconds']}))
