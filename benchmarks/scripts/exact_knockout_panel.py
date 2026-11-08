"""Real GPR knockout catalog and certified affine/facial reductions of cached581-D geometry."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import time
import numpy as np
from scipy.linalg import svd
from scipy.sparse import csc_matrix, hstack, vstack, eye
import highspy

B=Path(__file__).resolve().parents[1]
R=Path(os.environ.get('NAJA_KO_PANEL_OUT',str(B/'results/knockout_panel'))).resolve()
BASE=B/'data/models/iJO1366'
RAW=B/'data/raw/iJO1366.json'
BINS=[('0',0,0),('1-5',1,5),('6-20',6,20),('21-50',21,50),('51-200',51,200)]
RANK_TOL=1e-12
PRIMAL_TOL=1e-7

def read(p):return json.loads(p.read_text())
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def binname(k):return next((n for n,a,b in BINS if a<=k<=b),'outside')

def catalog():
    import cobra
    R.mkdir(parents=True,exist_ok=True)
    model=cobra.io.load_json_model(str(RAW))
    ids=(BASE/'gem/reaction_ids.txt').read_text().splitlines();idx={s:i for i,s in enumerate(ids)}
    with np.load(BASE/'rounding/polytope.npz') as p:T,shift,start,A,b=p['T'],p['shift'].reshape(-1),p['start'],p['A'],p['b'].reshape(-1)
    a_norms=np.linalg.norm(A,axis=1);positive=a_norms>0
    base_radius=float(np.min((b-A@start)[positive]/a_norms[positive]));T_norms=np.linalg.norm(T,axis=1)
    active=np.load(BASE/'active_reactions.npy');flux=T@start+shift
    genes=[];groups={}
    for gene in sorted(model.genes,key=lambda g:g.id):
        if not gene.id.startswith('b'):continue # s0001 is the spontaneous-reaction placeholder.
        disabled=sorted([r for r in gene.reactions if not r.gpr.eval({gene.id})],key=lambda r:idx[r.id])
        changed=[r for r in disabled if r.lower_bound!=0 or r.upper_bound!=0]
        reaction_ids=[r.id for r in changed];indices=[idx[s] for s in reaction_ids]
        k=sum(int(r.lower_bound!=0)+int(r.upper_bound!=0) for r in changed)
        varying=int(np.count_nonzero(active[indices]));norms=np.linalg.norm(T[indices],axis=1)
        residual=float(np.max(np.abs(flux[indices]))) if indices else 0.
        row=dict(gene=gene.id,gene_name=gene.name,disabled_reaction_ids=[r.id for r in disabled],
            knockout_reaction_ids=reaction_ids,changed_bound_rows=k,two_sided_encoding_rows=2*len(changed),
            stratum=binname(k),base_LP_varying_reactions=varying,base_LP_nearconstant_reactions=len(indices)-varying,
            all_affected_reactions_varying=bool(indices and varying==len(indices)),
            base_point_max_KO_residual=residual,base_point_weak_feasible_at_1e7=residual<=PRIMAL_TOL,
            minimum_KO_T_row_norm=float(norms.min()) if len(norms) else None,
            original_changed_bounds={r.id:[float(r.lower_bound),float(r.upper_bound)] for r in changed})
        row['cached_target_analytically_varying_reactions']=[ids[i] for i in indices if 2*base_radius*T_norms[i]>1e-7]
        genes.append(row);groups.setdefault(tuple(reaction_ids),[]).append(row)
    signatures=[]
    for signature,aliases in groups.items():
        row=dict(aliases[0]);row['genes_with_identical_bound_change_signature']=[g['gene'] for g in aliases]
        row['bound_change_signature']=hashlib.sha256(json.dumps(signature).encode()).hexdigest()
        signatures.append(row)
    signatures.sort(key=lambda r:r['gene'])
    clean=[r for r in signatures if r['all_affected_reactions_varying'] and r['minimum_KO_T_row_norm']>1e-9]
    control=next(r for r in signatures if r['changed_bound_rows']==0)
    selected=[control]
    for name,quota in [('1-5',20),('6-20',20),('21-50',9)]:
        pool=[r for r in clean if r['stratum']==name]
        picks=np.unique(np.linspace(0,len(pool)-1,min(quota,len(pool)),dtype=int)) if pool else []
        selected.extend(pool[int(i)] for i in picks)
    pgi=next(r for r in clean if r['gene']=='b4025')
    if not any(r['gene']=='b4025' for r in selected):selected.append(pgi)
    for row in clean:
        if len(selected)>=50:break
        if row['bound_change_signature'] not in {r['bound_change_signature'] for r in selected}:selected.append(row)
    if len(selected)>50:
        removable=next(i for i in range(len(selected)-1,0,-1) if selected[i]['gene']!='b4025')
        selected.pop(removable)
    top=max(selected,key=lambda r:r['changed_bound_rows'])
    pilot=[control,pgi,next(r for r in selected if r['stratum']=='6-20'),next(r for r in selected if r['stratum']=='21-50'),top]
    pilot=list({r['gene']:r for r in pilot}.values())
    availability={name:dict(real_genes=sum(r['stratum']==name for r in genes),
        unique_bound_change_signatures=sum(r['stratum']==name for r in signatures),
        clean_nontrivial_signatures=sum(r['stratum']==name for r in clean)) for name,_,_ in BINS}
    nontrivial=[r for r in genes if r['base_LP_varying_reactions']>0]
    report=dict(real_gene_count=len(genes),excluded_gene_placeholders=[g.id for g in model.genes if not g.id.startswith('b')],
        source_sha256=sha(RAW),base_geometry_sha256=sha(BASE/'rounding/polytope.npz'),base_dimension=581,
        stratum_definition='Number of original reaction bound sides changed to0; do not inflate with duplicate unchanged sides.',
        availability=availability,unique_bound_change_signatures=len(signatures),genes=genes,
        all_real_gene_base_point_weak_feasible_fraction=sum(r['base_point_weak_feasible_at_1e7'] for r in genes)/len(genes),
        nontrivial_varying_gene_count=len(nontrivial),nontrivial_gene_base_point_weak_feasible_fraction=sum(r['base_point_weak_feasible_at_1e7'] for r in nontrivial)/len(nontrivial),
        precision_contract='Original floating cached581-D model; numerical rank1e-12, original-bound/KO validation1e-7. No bound or delta epsilon relaxation. LP-nearconstant-only genes are catalogued, not silently zeroed into the pilot.',
        signatures=signatures)
    proven=[r for r in genes if r['cached_target_analytically_varying_reactions']]
    report['cached_analytic_variation_certificate']=dict(base_strict_ball_radius=base_radius,
        width_lower_bound='2*r_base*norm(T_reaction); both opposite ball points satisfy every cached A/b inequality',
        reaction_width_threshold=1e-7,certified_nontrivial_real_genes=len(proven),
        weak_feasible_genes_in_that_group=sum(r['base_point_weak_feasible_at_1e7'] for r in proven),
        weak_feasible_fraction=sum(r['base_point_weak_feasible_at_1e7'] for r in proven)/len(proven))
    report['selection_mask_provenance']='Prior raw-model LP mask is candidate classification only, not an exact cached-target certificate; never inherited into conditional masks.'
    save(R/'gene_catalog.json',report)
    manifest=dict(status='catalogued',source=str(RAW),source_sha256=sha(RAW),base_model=str(BASE),
        base_geometry_sha256=report['base_geometry_sha256'],availability=availability,pilot_genes=[r['gene'] for r in pilot],
        conditions=selected,selection='50 distinct bound-change signatures, one zero-effect control; nonzero cases first require every affected reaction to be LP-varying and nondegenerate in cached geometry. Final affine-face duplicates are reported separately.',
        geometry_directory_template=str(R/'conditions/<gene>/models/shared'))
    save(R/'panel_manifest.json',manifest)
    return manifest

def lp_radius(A,b,out,round_index):
    norms=np.linalg.norm(A,axis=1);assert np.all(norms>0)
    Q=A/norms[:,None];rhs=b/norms;matrix=csc_matrix(np.column_stack([Q,np.ones(len(b))]))
    lp=highspy.HighsLp();lp.num_col_=A.shape[1]+1;lp.num_row_=len(b)
    lp.col_cost_=np.r_[np.zeros(A.shape[1]),-1.];lp.col_lower_=np.full(lp.num_col_,-highspy.kHighsInf);lp.col_upper_=np.full(lp.num_col_,highspy.kHighsInf)
    lp.row_lower_=np.full(len(b),-highspy.kHighsInf);lp.row_upper_=rhs
    lp.a_matrix_.format_=highspy.MatrixFormat.kColwise;lp.a_matrix_.start_=matrix.indptr;lp.a_matrix_.index_=matrix.indices;lp.a_matrix_.value_=matrix.data
    h=highspy.Highs();t=time.perf_counter()
    for name,value in [('output_flag',True),('log_to_console',False),('log_file',str(out/f'LP-{round_index}.log')),('threads',1),('solver','simplex'),('small_matrix_value',1e-12),('primal_feasibility_tolerance',1e-9),('dual_feasibility_tolerance',1e-9),('time_limit',30.)]:
        assert h.setOptionValue(name,value)==highspy.HighsStatus.kOk,(name,value)
    passed=h.passModel(lp);assert passed in (highspy.HighsStatus.kOk,highspy.HighsStatus.kWarning)
    construction=time.perf_counter()-t;t=time.perf_counter();status=h.run();elapsed=time.perf_counter()-t
    record=dict(round=round_index,model_construction_seconds=construction,solve_seconds=elapsed,pass_status=str(passed),run_status=str(status),model_status=str(h.getModelStatus()),iterations=h.getInfo().simplex_iteration_count)
    if h.getModelStatus()!=highspy.HighsModelStatus.kOptimal:return None,None,record
    sol=np.asarray(h.getSolution().col_value);dual=-np.asarray(h.getSolution().row_dual)
    record.update(radius=float(sol[-1]),max_scaled_primal_violation=float(np.maximum(Q@sol[:-1]+sol[-1]-rhs,0).max()))
    assert record['max_scaled_primal_violation']<=1e-7,record
    return sol,dual,record

def affine_basis(E,f,start):
    if not len(E):return start.copy(),np.eye(len(start)),dict(rank=0,singular_values=[],scaled_equality_residual=0.)
    norms=np.linalg.norm(E,axis=1)
    if np.any(norms==0):raise ValueError('Exact zero KO equation requires separate redundancy certificate')
    scaled=E/norms[:,None];rhs=f/norms
    U,s,V=svd(scaled,full_matrices=True,check_finite=True)
    rank=int(np.count_nonzero(s>RANK_TOL*s[0]));project=start+V[:rank].T@((U[:,:rank].T@(rhs-scaled@start))/s[:rank]);N=V[rank:].T
    residual=float(np.max(np.abs(scaled@project-rhs)))
    assert residual<=1e-8,('Inconsistent or precision-ambiguous exact KO equations',residual)
    return project,N,dict(rank=rank,singular_values=s.tolist(),relative_rank_threshold=RANK_TOL,
        ranks_at_1e11_1e12_1e13=[int(np.count_nonzero(s>v*s[0])) for v in [1e-11,1e-12,1e-13]],
        scaled_equality_residual=residual,nullspace_residual=float(np.max(np.abs(scaled@N))) if N.size else 0.)

def broad_exposing_dual(Q,rhs,out,iteration):
    """Seek broad positive support of a valid exposing vector, not active rows."""
    m,d=Q.shape;delta=1e-6
    equations=np.vstack([Q.T,rhs,np.ones(m)])
    scales=np.maximum(np.max(np.abs(equations),axis=1),1e-300)
    equalities=csc_matrix(equations/scales[:,None])
    matrix=vstack([hstack([equalities,csc_matrix((d+2,m))]),hstack([-eye(m),delta*eye(m)])],format='csc')
    lp=highspy.HighsLp();lp.num_col_=2*m;lp.num_row_=d+2+m
    lp.col_cost_=np.r_[np.zeros(m),-delta*np.ones(m)];lp.col_lower_=np.zeros(2*m);lp.col_upper_=np.ones(2*m)
    lp.row_lower_=np.r_[np.zeros(d+1),1./scales[-1],np.full(m,-highspy.kHighsInf)]
    lp.row_upper_=np.r_[np.zeros(d+1),1./scales[-1],np.zeros(m)]
    lp.a_matrix_.format_=highspy.MatrixFormat.kColwise;lp.a_matrix_.start_=matrix.indptr;lp.a_matrix_.index_=matrix.indices;lp.a_matrix_.value_=matrix.data
    h=highspy.Highs()
    for name,value in [('output_flag',True),('log_to_console',False),('log_file',str(out/f'dual-support-{iteration}.log')),('threads',1),('solver','simplex'),('small_matrix_value',1e-12),('primal_feasibility_tolerance',1e-9),('dual_feasibility_tolerance',1e-9),('time_limit',30.)]:
        assert h.setOptionValue(name,value)==highspy.HighsStatus.kOk
    passed=h.passModel(lp);assert passed in (highspy.HighsStatus.kOk,highspy.HighsStatus.kWarning)
    t=time.perf_counter();status=h.run();elapsed=time.perf_counter()-t
    record=dict(role='broad_support_exposing_dual',seconds=elapsed,delta=delta,objective_scale=delta,
        equation_row_scaling_min=float(scales.min()),equation_row_scaling_max=float(scales.max()),status=str(status),model_status=str(h.getModelStatus()))
    if h.getModelStatus()!=highspy.HighsModelStatus.kOptimal:return None,record
    sol=np.asarray(h.getSolution().col_value);raw=sol[:m];weights=np.maximum(raw,0)
    mass=float(weights.sum())
    if mass<=0:return None,record
    weights/=mass
    cancel=float(np.max(np.abs(Q.T@weights)));b_cancel=float(abs(rhs@weights))
    record.update(raw_weight_min=float(raw.min()),raw_weight_sum=float(raw.sum()),support_objective=float(sol[m:].sum()),
        cancellation_A_inf=cancel,cancellation_b_abs=b_cancel,positive_rows=int(np.count_nonzero(weights>1e-8)))
    if raw.min() < -1e-9 or abs(raw.sum()-1)>1e-7 or cancel>1e-8 or b_cancel>1e-8:return None,record
    return weights,record

def prepare(condition):
    gene=condition['gene'];out=R/'conditions'/gene;done=out/'condition.json'
    if done.exists() and read(done).get('status') in ['ready','infeasible','problematic','precision_ambiguous','zero_dimensional']:return read(done)
    out.mkdir(parents=True,exist_ok=True);begun=time.perf_counter()
    with np.load(BASE/'rounding/polytope.npz') as p:A0,b0,T0,shift0,start0=(p[k].copy() for k in ['A','b','T','shift','start'])
    b0=b0.reshape(-1);shift0=shift0.reshape(-1)
    ids=(BASE/'gem/reaction_ids.txt').read_text().splitlines();index={s:i for i,s in enumerate(ids)}
    ko=[index[s] for s in condition['knockout_reaction_ids']]
    E=T0[ko].copy();f=-shift0[ko].copy();facial=[];LPs=[];ranks=[];base_rows=np.arange(len(b0));start=None
    result=dict(**condition,status='preparing',base_geometry_sha256=sha(BASE/'rounding/polytope.npz'),
        base_dimension=581,base_point_weak_feasible=condition['base_point_weak_feasible_at_1e7'],
        constraint_epsilon=0.,rank_relative_tolerance=RANK_TOL,validation_tolerance=PRIMAL_TOL,
        geometry_directory=str(out/'models/shared'),LPs=LPs,facial_reductions=facial)
    save(done,result)
    for iteration in range(24):
        t=time.perf_counter();z0,N,rank=affine_basis(E,f,start0);rank['seconds']=time.perf_counter()-t;ranks.append(rank)
        if len(set(rank.get('ranks_at_1e11_1e12_1e13',[0])))>1:
            result.update(status='precision_ambiguous',reason='Affine rank changes across1e-11/1e-12/1e-13; not released as exact same-target geometry');break
        A=A0@N;b=b0-A0@z0;original_norms=np.linalg.norm(A0,axis=1);norms=np.linalg.norm(A,axis=1)
        # A row in the explicit equality span is constant on this affine hull.
        zero=norms<=2e-12*np.maximum(original_norms,1e-300)
        relative_constants=b/np.maximum(original_norms,1e-300)
        if np.any(zero & (relative_constants < -1e-8)):
            result.update(status='infeasible',reason='KO affine hull contradicts a certified constant base inequality');break
        removed=[dict(base_row=int(i),projected_norm=float(norms[i]),original_norm=float(original_norms[i]),constant_bound=float(b[i]),normalized_constant_bound=float(relative_constants[i])) for i in np.flatnonzero(zero)]
        keep=~zero;A,b=A[keep],b[keep];norms=norms[keep];kept=base_rows[keep]
        if N.shape[1]==0:
            result.update(status='problematic',reason='Zero-dimensional exact target; sampler expects positive dimension');break
        min_projected=float(np.min(b/norms))
        rank['projected_origin_min_normalized_slack']=min_projected
        if iteration==0:result['initial_projection_strict_interior']=min_projected>1e-8
        if min_projected>1e-8:
            start=np.zeros(N.shape[1]);radius=min_projected
            result.update(projected_point_strict_interior=result['initial_projection_strict_interior'],final_origin_strict_after_facial_reduction=True,
                interior_method='analytic_projection_checked_all_normalized_constraints' if iteration==0 else 'analytic_origin_after_certified_facial_reduction');break
        result.setdefault('projected_point_strict_interior',False)
        sol,dual,record=lp_radius(A,b,out,iteration);LPs.append(record);save(done,result)
        if sol is None:result.update(status='problematic',reason='Interior LP was not optimal');break
        radius=float(sol[-1])
        if radius < -1e-8:result.update(status='infeasible',reason='Normalized Chebyshev LP certifies negative maximum radius');break
        if radius>1e-8:
            start=sol[:-1];result['interior_method']='normalized_Chebyshev_LP_after_exact_affine_reduction';break
        Q=A/norms[:,None];rhs=b/norms
        raw_min=float(dual.min());assert raw_min>=-1e-9
        dual=np.maximum(dual,0);mass=float(dual.sum());assert mass>0
        weights=dual/mass
        if os.environ.get('NAJA_KO_BROAD_DUAL')=='1':
            broad,broad_record=broad_exposing_dual(Q,rhs,out,iteration)
            record['broad_exposing_dual']=broad_record
            if broad is not None:weights=(weights+broad)/2
        else:broad_record=dict(status='disabled_after_numerically_unstable_auxiliary_audit')
        cancel=float(np.max(np.abs(Q.T@weights)));objective=float(abs(rhs@weights))
        support=weights>1e-8
        certificate=dict(iteration=iteration,nonnegative=True,weight_sum=float(weights.sum()),
            dual_A_cancellation_inf=cancel,dual_b_cancellation_abs=objective,
            support_base_rows=kept[support].tolist(),support_weights=weights[support].tolist(),
            interpretation='Positive dual support is universally tight by nonnegative slack cancellation; not merely active at one vertex.')
        certificate.update(raw_Cheby_dual_min=raw_min,all_base_rows=kept.tolist(),full_weights=weights.tolist(),broad_support=broad_record)
        if cancel>1e-8 or objective>1e-8 or not np.any(support):
            result.update(status='problematic',reason='Facial exposing dual did not pass cancellation certificate',failed_certificate=certificate);break
        facial.append(certificate)
        append=kept[support];newE=np.vstack([E,A0[append]]);newf=np.r_[f,b0[append]]
        _,_,next_rank=affine_basis(newE,newf,start0)
        if next_rank['rank']<=rank['rank']:
            result.update(status='problematic',reason='Certified facial reduction did not increase affine rank');break
        E,f=newE,newf
    else:result.update(status='problematic',reason='Facial reduction iteration cap24')
    if start is None:
        result.update(preparation_seconds=time.perf_counter()-begun,affine_rank_history=ranks);save(done,result);return result
    modeldir=out/'models/shared';(modeldir/'rounding').mkdir(parents=True,exist_ok=True)
    shutil.copytree(BASE/'gem',modeldir/'gem',dirs_exist_ok=True)
    lower=np.loadtxt(modeldir/'gem/l_bounds.csv',delimiter=',').reshape(-1);upper=np.loadtxt(modeldir/'gem/u_bounds.csv',delimiter=',').reshape(-1)
    lower[ko]=0.;upper[ko]=0.
    np.savetxt(modeldir/'gem/l_bounds.csv',lower,delimiter=',');np.savetxt(modeldir/'gem/u_bounds.csv',upper,delimiter=',')
    T=T0@N;shift=T0@z0+shift0
    np.savez(modeldir/'rounding/polytope.npz',A=A,b=b,start=start,T=T,shift=shift)
    np.savez(out/'embedding.npz',z0=z0,N=N,E=E,f=f,kept_base_rows=kept)
    raw=read(RAW)
    for r in raw['reactions']:
        if r['id'] in condition['knockout_reaction_ids']:r['lower_bound']=r['upper_bound']=0.
    source=out/'conditional_model.json';save(source,raw)
    save(modeldir/'rounding/manifest.json',dict(format_version=2,model_name='shared',rounding_dim=N.shape[1],n_reactions=len(ids),n_constraints=len(b)))
    # Small ball wholly inside the final affine polytope, then original checks.
    rng=np.random.default_rng(101);directions=rng.normal(size=(32,N.shape[1]));directions/=np.linalg.norm(directions,axis=1)[:,None]
    samples=np.vstack([start,start+0.1*radius*directions]);flux=samples@T.T+shift
    original_z=z0+samples@N.T
    bound=np.maximum(np.maximum(lower-flux,flux-upper),0)
    bounds_error=float(bound.max());base_error=float(np.maximum(original_z@A0.T-b0,0).max())
    KO_error=float(np.max(np.abs(flux[:,ko]))) if ko else 0.
    assert np.isfinite(flux).all() and max(bounds_error,base_error,KO_error)<=PRIMAL_TOL,(gene,bounds_error,base_error,KO_error)
    assert float(np.min(b-A@start))>0,(gene,'not strictly interior')
    rank=ranks[-1]['rank'];projector=np.eye(581)-N@N.T;offset=projector@z0
    signature=hashlib.sha256(np.rint(projector*1e10).astype(np.int64).tobytes()+np.rint(offset*1e10).astype(np.int64).tobytes()).hexdigest()
    fixed=[ids[i] for i in range(len(ids)) if np.linalg.norm(T[i])<=1e-11*max(np.linalg.norm(T0[i]),1e-300)]
    result.update(status='ready',conditional_dimension=N.shape[1],rows=len(b),affine_rank=rank,affine_rank_history=ranks,
        strict_interior_radius=radius,min_raw_start_slack=float(np.min(b-A@start)),certified_affine_constant_base_rows=removed,
        no_bound_epsilon_relaxation=True,source=str(source),source_sha256=sha(source),
        geometry_sha256=sha(modeldir/'rounding/polytope.npz'),embedding=str(out/'embedding.npz'),
        effective_affine_face_signature_rounded_1e10=signature,fixed_reaction_ids=fixed,
        fixed_reaction_values={name:float(shift[index[name]]) for name in fixed},
        containment_proof=dict(retained_A_max_error=float(np.max(np.abs(A-A0[kept]@N))),
            retained_b_max_error=float(np.max(np.abs(b-(b0[kept]-A0[kept]@z0)))),
            dropped_rows='Equality-span residual and constant-bound certificates recorded; numerical rank contract explicit.'),
        validation=dict(points=len(samples),max_original_bound_violation=bounds_error,max_base_Az_minus_b_violation=base_error,max_KO_flux_residual=KO_error),
        preparation_seconds=time.perf_counter()-begun,LP_solve_seconds=sum(lp['solve_seconds']+lp.get('broad_exposing_dual',{}).get('seconds',0.) for lp in LPs),LP_calls=len(LPs),LP_skipped=len(LPs)==0)
    meta=read(BASE/'meta.json');meta.update(model_id='iJO1366_geneKO_'+gene,n=len(ids),d=N.shape[1],m=len(b),
        source=str(source),source_sha256=sha(source),solver='highspy',preprocessing_variant='Exact KO affine/facial restriction of cached581-D target; no MVE',
        timings=dict(total_seconds=result['preparation_seconds'],interior_LP_seconds=result['LP_solve_seconds']),
        min_start_slack=result['min_raw_start_slack'],hostname=os.uname().nodename,precision_contract=result.get('validation_tolerance'))
    save(modeldir/'meta.json',meta);save(done,result);return result

def summarize():
    rows=[read(p) for p in sorted((R/'conditions').glob('*/condition.json'))]
    save(R/'geometry_summary.json',dict(conditions=rows,ready=sum(r['status']=='ready' for r in rows),
        infeasible=sum(r['status']=='infeasible' for r in rows),problematic=sum(r['status']=='problematic' for r in rows),
        sampling_ready_directories=[r['geometry_directory'] for r in rows if r['status']=='ready']))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['catalog','pilot','panel']);p.add_argument('--retry-problematic',action='store_true');p.add_argument('--genes',nargs='+');args=p.parse_args()
    manifest=read(R/'panel_manifest.json') if (R/'panel_manifest.json').exists() else catalog()
    if args.action=='catalog':return
    conditions=[r for r in manifest['conditions'] if args.action=='panel' or r['gene'] in manifest['pilot_genes']]
    if args.genes:
        candidates=read(R/'gene_catalog.json')['signatures'] if (R/'gene_catalog.json').exists() else []
        available={r['gene']:r for r in manifest['conditions']+candidates}
        conditions=[available[g] for g in args.genes if g in available]
        conditions.sort(key=lambda r:args.genes.index(r['gene']))
    if args.action=='pilot' and not args.genes:conditions.sort(key=lambda r:manifest['pilot_genes'].index(r['gene']))
    for c in conditions:
        existing=R/'conditions'/c['gene']
        if args.retry_problematic and (existing/'condition.json').exists() and read(existing/'condition.json').get('status') in ['problematic','preparing']:
            archive=R/'audit'/('prior-'+str(time.time_ns()))/c['gene'];archive.parent.mkdir(parents=True,exist_ok=True);existing.rename(archive)
        save(R/'progress.json',dict(gene=c['gene'],action=args.action,status='preparing',job_id=os.getenv('SLURM_JOB_ID')))
        try:r=prepare(c)
        except Exception as e:
            out=R/'conditions'/c['gene'];out.mkdir(parents=True,exist_ok=True)
            r=read(out/'condition.json') if (out/'condition.json').exists() else dict(c)
            r.update(status='problematic',reason=repr(e));save(out/'condition.json',r)
        summarize();print(c['gene'],r['status'],r.get('conditional_dimension'),r.get('LP_calls'),r.get('reason'),flush=True)
    save(R/'progress.json',dict(action=args.action,status='finished',job_id=os.getenv('SLURM_JOB_ID')))

if __name__=='__main__':main()
