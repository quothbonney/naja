"""Independent final GPR and affine/certificate checks of published KO targets."""
from pathlib import Path
import json
import os
import time
import numpy as np
import cobra
from exact_knockout_panel import affine_basis

B=Path(__file__).resolve().parents[1];R=B/'results/knockout_panel'
begun=time.perf_counter();manifest=json.loads((R/'ready_manifest.json').read_text())
cobra.Configuration().solver='glpk';model=cobra.io.load_json_model(str(B/'data/raw/iJO1366.json'))
ids=[r.id for r in model.reactions];index={r:i for i,r in enumerate(ids)}
original={r.id:r.bounds for r in model.reactions}
with np.load(B/'data/models/iJO1366/rounding/polytope.npz') as p:A0,b0,T0,shift0,start0=[p[k] for k in ['A','b','T','shift','start']]
b0=b0.reshape(-1);shift0=shift0.reshape(-1);rows=[]
for condition in manifest['conditions']:
    gene=condition['gene'];result=dict(gene=gene,status='passed')
    try:
        with model:
            model.genes.get_by_id(gene).knock_out()
            actual={r.id for r in model.reactions if r.bounds!=original[r.id]}
            assert actual==set(condition['knockout_reaction_ids']),('GPR mismatch',actual)
        raw=json.loads(Path(condition['source']).read_text());assert [r['id'] for r in raw['reactions']]==ids
        for reaction in raw['reactions']:
            expected=(0.,0.) if reaction['id'] in actual else original[reaction['id']]
            assert (reaction['lower_bound'],reaction['upper_bound'])==expected
        directory=Path(condition['geometry_directory'])
        with np.load(directory/'rounding/polytope.npz') as p:A,b,T,shift,start=[p[k] for k in ['A','b','T','shift','start']]
        with np.load(condition['embedding']) as p:z0,N,kept=p['z0'],p['N'],p['kept_base_rows']
        assert np.min(b-A@start)>0
        assert np.max(np.abs(A-A0[kept]@N))<=1e-10
        assert np.max(np.abs(b-(b0[kept]-A0[kept]@z0)))<=1e-10
        assert np.max(np.abs(T-T0@N))<=1e-9 and np.max(np.abs(shift-(T0@z0+shift0)))<=1e-9
        ko=[index[s] for s in condition['knockout_reaction_ids']];E=T0[ko].copy();f=-shift0[ko].copy()
        maximum_A=maximum_b=0.
        for certificate in condition['facial_reductions']:
            p,Nstep,rank=affine_basis(E,f,start0)
            assert len(set(rank.get('ranks_at_1e11_1e12_1e13',[0])))==1
            retained=np.asarray(certificate['all_base_rows'],dtype=int);Ar=A0[retained]@Nstep;br=b0[retained]-A0[retained]@p
            norms=np.linalg.norm(Ar,axis=1);w=np.asarray(certificate['full_weights'])
            assert w.min()>=0 and abs(w.sum()-1)<=1e-8
            a=float(np.max(np.abs((Ar/norms[:,None]).T@w)));bb=float(abs((br/norms)@w))
            maximum_A=max(maximum_A,a);maximum_b=max(maximum_b,bb)
            assert a<=1e-8 and bb<=1e-8,('Exposing certificate failed',a,bb)
            exposed=np.asarray(certificate['support_base_rows'],dtype=int);E=np.vstack([E,A0[exposed]]);f=np.r_[f,b0[exposed]]
        assert condition['validation']['max_original_bound_violation']<=1e-7
        assert condition['validation']['max_KO_flux_residual']<=1e-7
        assert json.loads((directory/'external_stoichiometry_check.json').read_text())['passed']
        result.update(max_recomputed_dual_A_error=maximum_A,max_recomputed_dual_b_error=maximum_b,dimension=A.shape[1],GPR_verified_by_actual_cobra_knock_out=True)
    except Exception as error:result.update(status='failed',error=repr(error))
    rows.append(result)
report=dict(status='passed' if all(r['status']=='passed' for r in rows) else 'failed',targets=len(rows),rows=rows,wall_seconds=time.perf_counter()-begun,slurm_job_id=os.getenv('SLURM_JOB_ID'))
(R/'final_geometry_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
raise SystemExit(0 if report['status']=='passed' else 1)
