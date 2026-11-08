"""Detached bounded exact-KO geometry controller; samples are owned by the quality harness."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl
import csv
import json
import os
import subprocess
import sys
import time
import numpy as np
from scipy import sparse

B=Path(__file__).resolve().parents[1];R=B/'results/knockout_panel'
STOP=datetime(2026,11,8,10,30,tzinfo=timezone.utc)
END=datetime(2026,11,8,11,0,tzinfo=timezone.utc)
TERMINAL={'ready','infeasible','problematic','precision_ambiguous','zero_dimensional'}
GALLERY={}
STOICH=None

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):return default
def save(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
def fingerprint(row):
    path=row['embedding']
    if path not in GALLERY:
        with np.load(path) as p:N,z=p['N'],p['z0']
        P=np.eye(N.shape[0])-N@N.T;GALLERY[path]=(P,P@z)
    return GALLERY[path]
def stoichiometry(row):
    global STOICH
    out=Path(row['geometry_directory']);path=out/'external_stoichiometry_check.json'
    prior=read(path)
    if prior:return prior
    started=time.perf_counter()
    if STOICH is None:
        base=B/'data/models/iJO1366/gem'
        STOICH=(sparse.csr_matrix(np.loadtxt(base/'A_eq.csv',delimiter=',')),np.loadtxt(base/'b_eq.csv',delimiter=',').reshape(-1))
    S,rhs=STOICH
    with np.load(out/'rounding/polytope.npz') as p:start,T,shift=p['start'],p['T'],p['shift'].reshape(-1)
    rng=np.random.default_rng(101);directions=rng.normal(size=(32,len(start)));directions/=np.linalg.norm(directions,axis=1)[:,None]
    points=np.vstack([start,start+0.1*row['strict_interior_radius']*directions]);flux=points@T.T+shift
    error=float(np.max(np.abs(S@flux.T-rhs[:,None])))
    result=dict(points=len(points),max_original_Sv_minus_b=error,tolerance=1e-7,passed=error<=1e-7,
        external_validation_seconds=time.perf_counter()-started,scope='External diagnostics excluded from geometry preparation timer; common S loading is one-time in controller.')
    save(path,result);return result
def ready_manifest():
    rows=[read(p) for p in sorted((R/'conditions').glob('*/condition.json'))];rows=[r for r in rows if r]
    valid=[];duplicates=[]
    for row in rows:
        if row.get('status')!='ready':continue
        if any(len(set(h.get('ranks_at_1e11_1e12_1e13',[0])))>1 for h in row.get('affine_rank_history',[])):
            continue
        sv=stoichiometry(row)
        if not sv['passed']:continue
        row['external_stoichiometry_check']=sv
        P,z=fingerprint(row);duplicate=None
        for previous in valid:
            if previous['conditional_dimension']!=row['conditional_dimension']:continue
            Q,w=fingerprint(previous)
            pe=float(np.max(np.abs(P-Q)));ze=float(np.max(np.abs(z-w)))
            if pe<=1e-10 and ze<=1e-9:
                duplicate=dict(gene=row['gene'],duplicate_of=previous['gene'],projector_error=pe,affine_offset_error=ze);break
        if duplicate:duplicates.append(duplicate)
        else:valid.append(row)
    result=dict(status='complete' if len(valid)>=50 else 'in_progress',unique_ready_count=len(valid),
        nontrivial_unique_ready_count=sum(r['changed_bound_rows']>0 for r in valid),
        conditions=valid,duplicate_targets=duplicates,all_attempted_conditions=rows,
        numerical_equivalence_contract='Same affine-face projector within1e-10 and offset1e-9, same cached base target; rank stable at1e-11/12/13. No epsilon-relaxed KO bounds.',
        sampling_ready_directories=[r['geometry_directory'] for r in valid],updated_utc=datetime.now(timezone.utc).isoformat())
    save(R/'ready_manifest.json',result)
    table=[]
    for row in valid:
        table.append(dict(gene=row['gene'],stratum=row['stratum'],changed_bound_rows=row['changed_bound_rows'],dimension=row['conditional_dimension'],LP_calls=row['LP_calls'],LP_skipped=row['LP_calls']==0,
            base_weak_feasible=row['base_point_weak_feasible'],initial_projection_strict=row.get('initial_projection_strict_interior',row.get('projected_point_strict_interior')),
            preparation_seconds=row['preparation_seconds'],LP_solve_seconds=row['LP_solve_seconds'],max_original_bound_violation=row['validation']['max_original_bound_violation'],
            max_KO_flux_residual=row['validation']['max_KO_flux_residual'],max_S_residual=row['external_stoichiometry_check']['max_original_Sv_minus_b'],model_directory=row['geometry_directory']))
    if table:
        with (R/'geometry_table.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
        nontrivial=[r for r in table if r['changed_bound_rows']>0]
        save(R/'geometry_metrics_snapshot.json',dict(unique_ready=len(table),nontrivial_ready=len(nontrivial),
            LP_skip_fraction_in_valid_nontrivial_subset=sum(r['LP_skipped'] for r in nontrivial)/len(nontrivial) if nontrivial else None,
            median_preparation_seconds=float(np.median([r['preparation_seconds'] for r in nontrivial])) if nontrivial else None,
            scope='Validity-selected completed targets; not a population LP-skip estimate or a sampling convergence table.'))
    return result

def main():
    R.mkdir(parents=True,exist_ok=True);lock=(R/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (R/'controller.pid').write_text(str(os.getpid())+'\n')
    catalog=read(R/'gene_catalog.json');manifest=read(R/'panel_manifest.json')
    initial=manifest['conditions'][:50]
    clean=[r for r in catalog['signatures'] if r['all_affected_reactions_varying'] and r['minimum_KO_T_row_norm']>1e-9]
    reversible=[r for r in clean if all(lo<0<hi for lo,hi in r['original_changed_bounds'].values())]
    ordered=[];seen=set()
    for r in initial+clean:
        if r['gene'] not in seen:ordered.append(r);seen.add(r['gene'])
    state=read(R/'controller_state.json',dict(jobs=[],candidate_count=len(ordered),max_concurrent=2,max_attempts=400,
        stop_new_jobs_utc=STOP.isoformat(),end_utc=END.isoformat(),selection='Initial stratified signatures, then deterministic replacements based only on geometry validity/equivalence; never mixing or speed.'))
    known={j['job_id'] for j in state['jobs']}
    for line in subprocess.check_output(['/usr/bin/squeue','-h','-u',str(os.getuid()),'-o','%A|%j'],text=True).splitlines():
        job_id,name=line.split('|',1)
        if name.startswith('bench-exact-KO-b') and job_id not in known:
            state['jobs'].append(dict(gene=name.removeprefix('bench-exact-KO-'),job_id=job_id,recovered_on_restart=True))
    while datetime.now(timezone.utc)<END:
        ready=ready_manifest()
        state['unique_ready_count']=ready['unique_ready_count']
        live=set(subprocess.check_output(['/usr/bin/squeue','-h','-u',str(os.getuid()),'-o','%A'],text=True).split())
        active=[j for j in state['jobs'] if j['job_id'] in live]
        for job in state['jobs']:
            if job['job_id'] in live or job.get('finalized'):continue
            path=R/'conditions'/job['gene']/'condition.json';row=read(path,{})
            if row.get('status') not in TERMINAL:
                row.update(gene=job['gene'],status='problematic',reason='Geometry job ended or reached its cap before producing a validated target; no sampling eligibility.',job_id=job['job_id'])
                path.parent.mkdir(parents=True,exist_ok=True);save(path,row)
            job['finalized']=True
        if len(ready['conditions'])>=50 and not active:state['status']='achieved_50_unique_ready';save(R/'controller_state.json',state);break
        if len(state['jobs'])>=state['max_attempts'] and not active:state['status']='attempt_cap_reached';save(R/'controller_state.json',state);break
        if datetime.now(timezone.utc)>=STOP and not active:state['status']='launch_deadline_reached';save(R/'controller_state.json',state);break
        busy={j['gene'] for j in active};launched={j['gene'] for j in state['jobs']}
        # Establish five clean targets before expanding the harder stratified set.
        candidates=(reversible+ordered) if ready['unique_ready_count']<5 else ordered
        for row in candidates:
            if ready['unique_ready_count']>=50 or len(active)>=state['max_concurrent'] or len(state['jobs'])>=state['max_attempts'] or datetime.now(timezone.utc)>=STOP:break
            gene=row['gene'];prior=read(R/'conditions'/gene/'condition.json',{})
            if gene in busy or gene in launched or prior.get('status') in TERMINAL:continue
            command=['/usr/bin/sbatch','--parsable',f'--chdir={B}',f'--job-name=bench-exact-KO-{gene}',
                f'--output={B}/logs/exact-KO-{gene}-%j.log','--time=00:10:00',str(B/'scripts/exact_knockout_panel.sbatch'),'panel','--genes',gene,'--retry-problematic']
            job_id=subprocess.check_output(command,text=True).strip().split(';')[0]
            job=dict(gene=gene,job_id=job_id,command=command,submitted_utc=datetime.now(timezone.utc).isoformat())
            state['jobs'].append(job);active.append(job);launched.add(gene);busy.add(gene)
            print('Submitted',gene,job_id,flush=True)
        state.update(status='running',unique_ready_count=ready['unique_ready_count'],updated_utc=datetime.now(timezone.utc).isoformat())
        save(R/'controller_state.json',state);time.sleep(20)
    else:
        state['status']='hard_deadline_reached';save(R/'controller_state.json',state)
    ready_manifest()

if __name__=='__main__':main()
