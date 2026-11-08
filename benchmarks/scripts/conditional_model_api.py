"""Stable model-directory API for conditional global rerounding and common masks."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
B=Path(__file__).resolve().parents[1]

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write_json(path,value):
    path=Path(path);temp=path.with_name(path.name+f'.tmp.{os.getpid()}');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def install_cached_mask(model, active, report):
    import numpy as np
    model=Path(model);assert report['geometry_sha256']==sha(model/'rounding/polytope.npz')
    original=model/'diagnostic_mask.json'
    if original.exists() and not (model/'diagnostic_mask.before_cached_target.json').exists():
        shutil.copy2(original,model/'diagnostic_mask.before_cached_target.json')
        shutil.copy2(model/'active_reactions.npy',model/'active_reactions.before_cached_target.npy')
    temp=model/f'.active.{os.getpid()}.npy';np.save(temp,active);temp.replace(model/'active_reactions.npy')
    report=dict(report,target_geometry_kind='exact_cached_polytope')
    write_json(original,report)

def ensure_cached_mask(model):
    import numpy as np
    from cached_target_mask import build
    model=Path(model);p=model/'diagnostic_mask.json'
    if p.exists():
        report=json.loads(p.read_text())
        if report.get('target_geometry_kind')=='exact_cached_polytope' and report['geometry_sha256']==sha(model/'rounding/polytope.npz'):
            return np.load(model/'active_reactions.npy'),report
    active,report=build(model);install_cached_mask(model,active,report);return active,report

def transport_mask(shared,glob):
    import numpy as np
    shared,glob=Path(shared),Path(glob)
    proof=json.loads((glob/'global_reround_proof.json').read_text())
    assert proof['status']=='passed' and proof['source_geometry_sha256']==sha(shared/'rounding/polytope.npz')
    assert proof['destination_geometry_sha256']==sha(glob/'rounding/polytope.npz')
    active,report=ensure_cached_mask(shared)
    report=dict(report,model=glob.name,geometry_sha256=sha(glob/'rounding/polytope.npz'),
                certificate_geometry_sha256=report['geometry_sha256'],
                exact_target_transport='Same reaction ranges under validated invertible square affine change; no second reduction or target relaxation.',
                map_proof=str((glob/'global_reround_proof.json').resolve()))
    install_cached_mask(glob,active,report)
    return active,report

def prepare_global(shared,output):
    """Run full global PolyRound on this exact cached conditional input."""
    shared,output=Path(shared),Path(output)
    ensure_cached_mask(shared)
    subprocess.run([str(B/'env-cpu/bin/python'),str(B/'scripts/conditioned_global_reround.py'),
                    '--source',str(shared),'--output',str(output)],check=True)
    transport_mask(shared,output)
    return json.loads((output/'global_reround_proof.json').read_text())

def compile_conditional_mask(condition_root,parent_model=None,parent_certificate=None):
    """Inherit only exact-CACHED-parent certificates under checked containment.

    Newly fixed explicit bounds are exact target constraints. All other new or
    unresolved candidates remain active; no claim of fresh per-reaction LPs.
    """
    import copy,time
    import numpy as np
    started=time.perf_counter();root=Path(condition_root);model=root/'models/shared'
    parent_model=Path(parent_model or B/'data/models/iJO1366')
    parent_certificate=Path(parent_certificate or B/'results/knockout_panel/base_cached_mask')
    info=json.loads((root/'condition.json').read_text());assert info['status']=='ready' and info.get('no_bound_epsilon_relaxation')
    parent=json.loads((parent_certificate/'diagnostic_mask.json').read_text())
    assert parent.get('target_geometry_kind')=='exact_cached_polytope'
    assert parent['geometry_sha256']==sha(parent_model/'rounding/polytope.npz')==info['base_geometry_sha256']
    target=dict(np.load(model/'rounding/polytope.npz'));base=dict(np.load(parent_model/'rounding/polytope.npz'));embed=dict(np.load(root/'embedding.npz'))
    N,z0,keep=embed['N'],embed['z0'],embed['kept_base_rows'].astype(int)
    Aexpected=base['A'][keep]@N;bexpected=base['b'].reshape(-1)[keep]-base['A'][keep]@z0
    assert np.max(np.abs(target['A']-Aexpected)/(1+np.abs(Aexpected)))<=1e-10
    assert np.max(np.abs(target['b'].reshape(-1)-bexpected)/(1+np.abs(bexpected)))<=1e-10
    assert np.max(np.abs(target['T']-base['T']@N)/(1+np.abs(target['T'])))<=1e-10
    assert np.max(np.abs(target['shift'].reshape(-1)-(base['T']@z0+base['shift'].reshape(-1))))<=1e-7
    dropped=np.setdiff1d(np.arange(len(base['b'])),keep)
    if len(dropped):
        mapped=base['A'][dropped]@N;norm=np.linalg.norm(base['A'][dropped],axis=1)
        assert np.max(np.linalg.norm(mapped,axis=1)/np.maximum(norm,1e-300))<=1e-9
        assert np.min(base['b'].reshape(-1)[dropped]-base['A'][dropped]@z0)>=-1e-7
    original=json.loads(Path(json.loads((parent_model/'meta.json').read_text())['source']).read_text())
    meta=json.loads((model/'meta.json').read_text());raw=json.loads(Path(meta['source']).read_text())
    ids=[r['id'] for r in original['reactions']];assert ids==[r['id'] for r in raw['reactions']]
    assert original['metabolites']==raw['metabolites']
    for old,new in zip(original['reactions'],raw['reactions']):
        assert old['metabolites']==new['metabolites']
        assert old.get('objective_coefficient',0)==new.get('objective_coefficient',0)
        assert old['lower_bound']<=new['lower_bound']<=new['upper_bound']<=old['upper_bound']
    assert (model/'gem/reaction_ids.txt').read_text().splitlines()==ids
    active=np.load(parent_certificate/'active_reactions.npy').copy()
    certified={r['index']:dict(r,inherited_from_exact_cached_parent=True) for r in parent['reactions'] if r['nearconstant_excluded']}
    forced=set(info.get('knockout_reaction_ids',[]))
    for i,r in enumerate(raw['reactions']):
        original_norm=float(np.linalg.norm(base['T'][i]))
        imposed=(r['id'] in forced and r['lower_bound']==r['upper_bound'] and
                 np.linalg.norm(target['T'][i])<=1e-10*max(original_norm,1e-300) and
                 abs(float(target['shift'].reshape(-1)[i])-r['lower_bound'])<=1e-7)
        if imposed:
            active[i]=False;certified[i]=dict(index=i,reaction=ids[i],nearconstant_excluded=True,
                minimum=r['lower_bound'],maximum=r['upper_bound'],range=0.,certificate='Explicit fixed reaction bound in exact conditional target; no new LP')
    tiny=np.flatnonzero(np.max(np.abs(target['T']),axis=1,initial=0)<=1e-6)
    records=[]
    for i in sorted(set(tiny.tolist())|set(certified)):
        records.append(certified.get(i,dict(index=i,reaction=ids[i],T_row_max=float(np.abs(target['T'][i]).max(initial=0)),
            nearconstant_excluded=False,certificate='Unresolved/new candidate remains active; no unverified exclusion')))
    report=dict(schema_version=1,model=model.name,source=meta['source'],source_sha256=sha(Path(meta['source'])),
        geometry_sha256=sha(model/'rounding/polytope.npz'),target_geometry_kind='exact_cached_polytope',
        solver='Verified parent cached-target GLPK ranges transported by exact target inclusion; explicit fixed bounds need no LP',
        solver_feasibility_tolerance=1e-9,range_tolerance=1e-7,candidate_tolerance=1e-6,
        policy='Only certified cached-parent nearconstant reactions and explicit conditional fixed bounds excluded. Other/new unresolved candidates stay active. No raw-original-model LP certificate used as cached-target oracle.',
        n_reactions=len(ids),candidates=len(records),active_reactions=int(active.sum()),excluded_nearconstant_reactions=int((~active).sum()),
        parent_certificate=str(parent_certificate),parent_geometry_sha256=parent['geometry_sha256'],parent_mask_sha256=sha(parent_certificate/'active_reactions.npy'),
        target_containment_proof='Checked kept-row A/b projection, removed constant rows, T/shift composition, unchanged reaction order/S/objective and nested bounds; no epsilon.',
        reactions=records,wall_seconds=time.perf_counter()-started,new_conditional_LP_calls=0)
    install_cached_mask(model,active,report);write_json(root/'conditional_mask_inheritance.json',report)
    return active,report
