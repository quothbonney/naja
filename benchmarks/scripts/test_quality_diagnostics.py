#!/usr/bin/env python3
"""Verify exact-cache acceptance/rejection and streaming Rhat mathematics."""
from pathlib import Path
import json
import shutil
import sys
import tempfile
import numpy as np

B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B));sys.path.insert(0,str(B/'scripts'))
import metrics
from quality_diagnostics import sha,save_identity,reuse,rhat_precheck

with tempfile.TemporaryDirectory(dir=B/'tmp') as directory:
    root=Path(directory);model=root/'model';(model/'rounding').mkdir(parents=True);(model/'gem').mkdir()
    np.savez(model/'rounding/polytope.npz',A=np.vstack([np.eye(2),-np.eye(2)]),b=np.ones(4)*100,
             start=np.zeros(2),T=np.eye(2),shift=np.zeros(2))
    np.savetxt(model/'gem/l_bounds.csv',[-100,-100]);np.savetxt(model/'gem/u_bounds.csv',[100,100])
    (model/'gem/reaction_ids.txt').write_text('x\ny\n')
    mask=np.ones(2,dtype=bool);np.save(model/'active_reactions.npy',mask)
    (model/'diagnostic_mask.json').write_text(json.dumps({'geometry_sha256':sha(model/'rounding/polytope.npz'),
        'source_sha256':'synthetic','range_tolerance':1e-7,'policy':'synthetic identity model, both coordinates variable'}))
    pilot,final=root/'pilot',root/'final';pilot.mkdir();final.mkdir()
    x=np.random.default_rng(42).normal(size=(4,128,2)).astype(np.float32)
    np.save(pilot/'samples.npy',x);shutil.copy2(pilot/'samples.npy',final/'samples.npy')
    check=rhat_precheck(pilot/'samples.npy',model)
    expected=np.median(metrics.split_rhat(x.astype(np.float64)))
    np.testing.assert_allclose(check['split_rhat_median'],expected,rtol=1e-12,atol=1e-12)
    assert check['checked_samples']==4*128 and check['sample_bound_violation_fraction']==0
    metrics.evaluate(pilot/'samples.npy',model,pilot,prefixes=[128])
    save_identity(pilot/'samples.npy',model,pilot)
    assert reuse(final/'samples.npy',model,final,2.0,pilot),'Identical data should reuse'
    m=json.loads((final/'metrics.json').read_text())
    assert m['samples']==str((final/'samples.npy').resolve()) and m['wall_seconds']==2
    assert m['samples_per_second']==256
    altered=x.copy();altered[0,0,0]+=1
    np.save(final/'samples.npy',altered)
    assert not reuse(final/'samples.npy',model,final,2.0,pilot),'One altered sample must reject'
    np.save(final/'samples.npy',x.astype(np.float64))
    assert not reuse(final/'samples.npy',model,final,2.0,pilot),'Changed dtype must reject'
    np.save(final/'samples.npy',x.reshape(4,64,4))
    assert not reuse(final/'samples.npy',model,final,2.0,pilot),'Changed shape must reject'
    shutil.copy2(pilot/'samples.npy',final/'samples.npy')
    np.save(model/'active_reactions.npy',np.array([True,False]))
    assert not reuse(final/'samples.npy',model,final,2.0,pilot),'Changed mask must reject'
    np.save(model/'active_reactions.npy',mask)
    geometry=model/'rounding/polytope.npz';geometry.write_bytes(geometry.read_bytes()+b'changed')
    assert not reuse(final/'samples.npy',model,final,2.0,pilot),'Changed geometry must reject'
print('PASS: streaming Rhat matches direct formula; exact bytes accepted; sample/dtype/shape/mask/geometry changes rejected.')
