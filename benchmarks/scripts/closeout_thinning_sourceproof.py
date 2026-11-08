#!/usr/bin/env python3
"""Verify thinning units from official hopsy/HOPS sources and native binary behavior."""
import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import hopsy
import hopsy.core
B=Path(__file__).resolve().parents[1];OUT=B/'results/closeout/task2_3/sourceproof';OUT.mkdir(parents=True,exist_ok=True)
source=B/'source/hopsy-1.7.0';paths={'python_sample':'src/hopsy/misc.py','uniform_proposal_alias':'src/hopsy/proposal.hpp','chain_adapter':'extern/hops/src/hops/MarkovChain/MarkovChainAdapter.hpp','draw_adapter':'extern/hops/src/hops/MarkovChain/Draw/NoOpDrawAdapter.hpp','coordinate_proposal':'extern/hops/src/hops/MarkovChain/Proposal/CoordinateHitAndRunProposal.hpp'}
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
evidence=[]
for name,relative in paths.items():
 p=source/relative;shutil.copy2(p,OUT/p.name);evidence.append(dict(role=name,path=str(p),sha256=sha(p),snapshot=str(OUT/p.name)))
assert 'for (long i = 0; i < thinning; ++i)' in (source/paths['chain_adapter']).read_text()
assert '++coordinateToUpdate %= state.rows();' in (source/paths['coordinate_proposal']).read_text()
assert 'markov_chain.draw(rng, thinning)' in (source/paths['python_sample']).read_text()
A=np.vstack((np.eye(3),-np.eye(3)));problem=hopsy.Problem(A,np.ones(6));chain=hopsy.MarkovChain(problem,proposal=hopsy.UniformCoordinateHitAndRunProposal,starting_point=np.zeros(3));rng=hopsy.RandomNumberGenerator(101);updates=[]
for i in range(12):
 old=chain.state.copy();acceptance,new=chain.draw(rng,thinning=1);changed=np.flatnonzero(new!=old).tolist();assert changed==[(i+1)%3];updates.append(dict(call=i,thinning=1,changed_coordinates=changed,acceptance=acceptance))
report=dict(generated_utc=datetime.now(timezone.utc).isoformat(),conclusion='One hopsy thinning unit is one coordinate proposal/update, not a full d-coordinate sweep. UniformCoordinateHitAndRun uses deterministic cyclic coordinate scan; Naja uses random coordinate scan. Comparing actual coordinate-update counts is valid, but the scan kernels differ.',hopsy_version=hopsy.__version__,hopsy_build=hopsy.__build__,native_module=str(hopsy.core.__file__),native_module_sha256=sha(hopsy.core.__file__),source_archive_manifest=json.loads((B/'audit/hopsy_native_build.json').read_text())['source'],source_files=evidence,native_three_dimensional_box_smoke_check=updates,count_formula='sum_over_chains((retained_draws+discarded_warmup_draws)*thinning)',naja_task3_actual_updates=16*11000*96,naja_task3_retained_only_updates=16*10000*96)
assert report['native_module_sha256']==json.loads((B/'audit/hopsy_native_build.json').read_text())['module_sha256']
report['native_module_matches_benchmark_build_manifest']=True
report['proof_script_sha256']=sha(Path(__file__))
(OUT/'thinning_semantics.json').write_text(json.dumps(report,indent=2)+'\n')
(OUT/'THINNING_SEMANTICS.md').write_text('''# Coordinate-update accounting\n\nOne hopsy thinning unit is **one coordinate update**, not one full coordinate sweep. Python `sample` calls `markov_chain.draw(rng, thinning)` once per retained draw; HOPS `MarkovChainAdapter` executes `thinning` calls to the draw implementation; the uniform draw adapter proposes and accepts once; the coordinate proposal advances exactly one coordinate.\n\nHopsy cycles deterministically through coordinates (`++coordinateToUpdate %= state.rows()`); Naja randomly selects a coordinate. They target the same uniform measure but have different scan kernels. The native hopsy module was also checked on a3-Dbox: successive thinning-1 calls changed coordinates1,2,0,1,2,0.\n\nTask3 performs16 × (10,000+1,000) ×96 = **16,896,000 actual coordinate updates**, including warmup. Retained-only updates are15,360,000. Median pooled reaction ESS /16.896 is ESS per million actual updates.\n\nFull source snapshots, official-sdist SHA, native-module SHA and the smoke-test observations are in [thinning_semantics.json](thinning_semantics.json). No estimator or sampling code was modified.\n''')
print(OUT/'thinning_semantics.json')
