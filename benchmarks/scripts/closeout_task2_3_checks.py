#!/usr/bin/env python3
"""Read-only final checks and reserved-resource accounting for closeout tasks2/3."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
B=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=B/'results/closeout/task2_3');a=p.parse_args();R=a.out.resolve();t2=json.loads((R/'task2_barrier_summary.json').read_text());t3=json.loads((R/'task3_off_summary.json').read_text());binary=t2['metadata']['binary_sha256'];checks=[]
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for row in t2['per_seed']:
 run=json.loads(Path(row['run_file']).read_text());c=run['profile']['config'];assert c['n_samples']==8 and run['barrier']=='rotate' and row['barrier_rotation_s']>0
 backend=t2['metadata'].get('task2_explicit_rotation_backend');runtime_gpu=None
 if backend is not None:
  values=[line.split('=',1)[1].strip() for line in (Path(run['profile_path']).parent/'config_used.txt').read_text().splitlines() if line.startswith('BARRIER_GPU_ROTATE=')]
  assert values and all(v==('true' if backend=='gpu' else 'false') for v in values);runtime_gpu=values[-1]=='true'
 checks.append(dict(task=2,model=row['model'],seed=row['seed'],binary_matches=run['exe_sha256']==binary,dimensions_match=(c['reduced_dim']==row['d'] and c['constraints']==row['m']),actual_seed_matches=c['seed']==row['seed'],starting_point_LP_s=row['starting_point_LP_s'],actual_runtime_GPU_rotation=runtime_gpu))
for row in t3['per_seed']:
 run=json.loads(Path(row['run_file']).read_text());c=run['profile']['config'];samples=Path(row['run_file']).parent/'samples.npy';header=np.load(samples,mmap_mode='r');assert header.shape==(16,10000,581) and header.dtype==np.float32
 assert c['n_samples']==11000 and c['thinning']==96 and c['n_chains']==16 and run['barrier']=='off' and run['host_prefault_threads']==0
 checks.append(dict(task=3,seed=row['seed'],binary_matches=run['exe_sha256']==binary,actual_seed_matches=c['seed']==row['seed'],actual_thinning=c['thinning'],rotation_phase_is_zero=run['profile']['timing']['barrier_rotation_s']==0,dtype=str(header.dtype),host_prefault=run['host_prefault_threads'],sample_shape=list(header.shape),samples_sha256=sha(samples),legacy_LP_mask_sha256=row['legacy_mask_sha256'],sample_bound_violation_fraction=row['sample_bound_violation_fraction']))
assert all(c['binary_matches'] and c['actual_seed_matches'] and c.get('dimensions_match',True) for c in checks)
prior=R.parent/'artifact_checks.json'
if R.name=='final-release' and prior.exists():
 old={r['seed']:r['samples_sha256'] for r in json.loads(prior.read_text())['checks'] if r['task']==3}
 parity=[dict(seed=r['seed'],final_samples_sha256=r['samples_sha256'],FCC_samples_sha256=old[r['seed']],bitwise_equal=r['samples_sha256']==old[r['seed']]) for r in checks if r['task']==3]
 (R/'task3_FCC_OFF_parity.json').write_text(json.dumps(dict(scope='Final c9 rotation-OFF controls versus retained FCC rotation-OFF controls; does not assert rotated-path parity.',seeds=parity),indent=2)+'\n');assert all(r['bitwise_equal'] for r in parity)
proof=R/'sourceproof/naja_chr.cu';(R/'sourceproof/naja_production_source.json').write_text(json.dumps(dict(git_commit=t2['metadata']['production_git_commit'],binary_sha256=binary,source_file=str(proof),source_sha256=sha(proof),coordinate_selection='ei = curand_uniform_double * cols; random coordinate scan',update_loop='samples_per_chain * thinning updates per chain, including warmup'),indent=2)+'\n')
(R/'artifact_checks.json').write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),status='passed',checker_source_sha256=sha(Path(__file__)),checks=checks),indent=2)+'\n')
job=json.loads((R/'slurm_job.json').read_text())['job'];account=subprocess.check_output(['sacct','-n','-X','-j',job,'-o','JobID,State,ExitCode,ElapsedRaw,AllocCPUS,TotalCPU,ReqMem,AllocTRES','-P'],text=True);(R/'slurm_accounting.psv').write_text(account);fields=account.strip().split('|');assert fields[1]=='COMPLETED' and fields[2]=='0:0';elapsed=int(fields[3]);(R/'resource_accounting.json').write_text(json.dumps(dict(job=job,state=fields[1],exit_code=fields[2],elapsed_seconds=elapsed,allocated_GPUs=1,reserved_GPU_hours=elapsed/3600,maximum_allowed_GPU_hours=1.0,within_user_budget=elapsed<3600),indent=2)+'\n');print('Checked',len(checks),'production runs; GPU-hours',elapsed/3600)
