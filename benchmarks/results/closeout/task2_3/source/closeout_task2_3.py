#!/usr/bin/env python3
"""Closeout only: production barrier timings and legacy-matched rotation-OFF control."""
import argparse,csv,hashlib,json,os,platform,statistics,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[1]
MODELS=['e_coli_core','iAB_RBC_283','iJR904','iMM904','iAF1260','iJO1366','iML1515','RECON1'];SEEDS=[101,202,303]
p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--stage',choices=['task2','task3','both'],default='both');p.add_argument('--out',type=Path,default=B/'results/closeout/task2_3');a=p.parse_args();a.binary=a.binary.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def csvsave(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def call(command,log):
 start=time.perf_counter();log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('w') as f:subprocess.run(list(map(str,command)),check=True,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
 return time.perf_counter()-start
binary_hash=sha(a.binary);helper_hash=sha(Path(__file__));metadata=dict(created_utc=datetime.now(timezone.utc).isoformat(),binary=str(a.binary),binary_sha256=binary_hash,production_git_commit=a.commit,source_script_sha256=helper_hash,slurm_job_id=os.environ.get('SLURM_JOB_ID'),hostname=platform.node(),CPU_allocation=os.environ.get('SLURM_CPUS_PER_TASK'),models=MODELS,seeds=SEEDS,task2_protocol='16 chains, 8 retained draws, warmup 0, thinning 1, TPB 128, barrier rotate, float64 export, host prefault 0; phase timing only, not convergence',task3_protocol='iJO1366, 16 chains, 1,000 warmup + 10,000 retained draws, thinning 96, TPB 128, barrier off, float32 export, host prefault 0; same cached polytope/start/legacy LP mask as historical table',scope='GPU sampler phase primary; profiletotal and adapter/output walls secondary. Original canonical outputs and frozen estimators untouched.')
save(a.out/'manifest.json',metadata)

def sample(model,seed,task):
 directory=B/'data/models'/model;dest=a.out/task/model/f'seed{seed}';runpath=dest/'run.json'
 barrier='rotate' if task=='task2_barrier' else 'off';draws=8 if task=='task2_barrier' else 10000;warm=0 if task=='task2_barrier' else 1000;thin=1 if task=='task2_barrier' else 96;dtype='float64' if task=='task2_barrier' else 'float32'
 command=[B/'scripts/enter.sh',B/'env-gpu/bin/python',B/'naja_runner.py','--exe',a.binary,'--model',directory,'--out',dest,'--seed',seed,'--chains',16,'--draws',draws,'--warmup',warm,'--thinning',thin,'--tpb',128,'--barrier',barrier,'--sample-dtype',dtype,'--host-prefault-threads',0,'--timeout',1200]
 if not runpath.exists():
  seconds=call(command,dest.parent/f'{dest.name}.adapter.log');save(dest/'closeout_command.json',dict(argv=list(map(str,command)),external_adapter_wall_seconds=seconds,production_git_commit=a.commit,binary_sha256=binary_hash))
 run=read(runpath);assert run['status']=='complete' and run['exe_sha256']==binary_hash and run['seed']==seed and run['barrier']==barrier and run['sample_dtype']==dtype
 assert run['chains']==16 and run['draws']==draws and run['warmup']==warm and run['requested_thinning']==thin and run['host_prefault_threads']==0
 if barrier=='off':assert run['profile']['timing']['barrier_rotation_s']==0 and not any(z.startswith('--barrier-') for z in run['command'])
 return directory,dest,run

if a.stage in ('task2','both'):
 rows=[]
 for model in MODELS:
  directory=B/'data/models'/model;model_meta=read(directory/'meta.json');geometry=dict(np.load(directory/'rounding/polytope.npz'));n=geometry['T'].shape[0];d=geometry['A'].shape[1];m=geometry['A'].shape[0]
  for seed in SEEDS:
   directory,dest,run=sample(model,seed,'task2_barrier');phase=run['profile']['timing']
   rows.append(dict(model=model,seed=seed,n=n,d=d,m=m,barrier_rotation_s=phase['barrier_rotation_s'],native_profile_total_s=phase['total_s'],adapter_wall_seconds=run['wall_including_normalization_seconds'],gpu_sampling_s=phase['gpu_sampling_s'],load_polytope_s=phase['load_polytope_s'],starting_point_LP_s=phase.get('cube_center_lp_s'),original_CPU_round_component_s=model_meta['timings']['round_seconds'],original_CPU_polyround_total_s=model_meta['timings']['polyround_seconds'],original_CPU_prep_host=model_meta['hostname'],geometry_sha256=sha(directory/'rounding/polytope.npz'),start_sha256=hashlib.sha256(geometry['start'].tobytes()).hexdigest(),binary_sha256=binary_hash,git_commit=a.commit,profile_file=run['profile_path'],run_file=str(dest/'run.json')))
  csvsave(a.out/'task2_per_seed.csv',rows)
 summary=[]
 for model in MODELS:
  rr=[r for r in rows if r['model']==model];summary.append(dict(model=model,n=rr[0]['n'],d=rr[0]['d'],m=rr[0]['m'],seeds=SEEDS,median_barrier_rotation_s=statistics.median(r['barrier_rotation_s'] for r in rr),minimum_barrier_rotation_s=min(r['barrier_rotation_s'] for r in rr),maximum_barrier_rotation_s=max(r['barrier_rotation_s'] for r in rr),median_native_total_s=statistics.median(r['native_profile_total_s'] for r in rr),original_CPU_round_component_s=rr[0]['original_CPU_round_component_s'],original_CPU_prep_host=rr[0]['original_CPU_prep_host']))
 save(a.out/'task2_barrier_summary.json',dict(metadata=metadata,rows=summary,per_seed=rows,note='Production barrier_rotation_s includes Hessian/eigensolve, rotation, transfers and host work inside the phase; it is not pure GPU kernel duration. Runs start from already PolyRound-rounded cached models. Historical CPU round time has a different preprocessing input and is not a replacement-speedup baseline; e_coli CPU preparation ran on the login host, the others on the H100 node CPU. CPU references are original single measurements; GPU medians use three distinct seeds.'))
 csvsave(a.out/'task2_barrier_summary.csv',summary)
 lines=['# Task2: production barrier phase across eight models','',f'Production commit`{a.commit}`, binarySHA256`{binary_hash}`.','',metadata['task2_protocol'],'','| Model | d | m | Median barrier phase s | Min–max s | Historical CPU MVE-round s |','|---|---:|---:|---:|---:|---:|']
 for r in summary:lines.append(f"| {r['model']} | {r['d']} | {r['m']} | {r['median_barrier_rotation_s']:.6f} | {r['minimum_barrier_rotation_s']:.6f}–{r['maximum_barrier_rotation_s']:.6f} | {r['original_CPU_round_component_s']:.3f} |")
 lines+=['','The measured phase includes GPU math, transfers and host work inside barrier_rotation_s. This is additional correction on already-rounded cached geometry. The historical CPU column is original PolyRound rounding on its original input; this barrier phase does not replace full base preparation. CPU references are single observations; e_coli_core used the login CPU and the other seven used the H100 node CPU.','', '[Per-seed phases and commands](task2_per_seed.csv) · [Full summary/provenance](task2_barrier_summary.json).']
 (a.out/'TASK2_BARRIER.md').write_text('\n'.join(lines)+'\n')

if a.stage in ('task3','both'):
 rows=[];directory=B/'data/models/iJO1366';mask=read(directory/'diagnostic_mask.json');mask_hash=sha(directory/'active_reactions.npy');geometry=dict(np.load(directory/'rounding/polytope.npz'))
 assert mask.get('target_geometry_kind')!='exact_cached_polytope';assert mask['excluded_nearconstant_reactions']==878
 for seed in SEEDS:
  directory,dest,run=sample('iJO1366',seed,'task3_rotation_off');metricpath=dest/'metrics.json'
  command=[B/'env-cpu/bin/python',B/'metrics.py','--samples',dest/'samples.npy','--model-dir',directory,'--output',dest,'--prefixes',10000,'--workers',4]
  if not metricpath.exists():
   elapsed=call(command,dest.parent/f'{dest.name}.metrics.log');save(dest/'closeout_metrics_command.json',dict(argv=list(map(str,command)),diagnostics_wall_seconds=elapsed,metrics_sha256=sha(B/'metrics.py')))
  metric=read(metricpath);assert metric['metrics_code_sha256']==sha(B/'metrics.py') and metric['diagnostic_mask_status']=='model_lp_verified'
  phase=run['profile']['timing'];actual=16*11000*96;retained=16*10000*96
  rows.append(dict(seed=seed,split_rhat_median=metric['split_rhat_median'],split_rhat_p95=metric['split_rhat_p95'],bulk_ess_median=metric['bulk_ess_median'],coordinate_updates_including_warmup=actual,coordinate_updates_retained_only=retained,median_ESS_per_million_actual_updates=metric['bulk_ess_median']/(actual/1e6),median_ESS_per_million_retained_updates=metric['bulk_ess_median']/(retained/1e6),sampler_phase_s=phase['gpu_sampling_s'],median_ESS_per_sampler_second=metric['bulk_ess_median']/phase['gpu_sampling_s'],native_total_s=phase['total_s'],adapter_wall_s=run['wall_including_normalization_seconds'],load_s=phase['load_polytope_s'],barrier_s=phase['barrier_rotation_s'],upload_s=phase['upload_to_gpu_s'],download_s=phase['download_from_gpu_s'],backtransform_s=phase['backtransform_s'],write_s=phase['write_output_s'],sample_bound_violation_fraction=metric['sample_bound_violation_fraction'],max_absolute_bound_violation=metric['max_absolute_bound_violation'],legacy_LP_excluded_reactions=878,legacy_mask_sha256=mask_hash,geometry_sha256=sha(directory/'rounding/polytope.npz'),start_sha256=hashlib.sha256(geometry['start'].tobytes()).hexdigest(),metrics_sha256=metric['metrics_code_sha256'],binary_sha256=binary_hash,git_commit=a.commit,run_file=str(dest/'run.json'),metric_file=str(metricpath)))
  csvsave(a.out/'task3_per_seed.csv',rows)
 scalar=['split_rhat_median','split_rhat_p95','bulk_ess_median','median_ESS_per_million_actual_updates','median_ESS_per_million_retained_updates','sampler_phase_s','median_ESS_per_sampler_second','native_total_s','adapter_wall_s','sample_bound_violation_fraction','max_absolute_bound_violation'];medians={k:statistics.median(r[k] for r in rows) for k in scalar}
 save(a.out/'task3_off_summary.json',dict(metadata=metadata,medians=medians,per_seed=rows,legacy_diagnostic_mask_sha256=mask_hash,legacy_diagnostic_mask=dict(excluded=878,source=str(directory/'diagnostic_mask.json'),scope='Explicit user-requested legacy LP mask to match the historical thinning table; separate from the 478-exclusion cached-target conservative assay.'),source_proof='sourceproof/thinning_semantics.json',note='All three are fixed-budget rotation-OFF controls, not convergence-budget searches. Medians aggregate the per-reaction summaries across three seeds. The update denominator includes all chains and discarded warmup; the native kernel performs exactly nsamples × thinning updates per chain. Hopsy uses cyclic scan and Naja uses random scan. Sampler phase is primary; native, process and output walls are secondary.'))
 lines=['# Task3: iJO1366 rotation-OFF control','',f'Production commit`{a.commit}`, binarySHA256`{binary_hash}`.','',metadata['task3_protocol'],'','The legacy LP mask excludes 878 reactions, matching the historical table. Geometry, start and mask SHA256s plus exact commands are saved per seed. This is separate from the 478-exclusion cached-target conservative assay.','', '| Seed | Median Rhat | Reaction p95 Rhat | Median bulk ESS | ESS /million actualupdates | GPU sampler s | Native total s |','|---|---:|---:|---:|---:|---:|---:|']
 for r in rows:lines.append(f"| {r['seed']} | {r['split_rhat_median']:.6f} | {r['split_rhat_p95']:.6f} | {r['bulk_ess_median']:.2f} | {r['median_ESS_per_million_actual_updates']:.3f} | {r['sampler_phase_s']:.4f} | {r['native_total_s']:.4f} |")
 lines.append(f"| Median | {medians['split_rhat_median']:.6f} | {medians['split_rhat_p95']:.6f} | {medians['bulk_ess_median']:.2f} | {medians['median_ESS_per_million_actual_updates']:.3f} | {medians['sampler_phase_s']:.4f} | {medians['native_total_s']:.4f} |")
 lines+=['','Actual updates = 16 × 11,000 × 96 = 16,896,000 including warmup. Retained-only updates = 15,360,000. ESS is the reaction median of pooled bulk ESS; the primary normalized value is ESS / 16.896. No estimated prefix wall times are used. Hopsy thinning specifies one coordinate update, not a full dimensional sweep; its cyclic scan differs from Naja random scan. [Sourceproof](sourceproof/THINNING_SEMANTICS.md).','', '[Per-seed components](task3_per_seed.csv) · [Full summary/provenance](task3_off_summary.json).']
 (a.out/'TASK3_ROTATION_OFF.md').write_text('\n'.join(lines)+'\n')
print(a.out)
