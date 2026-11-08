#!/usr/bin/env python3
"""Read-only reconstruction of the exact-KO panel, with proof and scheduler audits."""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, statistics, subprocess
import numpy as np
B=Path(__file__).resolve().parents[1]; R=B/'results/knockout_panel'; OUT=R/'overnight_audit'; OUT.mkdir(exist_ok=True)
def read(p): return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
manifest=read(R/'ready_manifest.json'); unique={}
base_model=B/'data/models/iJO1366';base=dict(np.load(base_model/'rounding/polytope.npz'))
base_hash=sha(base_model/'rounding/polytope.npz');base_mask=np.load(R/'base_cached_mask/active_reactions.npy')
assert read(R/'base_cached_mask/diagnostic_mask.json')['geometry_sha256']==base_hash
for entry in manifest['conditions']:
 root=Path(entry['geometry_directory']).parent.parent
 if root.name not in unique:unique[root.name]=root
rows=[]; checks=[]; conditions=[]
for gene,root in unique.items():
 info=read(root/'condition.json'); shared=root/'models/shared'; p=dict(np.load(shared/'rounding/polytope.npz'))
 current_hash=sha(shared/'rounding/polytope.npz'); mask=np.load(shared/'active_reactions.npy')
 inheritance=read(root/'conditional_mask_inheritance.json');embed=dict(np.load(root/'embedding.npz'))
 N,z0,keep=embed['N'],embed['z0'],embed['kept_base_rows'].astype(int)
 assert inheritance['geometry_sha256']==current_hash and inheritance['parent_geometry_sha256']==base_hash==info['base_geometry_sha256']
 assert inheritance['parent_mask_sha256']==sha(R/'base_cached_mask/active_reactions.npy')
 expectedA=base['A'][keep]@N;expectedb=base['b'].ravel()[keep]-base['A'][keep]@z0
 assert np.max(np.abs(p['A']-expectedA)/(1+np.abs(expectedA)))<=1e-10
 assert np.max(np.abs(p['b'].ravel()-expectedb)/(1+np.abs(expectedb)))<=1e-10
 assert np.max(np.abs(p['T']-base['T']@N)/(1+np.abs(p['T'])))<=1e-10
 assert np.max(np.abs(p['shift'].ravel()-(base['T']@z0+base['shift'].ravel())))<=1e-7
 dropped=np.setdiff1d(np.arange(len(base['b'])),keep)
 if len(dropped):
  norms=np.linalg.norm(base['A'][dropped],axis=1)
  assert np.max(np.linalg.norm(base['A'][dropped]@N,axis=1)/np.maximum(norms,1e-300))<=1e-9
  assert np.min(base['b'].ravel()[dropped]-base['A'][dropped]@z0)>=-1e-7
 ids=(shared/'gem/reaction_ids.txt').read_text().splitlines();lower=np.loadtxt(shared/'gem/l_bounds.csv',delimiter=',').ravel();upper=np.loadtxt(shared/'gem/u_bounds.csv',delimiter=',').ravel()
 forced=set(info.get('knockout_reaction_ids',[]));expected_mask=base_mask.copy()
 for i,reaction in enumerate(ids):
  if reaction in forced and lower[i]==upper[i] and np.linalg.norm(p['T'][i])<=1e-10*max(float(np.linalg.norm(base['T'][i])),1e-300) and abs(float(p['shift'].ravel()[i])-lower[i])<=1e-7:expected_mask[i]=False
 assert np.array_equal(mask,expected_mask)
 slack=p['b'].ravel()-p['A']@p['start'].ravel(); assert slack.min()>0
 weighted=p['A']/slack[:,None]; singular=np.linalg.svd(weighted,compute_uv=False)
 condition=dict(gene=gene,control=gene=='b0002',changed_bound_rows=info['changed_bound_rows'],dimension=p['A'].shape[1],rows=p['A'].shape[0],barrier_kappa_at_saved_start=float((singular[0]/singular[-1])**2),barrier_min_eigenvalue=float(singular[-1]**2),barrier_kappa_method='squared condition number of diag(1/slack)A via SVD; avoids forming ill-conditioned normal equations',shared_preparation_seconds=info['preparation_seconds'],LP_calls=info.get('LP_calls'),source_geometry_sha256=current_hash)
 glob=root/'models/global_reround'; proof_path=glob/'global_reround_proof.json'
 check=dict(gene=gene,shared_target_hash=current_hash,cached_parent_containment_recomputed=True,conservative_inherited_mask_recomputed=True,parent_cached_exclusions=int((~base_mask).sum()),conditional_exclusions=int((~mask).sum()))
 if proof_path.exists():
  proof=read(proof_path); q=dict(np.load(glob/'rounding/coordinate_change_proof.npz')); g=dict(np.load(glob/'rounding/polytope.npz'))
  ae=np.max(np.abs(g['A']-q['positive_row_scales'][:,None]*(p['A']@q['P']))/(1+np.abs(g['A'])))
  be=np.max(np.abs(g['b'].ravel()-q['positive_row_scales']*(p['b'].ravel()-p['A']@q['q']))/(1+np.abs(g['b'].ravel())))
  te=np.max(np.abs(g['T']-p['T']@q['P'])/(1+np.abs(g['T'])))
  se=np.max(np.abs(g['shift'].ravel()-(p['T']@q['q']+p['shift'].ravel()))/(1+np.abs(g['shift'].ravel())))
  assert max(ae,be,te,se)<1e-10 and np.all(q['positive_row_scales']>0)
  assert proof['status']=='passed' and proof['source_geometry_sha256']==current_hash and proof['destination_geometry_sha256']==sha(glob/'rounding/polytope.npz')
  assert np.array_equal(mask,np.load(glob/'active_reactions.npy'))
  check.update(status='passed',A_relative_error=float(ae),b_relative_error=float(be),T_relative_error=float(te),shift_relative_error=float(se),same_reaction_mask=True)
  condition['global_round_algorithm_seconds']=proof['global_round_algorithm_seconds']
 else:
  check.update(status='global_round_unavailable',preparation=read(root/'quality/preparation_status.json'))
 checks.append(check)
 for variant in ('global_cpu','stale_cpu','naja_rotate'):
  path=root/'quality'/variant/'seed101/quality.json'
  if not path.exists():
   rows.append(dict(gene=gene,control=gene=='b0002',variant=variant,seed=101,status='unavailable',state_file=str(path)));continue
  record=read(path); row=dict(gene=gene,control=gene=='b0002',**record,state_file=str(path),barrier_kappa_at_saved_start=condition['barrier_kappa_at_saved_start'])
  if record['status']=='complete':
   model=glob if variant=='global_cpu' else shared
   assert record['geometry_sha256']==sha(model/'rounding/polytope.npz')
   assert record['sample_bound_violation_fraction']==0 and record['maximum_stoichiometric_residual']<=1e-7
   assert record['split_rhat_median']<=1.05 and record['bulk_ess_median']>=100
   row['global_round_algorithm_seconds']=condition.get('global_round_algorithm_seconds')
   row['shared_preparation_seconds']=info['preparation_seconds']
  if record.get('prefix_curve_json'):
   endpoint=max(read(record['prefix_curve_json']),key=lambda z:z['draws_per_chain'])
   row.update(endpoint_draws=endpoint['draws_per_chain'],endpoint_rhat=endpoint['split_rhat_median'],endpoint_ess=endpoint['bulk_ess_median'])
  rows.append(row)
 conditions.append(condition)
state=read(B/'results/conditioned_prototype/quality_consumer_state.json');ids=sorted(set(j['job'] for j in state['jobs']))
accounting=subprocess.check_output(['sacct','-n','-X','-j',','.join(ids),'-o','JobIDRaw,State,ExitCode,Elapsed','-P'],text=True)
(OUT/'slurm-accounting.psv').write_text(accounting)
groups=[]
for variant in ('global_cpu','stale_cpu','naja_rotate'):
 rr=[r for r in rows if not r['control'] and r['variant']==variant];passed=[r for r in rr if r['status']=='complete']
 groups.append(dict(variant=variant,denominator=len(rr),passed=len(passed),capped=sum(r['status']=='no_quality_within_cap' for r in rr),unavailable=sum(r['status']=='unavailable' for r in rr),median_sampler_seconds_among_passes=statistics.median(r['measured_sampler_seconds'] for r in passed),median_ess_per_sampler_second_among_passes=statistics.median(r['median_ess_per_sampler_second'] for r in passed)))
pairs=[]
for gene in unique:
 rr={r['variant']:r for r in rows if r['gene']==gene and r['status']=='complete'}
 if gene=='b0002':continue
 for reference in ('global_cpu','stale_cpu'):
  if reference in rr and 'naja_rotate' in rr:
   a,g=rr[reference],rr['naja_rotate'];pair=dict(gene=gene,reference=reference,cpu_sampler_seconds=a['measured_sampler_seconds'],naja_sampler_seconds=g['measured_sampler_seconds'],cpu_over_gpu_sampler_time=a['measured_sampler_seconds']/g['measured_sampler_seconds'],gpu_over_cpu_ess_rate=g['median_ess_per_sampler_second']/a['median_ess_per_sampler_second'],setup_excluded=True)
   if reference=='global_cpu':
    mve=a['global_round_algorithm_seconds'];cor=g['correction_seconds'];common=a['shared_preparation_seconds'];ct=mve+a['measured_sampler_seconds'];gt=cor+g['measured_sampler_seconds']
    pair.update(global_round_algorithm_seconds=mve,naja_correction_seconds=cor,common_conditional_geometry_preparation_seconds=common,cpu_incremental_preparation_plus_sampler_seconds=ct,naja_incremental_preparation_plus_sampler_seconds=gt,cpu_over_gpu_incremental_preparation_plus_sampler_time=ct/gt,cpu_over_gpu_time_with_common_conditional_geometry=(common+ct)/(common+gt),global_mve_over_naja_correction_time=mve/cor)
   pairs.append(pair)
report=dict(generated_utc=datetime.now(timezone.utc).isoformat(),protocol='16 chains, seed101, CPU thinning2000 / Naja1000, warmup1000 retained draws, median classical splitRhat<=1.05, pooled bulkESS>=100, exact bounds/mass balance1e-7. Fresh selected-budget retimes only.',denominator_note='50 nontrivial unique exact gene knockouts; b0002 control kept separate. Validity-selected rank-stable panel, not fully stratified or population representative.',conditions=conditions,groups=groups,rows=rows,paired_comparisons=pairs,proof_audit=checks,samples_rehashed=False,sample_integrity_note='Saved geometry hashes independently rechecked; full saved sample arrays not rehashed in this audit. Frozen metrics and unmasked feasibility/mass-balance certificates retained.',scheduler_audit='slurm-accounting.psv')
contingency=dict(both_pass=0,rotate_only=0,inherited_only=0,neither_pass=0);fastest=[]
for c in conditions:
 if c['control']:continue
 rr={r['variant']:r for r in rows if r['gene']==c['gene'] and r['status']=='complete'}
 rotation='naja_rotate' in rr;inherited='stale_cpu' in rr
 contingency['both_pass' if rotation and inherited else 'rotate_only' if rotation else 'inherited_only' if inherited else 'neither_pass']+=1
 if not rotation:continue
 candidates=[]
 for cpu_variant in ('global_cpu','stale_cpu'):
  if cpu_variant in rr:
   r=rr[cpu_variant];setup=c['global_round_algorithm_seconds'] if cpu_variant=='global_cpu' else 0.
   candidates.append((setup+r['measured_sampler_seconds'],cpu_variant,setup,r['measured_sampler_seconds']))
 if not candidates:continue
 cpu,variant,setup,draw=min(candidates);gpu=rr['naja_rotate']['correction_seconds']+rr['naja_rotate']['measured_sampler_seconds'];common=c['shared_preparation_seconds']
 fastest.append(dict(gene=c['gene'],selected_cpu_variant=variant,cpu_preparation_seconds=setup,cpu_sampler_seconds=draw,cpu_preparation_plus_sampler_seconds=cpu,naja_correction_plus_sampler_seconds=gpu,cpu_over_gpu_preparation_plus_sampler=cpu/gpu,cpu_over_gpu_with_common_conditional_preparation=(common+cpu)/(common+gpu)))
report.update(rotate_vs_inherited_pass_contingency=contingency,fastest_tested_valid_cpu_comparisons=fastest)
fallback=[]
froot=B/'results/conditioned_extensions/b0031_scaled_global';fp=froot/'b0031/quality/global_cpu/seed101/quality.json'
if fp.exists() and read(fp).get('status')=='complete':
 f=read(fp);source=dict(np.load(R/'conditions/b0031/models/shared/rounding/polytope.npz'));g=dict(np.load(froot/'model/rounding/polytope.npz'));change=dict(np.load(froot/'model/rounding/coordinate_change_proof.npz'));proof=read(froot/'model/global_reround_proof.json')
 ae=np.max(np.abs(g['A']-change['positive_row_scales'][:,None]*(source['A']@change['P']))/(1+np.abs(g['A'])))
 be=np.max(np.abs(g['b'].ravel()-change['positive_row_scales']*(source['b'].ravel()-source['A']@change['q']))/(1+np.abs(g['b'].ravel())))
 te=np.max(np.abs(g['T']-source['T']@change['P'])/(1+np.abs(g['T'])))
 se=np.max(np.abs(g['shift'].ravel()-(source['T']@change['q']+source['shift'].ravel()))/(1+np.abs(g['shift'].ravel())))
 assert max(ae,be,te,se)<1e-10 and np.all(change['positive_row_scales']>0)
 assert np.linalg.svd(change['P'],compute_uv=False)[-1]>0 and proof['no_target_relaxation'] and proof['no_second_affine_hull_reduction']
 assert np.array_equal(np.load(froot/'model/active_reactions.npy'),np.load(R/'conditions/b0031/models/shared/active_reactions.npy'))
 assert f['geometry_sha256']==sha(froot/'model/rounding/polytope.npz') and f['samples_sha256']==sha(Path(f['measured_run'])/'samples.npy')
 assert f['sample_bound_violation_fraction']==0 and f['maximum_stoichiometric_residual']<=1e-7 and f['split_rhat_median']<=1.05 and f['bulk_ess_median']>=100
 fallback.append(dict(gene='b0031',status='complete',default_failed_attempt_preserved=True,initial_coordinate_scale=proof['initial_coordinate_scale'],global_round_algorithm_seconds=proof['global_round_algorithm_seconds'],selected_draws=f['selected_draws'],measured_sampler_seconds=f['measured_sampler_seconds'],split_rhat_median=f['split_rhat_median'],bulk_ess_median=f['bulk_ess_median'],sample_bound_violation_fraction=0.,sample_sha256_rechecked=True,A_relative_error=float(ae),b_relative_error=float(be),T_relative_error=float(te),shift_relative_error=float(se),state_file=str(fp),proof_file=str(froot/'model/global_reround_proof.json')))
report.update(separate_scaled_coordinate_fallbacks=fallback,available_valid_full_cpu_references_including_scaled_fallback=49+len(fallback))
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
for name,data in [('per_condition',rows),('paired_comparisons',pairs),('geometry',conditions),('fastest_cpu_comparisons',fastest),('separate_fallback_references',fallback)]:
 keys=list(dict.fromkeys(k for r in data for k in r));f=(OUT/(name+'.csv')).open('w');w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data);f.close()
lines=['# Exact-knockout conditioned benchmark audit','',f"Updated {report['generated_utc']}.",'',report['protocol'], '',report['denominator_note'],'','| Pipeline | Pass /50 | Capped at50k | Unavailable | Median sampler s among passes |','|---|---:|---:|---:|---:|']
for g in groups:lines.append(f"| {g['variant']} | {g['passed']}/50 | {g['capped']} | {g['unavailable']} | {g['median_sampler_seconds_among_passes']:.4f} |")
lines += ['', 'These per-method timing medians use different passing subsets and must not be divided into a speedup. Paired comparisons in the CSV use only targets passing both methods. CPU draw-call critical path excludes startup/IPC/output; GPU sampler includes allocation/slack/RNG and excludes download/write. Setup and target geometry are separate.', '', '| Paired reference | Common passing targets | Median CPU/GPU sampler-time ratio | Median GPU/CPU ESS-rate ratio |','|---|---:|---:|---:|']
for ref in ('global_cpu','stale_cpu'):
 pp=[p for p in pairs if p['reference']==ref];lines.append(f"| {ref} | {len(pp)} | {statistics.median(p['cpu_over_gpu_sampler_time'] for p in pp):.4f} | {statistics.median(p['gpu_over_cpu_ess_rate'] for p in pp):.4f} |")
pp=[p for p in pairs if p['reference']=='global_cpu']
lines += ['', '## Reused geometry: preparation plus sampling', '',
 f"On the **{len(pp)}/50** nontrivial targets passing both methods, the median paired ratio "
 f"(full-global MVE algorithm time + CPU draw critical path) / (Naja correction + GPU sampler) is "
 f"**{statistics.median(p['cpu_over_gpu_incremental_preparation_plus_sampler_time'] for p in pp):.2f}×**. "
 f"The paired range is {min(p['cpu_over_gpu_incremental_preparation_plus_sampler_time'] for p in pp):.3f}–"
 f"{max(p['cpu_over_gpu_incremental_preparation_plus_sampler_time'] for p in pp):.2f}×; Naja is slower on at least one case. "
 f"Adding the common validated conditional-geometry preparation to both sides gives a median "
 f"**{statistics.median(p['cpu_over_gpu_time_with_common_conditional_geometry'] for p in pp):.2f}×**. "
 f"Global-MVE time / Naja-correction time alone has median "
 f"{statistics.median(p['global_mve_over_naja_correction_time'] for p in pp):.1f}×.", '',
 'The first ratio starts from the same ready shared conditional geometry; it includes mathematical rerounding/correction and sampling, excludes output transfer/writes, external validation, LP-mask compilation, imports, and one-time base preparation. It is a **preparation-reuse result**, not a sampling-kernel speedup. The second includes shared per-condition preparation, whose wall time also includes file assembly. All per-case components are saved in paired_comparisons.csv. Nine rotate failures and the unavailable global reference remain in the full denominator; passing-subset ratios cannot stand in for success on all50 targets.']
lines += ['',f"Independently recomputed all-halfspace and reaction-map equivalence for {sum(c['status']=='passed' for c in checks)} full-reround targets; common reaction masks match. The b0031 full-reround reference is unavailable because PolyRound reported 'Rounding transformation not full dimensional'; the valid shared target and this failure remain preserved.",'','Inherited CPU uses the mandatory orthonormal exact-KO affine-hull projection. It can already repair mixing. The finite-width PGI capacity example has a different geometry and should be described separately. The unchanged b0002 control is excluded from all aggregate performance rows. A single seed per KO does not establish replicate reliability.', '', '[Per-target data](per_condition.csv) · [Paired comparisons](paired_comparisons.csv) · [Full proof/status audit](summary.json) · [Scheduler outcomes](slurm-accounting.psv).']
lines += ['', '## Fastest tested valid CPU and success contingency', '',
 f"Among50 nontrivial targets: **{contingency['both_pass']}** pass both inherited CPU and rotate; "
 f"**{contingency['rotate_only']}** pass rotate only; **{contingency['inherited_only']}** pass inherited CPU only; "
 f"**{contingency['neither_pass']}** pass neither. Correction is not uniformly beneficial.", '',
 f"On the {len(fastest)} rotate-passing targets, selecting the faster **passing** CPU option per target "
 '(inherited CPU, or full global MVE + CPU when needed), the median paired preparation-plus-sampling '
 f"CPU/Naja ratio is **{statistics.median(p['cpu_over_gpu_preparation_plus_sampler'] for p in fastest):.3f}×**, "
 f"or **{statistics.median(p['cpu_over_gpu_with_common_conditional_preparation'] for p in fastest):.3f}×** "
 'with common conditional-geometry preparation included. This finite-grid selection uses observed '
 'quality; it is not an independently validated automatic selector. Thus the full-reround-only14× '
 'result must not be described as a speedup over the fastest available CPU pipeline. '
 '[Per-case components](fastest_cpu_comparisons.csv).']
if fallback:
 f=fallback[0]
 lines += ['', '## Separate scaled-coordinate recovery of the last CPU reference', '',
  'All50 nontrivial conditions now have a valid full-PolyRound CPU reference:49 default successes '
  'plus this separately labeled fallback. The original b0031 default failure and all original table rows remain preserved. '
  'Uniform initial coordinate scaling by1e-6 avoids the absolute PolyRound dimensionality gate; default '
  'PolyRound thresholds remain unchanged. The composed invertible square map, all original halfspaces, '
  'reaction map, common conservative mask and fresh selected samples were independently checked, with '
  'no target relaxation or second hull reduction.', '',
  '| Gene | Initial units scale | Global MVE s | Selected draws | CPU sampler s | Rhat | ESS |',
  '|---|---:|---:|---:|---:|---:|---:|',
  f"| b0031 | 1e-6 | {f['global_round_algorithm_seconds']:.3f} | {f['selected_draws']} | {f['measured_sampler_seconds']:.4f} | {f['split_rhat_median']:.5f} | {f['bulk_ess_median']:.1f} |", '',
  '[Recovery source/proof/commands](../../conditioned_extensions/b0031_scaled_global/) · '
  '[Audited fallback row](separate_fallback_references.csv). '
  'This does not change the41 original rotate-passing paired subset, because b0031 rotate failed the original50k cap.']
(OUT/'CONDITIONED_BENCHMARK.md').write_text('\n'.join(lines)+'\n');print(json.dumps(groups,indent=2));print(OUT)
