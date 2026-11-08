#!/usr/bin/env python3
"""Read-only closeout evidence: CLI syntax, measured extensions, sized asset inventory."""
from datetime import datetime,timezone
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import zipfile

B=Path(__file__).resolve().parents[1]
R=B/'results';C=R/'closeout';C.mkdir(exist_ok=True)
def read(path):return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def csv_write(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

now=datetime.now(timezone.utc).isoformat()
repo=B.parent/'naja-test-jack/naja'
commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
cli=read(C/'cli_validation.json')
cli_repo=Path(cli.get('source_repo',repo))
parser_paths=[cli_repo/'src/cli/prepare/cmd_prepare.cpp',cli_repo/'src/cli/sample/cmd_bulk.cpp',cli_repo/'src/cli/sample/common.cpp']
task5=dict(status='source_and_parser_validated',validation=str(C/'cli_validation.json'),source_base_commit=cli['source_commit'],
    parser_sha256={str(path):sha(path) for path in parser_paths},
    validated_binary_sha256=cli['binary_sha256'],tests=cli['tests'],
    scope='Existing projected exactKO b4025 fixture; prepare executed without sampling, bulk --dry-run only. Native prepare does inherited delta assembly, not new affine-hull elimination.',
    limitations=['Merged bulk parser supports explicit --seed and --sample-dtype; legacy C7-only validation is preserved separately.',
                 'n-samples is total native draws per chain; retained warmup is discarded externally in benchmark adapters.',
                 'Current prepare reports FEASIBLE_START_FAILED but writes all prepared names to out-model-list; that name list is not a guaranteed feasible subset.',
                 'The prepare subcommand does not accept --help in this binary; shared sample --help and bulk --help are saved.'])

fresh=read(R/'fresh16_validation/iJO1366/summary.json')
previous_notes=read(C/'task_notes.json') if (C/'task_notes.json').exists() else {}
previous_extensions={row['source_run']:row for row in previous_notes.get('task7',{}).get('rows',[])}
extensions=[]
for method in ('cpu','gpu'):
    for item in fresh['fixed_groups'][method]['rows']:
        if 'extension' not in item:continue
        value=item['extension'];folder=Path(value['run_dir']);run=read(folder/'run.json');metric=read(folder/'metrics.json')
        row=dict(method=method,seed=value['seed'],retained_per_chain=value['draws_per_chain'],chains=16,
            warmup_per_chain=value['warmup'],thinning=value['thinning'],fixed_budget_status=item['fixed_status'],
            extension_status=value['status'],sampler_phase_seconds=value['sampler_phase_seconds'],
            worker_draw_critical_path_seconds=value.get('worker_draw_critical_path_seconds'),
            process_wall_seconds_secondary=value['process_seconds_secondary'],
            native_profile_total_seconds=run.get('profile',{}).get('timing',{}).get('total_s'),
            median_split_rhat=value['split_rhat_median'],median_bulk_ess=value['pooled_bulk_ess_median'],
            sample_bound_violation_fraction=value['sample_bound_violation_fraction'],
            assay='historical878 original-model LP exclusions; not the new478 reassay',
            source_run=str(folder),samples_bytes=(folder/'samples.npy').stat().st_size,
            samples_sha256=previous_extensions.get(str(folder),{}).get('samples_sha256') or sha(folder/'samples.npy'),
            sample_hash_note='Reuses previously checked frozen-file SHA; availability refresh does not rehash large arrays.',run_json_sha256=sha(folder/'run.json'),
            metrics_json_sha256=sha(folder/'metrics.json'),estimator_sha256=metric['metrics_code_sha256'])
        if method=='gpu':row['sampler_binary_sha256']=run['exe_sha256']
        else:row['sampler_binary_sha256']=run['hopsy_module_sha256']
        row['hopsy_reported_build']=run.get('hopsy_build')
        extensions.append(row)
csv_write(C/'fresh_seed_extension_timings.csv',extensions)
task7=dict(status='measured_existing_records_no_reruns',source_base_commit='23873f01d3268a79bbf84499a66adce5e88cc6cb',
    source_scope='Historical campaign base plus recorded local benchmark patches; exact per-run binary/module hashes are authoritative. These were not rerun under finalC9.',
    rows=extensions,csv=str(C/'fresh_seed_extension_timings.csv'),
    timing_scope='CPU worker draw critical path excludes startup/IPC/output; inclusive CPU hopsy.sample is separate. GPU sampler includes allocation/slack/RNG; secondary process time includes output/normalization. All are actual executions, no prefix interpolation.',
    caveat='These extensions passed after the fixed10kCPU/2kGPU configuration failed for those seeds. They do not rescue fixed-budget generalization or the legacy2.73x claim.',
    separate_new_protocol=str(R/'fresh16_conservative/iJO1366/summary.json'))

candidates={
 'bulk512':['high_throughput/iJO1366/comparison.csv','high_throughput/iJO1366/comparison.json','high_throughput/iJO1366/AUDITED_COMPARISON.md'],
 'crossmodel':['crossmodel_extension/cached_assay_table.csv','crossmodel_extension/cached_assay_summary.json'],
 'conditioned50':['knockout_panel/overnight_audit/per_condition.csv','knockout_panel/overnight_audit/paired_comparisons.csv','knockout_panel/overnight_audit/fastest_cpu_comparisons.csv','knockout_panel/overnight_audit/summary.json','knockout_panel/overnight_audit/conditioned_panel_summary.svg','knockout_panel/overnight_audit/conditioned_panel_summary.pdf','knockout_panel/overnight_audit/figure_source.csv'],
 'capacity_curves':['conditioned_prototype/iJO1366/quality/rhat_curve.csv','conditioned_prototype/iJO1366/quality/rhat_vs_chain_length.svg','conditioned_prototype/iJO1366/quality/rhat_vs_chain_length.pdf','conditioned_prototype/iJO1366/quality/time_to_target.csv'],
 'nsight_timeline':['nsight_timeline/naja_h100_utilization_timeline.svg','nsight_timeline/naja_h100_utilization_timeline.pdf','nsight_timeline/timeline_source.csv','nsight_timeline/phases.csv','nsight_timeline/ncu/kernel_summary.csv','nsight_timeline/ncu/roofline_source.csv','nsight_timeline/commands.sh'],
 'barrier_dimension':['barrier_dimension_scaling/source.csv','barrier_dimension_scaling/medians.csv','barrier_dimension_scaling/barrier_dimension_scaling.svg','barrier_dimension_scaling/barrier_dimension_scaling.pdf','barrier_dimension_scaling/summary.json'],
 'fresh16_fixed':['fresh16_conservative/iJO1366/table.csv','fresh16_conservative/iJO1366/summary.json','fresh16_conservative/iJO1366/VALIDATION.md'],
 'fresh16_extensions':['closeout/fresh_seed_extension_timings.csv','fresh16_validation/iJO1366/summary.json'],
 'uniformity':['uniformity/summary.json'],
 'chain_scaling':['figures/chain_scaling_source.csv','figures/h100_chain_scaling.svg'],
 'verification':['setup/gpu-correctness.log','barrier_profile/verification/verification.json','coordinate_kernel/verification.json'],
 'environment':['continuation_20261108/ENVIRONMENT_AND_PROTOCOL.md','setup/h100-environment.txt','setup/binary.sha256','setup/source.patch'],
 'final_task1':['closeout/task1/summary.json','closeout/task1/TASK1_VERIFICATION.md','closeout/task1/iJO_parity.json','closeout/task1/f64_precision_retention_probe.json','closeout/task1/final_compatibility_default/summary.json','closeout/task1/final_compatibility_default/FINAL_GATE.md','closeout/task1/final_compatibility_default/gpu-correctness-tests.log','closeout/task1/RECON1_reduced_f64/original_bounds.json','closeout/task1/RECON1_reaction_f64/original_bounds.json'],
 'final_task2':['closeout/task2_3/final-release/TASK2_BARRIER.md','closeout/task2_3/final-release/task2_barrier_summary.csv','closeout/task2_3/final-release/task2_per_seed.csv','closeout/task2_3/final-release/task2_barrier_summary.json','closeout/task2_3/final-release/task2_barrier_dimension_scaling.svg','closeout/task2_3/final-release/task2_barrier_dimension_scaling.pdf','closeout/task2_3/final-release/manifest.json'],
 'final_task3':['closeout/task2_3/final-release/TASK3_ROTATION_OFF.md','closeout/task2_3/final-release/task3_per_seed.csv','closeout/task2_3/final-release/task3_off_summary.json','closeout/task2_3/final-release/task3_FCC_OFF_parity.json','closeout/task2_3/final-release/resource_accounting.json'],
 'final_task4':['closeout/task4/final_release/summary.json','closeout/task4/final_release/base/metrics.csv','closeout/task4/final_release/base/report.ncu-rep','closeout/task4/final_release/exact_PGI_KO/metrics.csv','closeout/task4/final_release/exact_PGI_KO/report.ncu-rep'],
 'bounded_Recon3D':['closeout/recon3d_bounded/protocol.json','closeout/recon3d_bounded/pilot_decision.json','closeout/recon3d_bounded/capture.json','closeout/recon3d_bounded/jobs.json','closeout/recon3d_bounded/summary.json','closeout/recon3d_bounded/RECON3D_ATTEMPT.md','closeout/recon3d_bounded/naja-fcc66fe-source.tar.gz','../scripts/recon3d_bounded.py','../scripts/recon3d_bounded_capture.sbatch','../scripts/recon3d_bounded_metrics.sbatch','../scripts/recon3d_bounded_report.py'],
 'bulk512_measured_experimental_code':['coordinate_kernel/source.tar.gz','coordinate_kernel/kernel.patch','coordinate_kernel/verification.json','coordinate_kernel/compiler_resources.json','../scripts/build_coordinate_kernel.sh','../build-h100-coordinate/CMakeCache.txt'],
 'bulk512_original_inputs':['../data/models/iJO1366/rounding/polytope.npz','../data/models/iJO1366/rounding/manifest.json','../data/models/iJO1366/meta.json','../data/models/iJO1366/gem/A_eq.csv','../data/models/iJO1366/gem/b_eq.csv','../data/models/iJO1366/gem/l_bounds.csv','../data/models/iJO1366/gem/u_bounds.csv','../data/models/iJO1366/gem/reaction_ids.txt','../data/raw/iJO1366.json','../data/raw/manifest.json'],
 'bulk512_assay_proofs':['high_throughput/iJO1366/cached_target_audit/model/active_reactions.npy','high_throughput/iJO1366/cached_target_audit/model/diagnostic_mask.json','high_throughput/iJO1366/cached_target_audit/reaggregation_validation.json','high_throughput/iJO1366/cached_target_audit/array_hash_verification.json','../scripts/reaggregate_cached_metrics.py','../scripts/validate_full64_reaggregation.py','../scripts/audit_full64_comparison.py','../scripts/cached_target_mask.py','../scripts/cached_target_mask_parallel.py'],
 'bulk512_drivers':['../naja_runner.py','../cpu_baseline.py','../metrics.py','../scripts/high_throughput_sample.py','../scripts/high_throughput_cpu.sbatch','../scripts/coordinate512_worker.py','../scripts/coordinate512.sbatch','../scripts/benchmark_coordinate_kernel.py','../scripts/benchmark_coordinate_kernel.sbatch','../scripts/timed_cpu_baseline.py','../scripts/hopsy_worker_timing.py','../scripts/quality_diagnostics.py'],
 'bulk512_native_CPU_build':['../audit/hopsy_native_build.json','../audit/hopsy_native_compiler_target.txt','../source/hopsy-1.7.0.tar.gz','../wheels-native/hopsy-1.7.0-cp312-cp312-linux_x86_64.whl','../scripts/record_native_build.py','../audit/hopsy-worker-timing-verification/iJO1366-real-parity.json','../audit/metrics-parallel-verification.json','../audit/high-throughput/cpu-selected-budget.json','../audit/high-throughput/full64-state.json'],
 'final_release_source':['closeout/naja-preprint-closeout-20261108.tar.gz']}
assets=[]
table2_manifest=read(C/'task2_3/final-release/manifest.json')
candidates['final_task2_prepared_inputs']=[str(B/'data/models'/model/leaf)
    for model in table2_manifest['models'] for leaf in ('rounding/polytope.npz','meta.json')]
uniformity=read(R/'uniformity/summary.json')
candidates['uniformity_retained_inputs']=[f'uniformity/{name}/samples.npy' for name in uniformity['runs']]
task4=read(C/'task4/final_release/summary.json')
candidates['final_task4_prepared_inputs']=[]
for item in task4['rows']:
    command=item['command'];model=Path(command[command.index('--model-dir')+1])
    candidates['final_task4_prepared_inputs'].append(str((model/'rounding/polytope.npz').relative_to(R))
        if model.is_relative_to(R) else str(model/'rounding/polytope.npz'))
    if item['case']=='exact_PGI_KO':
        for path in (model/'condition.json',model.parents[1]/'condition.json'):
            if path.exists():candidates['final_task4_prepared_inputs'].append(str(path.relative_to(R)))
# Capture/analysis files appear asynchronously; include only artifacts that exist,
# inspecting sizes without rereading large arrays.
recon_root=R/'closeout/recon3d_bounded'
for pattern in ('source_snapshots/*','main/run.json','main/samples.npy','main/metrics.json',
                'main/reaction_metrics.csv','main/mass_balance.json','main/native/**/profile.json',
                'main/native/**/samples.npy','pilot/run.json','pilot/samples.npy','pilot/metrics.json',
                'pilot/reaction_metrics.csv','pilot/mass_balance.json','slurm_accounting.psv'):
    candidates['bounded_Recon3D'].extend(str(path.relative_to(R)) for path in recon_root.glob(pattern) if path.is_file())
for group,paths in candidates.items():
    for rel in paths:
        path=(R/rel).resolve()
        assets.append(dict(group=group,path=str(path),relative_path=rel,exists=path.exists(),
            bytes=path.stat().st_size if path.exists() else None,
            modified_utc=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat() if path.exists() else None,
            paper_item=('Figure2_verified_byte_data_component' if group=='final_task4' else
                        'Table2_verified_barrier_column_data_component' if group=='final_task2' and path.suffix in ('.csv','.json') else
                        'unassigned_no_manuscript_source')))
bulk=read(R/'high_throughput/iJO1366/comparison.json')
for row in bulk['rows']:
    if row['assay']!='cached_target_mask478':continue
    path=Path(row['source_run'])/'samples.npy'
    assets.append(dict(group='bulk512_retained_array',path=str(path),relative_path=str(path.relative_to(R)),
        exists=path.exists(),bytes=path.stat().st_size,modified_utc=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
        paper_item='unassigned_no_manuscript_source'))
csv_write(C/'data_availability_candidates.csv',assets)
requested=[f'Table{i}' for i in range(2,6)]+[f'TableS{i}' for i in range(1,6)]+[f'Figure{i}' for i in range(1,4)]
overrides=read(C/'paper_mapping_overrides.json') if (C/'paper_mapping_overrides.json').exists() else {}
mapping=[]
for item in requested:
    row=dict(paper_item=item,status='UNVERIFIABLE_without_manuscript_source',verified_artifact=None,bytes=None,
        mapping_note='No exact assignment is asserted without manuscript source or explicit user mapping.')
    if item in overrides:
        row.update(overrides[item]);path=Path(row['verified_artifact']);row['bytes']=path.stat().st_size if path.exists() else None
    mapping.append(row)
csv_write(C/'paper_item_mapping.csv',mapping)
save(C/'data_availability_inventory.json',dict(snapshot_utc=now,assets=assets,requested_item_mapping=mapping,
    mapping_scope='User explicitly assigned Table2 barrier-column and Figure2 byte-data components. Other item numbering cannot be established without manuscript source. Generated benchmark PDFs alone do not establish manuscript numbering.'))
task9=dict(status='sized_assets_inventory_complete_two_components_user_assigned_other_numbering_unverifiable',source_base_commit=commit,
    mapping=mapping,assets_csv=str(C/'data_availability_candidates.csv'),inventory_json=str(C/'data_availability_inventory.json'),
    inspected_scope='Workspace root and project files; excluded dependency, environment, tool and package trees. git-tracked project contains no manuscript source.',
    byte_count_note='Exact on-disk bytes at snapshot time. Six512-chain retained arrays total'+str(sum(a['bytes'] for a in assets if a['group']=='bulk512_retained_array'))+'bytes; these were not copied or resampled.')
bundle=B/'data/models/iJO1366/rounding/polytope.npz'
with zipfile.ZipFile(bundle) as archive:
    entries={entry.filename:dict(uncompressed_bytes=entry.file_size,compressed_bytes=entry.compress_size) for entry in archive.infolist()}
task9['bulk512_reproduction_contract']=dict(measured_gpu_binary_sha256='51854688114677b14de3c110c262d210281de1f351f1ae6df6be994c2cc67e82',
    measured_code_archive=str(R/'coordinate_kernel/source.tar.gz'),measured_code_patch=str(R/'coordinate_kernel/kernel.patch'),
    prepared_model=str(B/'data/models/iJO1366'),rounding_bundle_members=entries,
    start_point='Stored as start.npy inside polytope.npz; no separate standalone start file is required.',
    raw_model_metadata=read(B/'data/models/iJO1366/meta.json'),
    measured_version_note='The512-chain timing row uses the archived experimental coordinate-specialized build, not the final generic C9release. Finalrelease source archive is listed separately.',
    CPU_native_manifest=str(B/'audit/hopsy_native_build.json'),
    refresh_policy='Stat sizes only for reproduction inputs/source archives and large retained arrays; no copying or rehashing.')
recon=R/'closeout/recon3d_bounded'
task8=dict(status='not_started')
if (recon/'protocol.json').exists():
    task8=dict(status=read(recon/'capture.json').get('status','pending') if (recon/'capture.json').exists() else 'pending',
        protocol=read(recon/'protocol.json'),capture=read(recon/'capture.json') if (recon/'capture.json').exists() else {},
        result=read(recon/'summary.json') if (recon/'summary.json').exists() else None,
        jobs=read(recon/'jobs.json'),source_commit='fcc66fe63d5a6700786aebcf0b925374340e4c8b',
        scope='Immutable FCC binary, explicit GPUrotation/whitening, cached equality-only target, no iterative MVE, one bounded GPUattempt. Final C9 CLI changed compatibility default but does not invalidate the explicitly selected GPUbackend run.')
notes=dict(generated_utc=now,task5=task5,task6=dict(status='WAIVED_BY_USER',action='No further corpus lookup or missing-input request.'),task7=task7,task8=task8,task9=task9)
save(C/'task_notes.json',notes)
prepare=next(t for t in cli['tests'] if t['name']=='prepare_existing_projected_target')['argv']
bulk_command=next(t for t in cli['tests'] if t['name']=='bulk_syntax_only_no_sampling')['argv']
lines=['# Closeout task notes: CLI, measured extensions and data availability','',f'Snapshot {now}. Tasks5/7/9 use existing data or syntax checks; task8 is the separately authorized one-hour-bounded GPU attempt.','',
 '## Task 5 — exact prepare/bulk commands','',
 'The following real fixture uses the already projected exact b4025 target. Preparation completed successfully; the merged bulk command with explicit seed and F64 output passed its parser/contract dry-run without GPU sampling. Source commit: `'+cli['source_commit']+'`; validated release binary SHA256: `'+cli['binary_sha256']+'`.','',
 '```bash',shlex.join(prepare),shlex.join(bulk_command),'```','',
 'Remove `--dry-run` from the second command to sample this fixture later. It produces 11,000 native draws/chain; discarding 1,000 warmup draws/chain remains an external benchmark step. This is not a new corpus run.','',
 'For an externally supplied conditioned library with a pre-rounded base, the supported prepare form is:','',
 '```bash','naja sample prepare --models-root MODEL_ROOT --model-list ALL_MODELS.txt \\\n  --base-model-dir BASE_MODEL --mode symlink --out-model-list PREPARED.txt',
 'naja sample bulk --models-root MODEL_ROOT --model-list PREPARED.txt \\\n  --out-root OUT --name RUN_NAME --gpus 0 --seed 101 --sample-dtype float64 \\\n  --n-chains 16 --n-samples 11000 --tpb 512 --thinning 1000 \\\n  --start-policy file --extra-constraints auto --constraint-eps 0 \\\n  --barrier-rotate --barrier-rotate-backend gpu --backmap \\\n  --bounds-policy filter --bounds-eps 1e-7 --write-npy','```','',
 'The capitalized paths in the second example are placeholders. Exact zero-width KO targets require affine-hull projection before inheritance can have a strict interior; native prepare does not perform that reduction.','',
 'Merged bulk supports `--seed` and `--sample-dtype`; the legacy C7 check is preserved separately. The per-model run manifests contain the exact benchmark commands. Prepare currently reports feasible-start failures without excluding those names from the written list. Shared `sample --help` is valid; `sample prepare --help` returns an unknown-flag error. See [CLI validation and saved help](cli_validation.json).','',
 '## Task 6 — waived','', 'Waived explicitly by the user. No further corpus lookup or missing-input request.','',
 '## Task 7 — actual fresh-seed extension timings','',
 '| Method | Seed | Retained/chain | Worker draw s | Inclusive sampler s | Actual process wall s |',
 '|---|---:|---:|---:|---:|---:|']
for row in extensions:
    worker='—' if row['worker_draw_critical_path_seconds'] is None else f"{row['worker_draw_critical_path_seconds']:.6f}"
    lines.append(f"| {row['method']} | {row['seed']} | {row['retained_per_chain']:,} | {worker} | {row['sampler_phase_seconds']:.6f} | {row['process_wall_seconds_secondary']:.6f} |")
lines += ['', 'All three extensions pass the historical 878-mask quality gate and unmasked 1e-7 bounds. These are actual fresh processes, not interpolated prefix times. GPU sampler time includes device allocation/slack/RNG setup; CPU worker time excludes IPC/output. Secondary process walls include output and relevant wrapper/normalization overhead. Original fixed-budget failures remain failures.','',
 '[Full timing/source/hash CSV](fresh_seed_extension_timings.csv). The separately completed new 16-chain protocol uses different seeds and fixed larger budgets; see [its report](../fresh16_conservative/iJO1366/VALIDATION.md).','',
 '## Task 9 — data availability and numbering','',
 'The user explicitly assigns the Task2 barrier-column data to Table2 and Task4 byte data to Figure2. Their exact source paths and sizes are recorded in [paper_item_mapping.csv](paper_item_mapping.csv). Only those components are verified; other columns/captions and the remaining numbered items are unassigned without manuscript source. Both measured Figure2-byte cases have m750; no m1500 conditional measurement is supplied.','',
 '[Content-organized artifact paths and exact byte sizes](data_availability_candidates.csv) · [Machine-readable inventory](data_availability_inventory.json). This includes available CSV/JSON tables, SVG/PDF figures, profiling source CSVs, hardware/correctness evidence and six 512-chain retained arrays. No large data were copied.','',
 'The512-chain measured GPU row uses experimental binary SHA51854688… and its own archived source.tar.gz/kernel.patch/verification records. The final generic C9release is a separate code artifact. The inventory now includes the original iJO1366 bundle (A,b,start,T,shift), GEM S/bounds/reaction IDs, raw-model metadata, sampler/diagnostic/instrumentation scripts, the conservative mask/proofs, native hopsy source/wheel/build manifest and timer verification. The start point is embedded as start.npy inside polytope.npz; no standalone start file is needed. Reproduction-asset refresh uses stat sizes only.','']
if task8['status']!='not_started':
    position=lines.index('## Task 9 — data availability and numbering')
    context=['## Task 8 — one bounded Recon3D attempt','',
        f"Capture state: **{task8['status']}**. Immutable FCC source/binary, explicit GPU rotation backend and whitening; one H100 allocation capped59minutes, pilot+one main run only. CPU normalization/diagnostics use a separate job after GPU release.",'',
        'The cached target has n10600,d4861,m18477. PolyRound-library equality elimination costs128.8417s, iterative MVE0s; total cached preprocessing287.7203s includes original-target FVA134.2945s. The2723 pruned items are strictly inactive inequality rows, not reactions. All10600 original reaction FVA widths are at least0.2; this assay uses no LP exclusions.','',
        'Pilot:512chains,64warmup+64retained,thinning810. Main:512chains,1000warmup+800retained,F64,TPB128; thinning follows the frozen pilot-lag/runtime rule. No additional attempt is permitted after OOM, timeout or failed quality. This equality-only study does not establish a matched-quality CPU sampling speedup or globally optimized time-to-quality.','',
        '[Frozen protocol](recon3d_bounded/protocol.json) · [Current capture](recon3d_bounded/capture.json) · [Slurm jobs](recon3d_bounded/jobs.json).','']
    if task8['result']:
        result=task8['result'];context += ['Final diagnostic state: **'+result['status']+'**. Full measured values are in [summary.json](recon3d_bounded/summary.json).','']
    else:context += ['Main-run quality remains unmeasured until CPU diagnostics finish; no convergence claim is assigned yet.','']
    lines[position:position]=context
position=lines.index('[Content-organized artifact paths and exact byte sizes](data_availability_candidates.csv) · [Machine-readable inventory](data_availability_inventory.json). This includes available CSV/JSON tables, SVG/PDF figures, profiling source CSVs, hardware/correctness evidence and six 512-chain retained arrays. No large data were copied.')
verified=['| Paper component | Verified source file | Bytes |','|---|---|---:|']
for item in mapping:
    if item['verified_artifact']:
        file=Path(item['verified_artifact']);verified.append(f"| {item['paper_item']} | [{file.name}]({file.relative_to(C)}) | {item['bytes']} |")
lines[position:position]=verified+['']
(C/'task_notes.md').write_text('\n'.join(lines))
print(C/'task_notes.md')
