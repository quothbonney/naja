#!/usr/bin/env python3
"""Isolated exact PGI-capacity target: prepare, run bounded pipelines, summarize."""
import argparse
import fcntl
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time

B = Path(__file__).resolve().parents[1]
R = B / 'results/conditioned_prototype'
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DEADLINE = datetime(2026, 11, 8, 11, 0, tzinfo=timezone.utc)
FROZEN = 'c7baf4f0ed282b936ac0dbac4b896bfbfc040eb06da6f84aff6605fddf4b0fc6'

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(obj, indent=2) + '\n')
    temporary.replace(path)

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def call(command, log, timeout=1200):
    remaining = (DEADLINE - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        raise TimeoutError('Conditional prototype deadline 2026-11-08 11:00 UTC')
    timeout = min(timeout, remaining)
    start = time.monotonic()
    with log.open('w') as stream:
        subprocess.run(list(map(str, command)), check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    return time.monotonic() - start

def prepare_shared(model):
    import numpy as np
    from scipy.optimize import linprog
    sys.path.insert(0, str(B))
    from diagnostic_mask import build_mask
    root = R / model
    root.mkdir(parents=True, exist_ok=True)
    done = root / 'preparation.json'
    if done.exists():
        assert json.loads(done.read_text())['status'] == 'complete'
        return
    source = B / f'data/raw/{model}.json'
    raw = json.loads(source.read_text())
    ids = [r['id'] for r in raw['reactions']]
    reaction = raw['reactions'][ids.index('PGI')]
    original_bounds = [reaction['lower_bound'], reaction['upper_bound']]
    reaction.update(lower_bound=-0.1, upper_bound=0.1)
    conditioned = root / f'{model}_PGI_capacity_0p1.json'
    save(conditioned, raw)
    base = B / f'data/models/{model}'
    inherited = root / 'models/shared'
    fresh = root / 'models/full_reround'
    inherited.mkdir(parents=True, exist_ok=True)
    (inherited / 'rounding').mkdir(exist_ok=True)
    shutil.copytree(base / 'gem', inherited / 'gem', dirs_exist_ok=True)
    t = time.monotonic()
    with np.load(base / 'rounding/polytope.npz') as z:
        A, b, T, shift = (z[k].copy() for k in ('A', 'b', 'T', 'shift'))
    b, shift = b.reshape(-1), shift.reshape(-1)
    i = ids.index('PGI')
    delta_A = np.vstack((T[i], -T[i]))
    delta_b = np.array([0.1 - shift[i], 0.1 + shift[i]])
    augmented_A, augmented_b = np.vstack((A, delta_A)), np.r_[b, delta_b]
    assembly_seconds = time.monotonic() - t
    t = time.monotonic()
    norms = np.linalg.norm(augmented_A, axis=1)
    nonzero=norms>0
    assert np.all(augmented_b[~nonzero]>0), 'Nonpositive zero-row bound'
    # Positive row scaling changes no halfspace. Explicit normalization keeps
    # HiGHS from numerically dropping tiny base rows; original A/b remain saved.
    lp = linprog(np.r_[np.zeros(T.shape[1]), -1.],
                 A_ub=np.column_stack((augmented_A[nonzero]/norms[nonzero,None], np.ones(nonzero.sum()))),
                 b_ub=augmented_b[nonzero]/norms[nonzero],
                 bounds=[(None, None)] * T.shape[1] + [(0, None)], method='highs',
                 options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not lp.success or lp.x[-1] <= 1e-8:
        raise ValueError(f'Capacity target not certified full dimensional: {lp.message}')
    start = lp.x[:-1]
    strict_lp_seconds = time.monotonic() - t
    assert np.min(augmented_b - augmented_A @ start) > 0
    np.savez(root / 'exact_delta.npz', extra_A=delta_A, extra_b=delta_b)
    np.savez(inherited / 'rounding/polytope.npz', A=augmented_A, b=augmented_b, T=T, shift=shift, start=start)
    lower = np.array([r['lower_bound'] for r in raw['reactions']])
    upper = np.array([r['upper_bound'] for r in raw['reactions']])
    np.savetxt(inherited / 'gem/l_bounds.csv', lower, delimiter=',')
    np.savetxt(inherited / 'gem/u_bounds.csv', upper, delimiter=',')
    center = T @ start + shift
    assert max(np.max(lower-center), np.max(center-upper)) <= 1e-7
    meta = json.loads((base / 'meta.json').read_text())
    meta.update(model_id=model+'_PGI_capacity_0p1', source=str(conditioned), source_sha256=sha(conditioned),
                inherited_from=str(base), m=len(augmented_b), min_start_slack=float(np.min(augmented_b-augmented_A@start)),
                timings=dict(assembly_seconds=assembly_seconds, strict_interior_lp_seconds=strict_lp_seconds,
                             total_seconds=assembly_seconds+strict_lp_seconds),
                objective_constraint='none; original bounds except declared PGI capacity [-0.1,0.1]',
                preprocessing_variant='shared_base_plus_exact_capacity_rows',
                command=sys.argv,hostname=platform.node())
    save(inherited / 'meta.json', meta)
    save(inherited / 'rounding/manifest.json', dict(format_version=2, model_name='shared', rounding_dim=T.shape[1],
                                                  n_reactions=len(ids), n_constraints=len(augmented_b)))
    save(root/'shared-preparation.json', dict(model=model,
         condition='PGI capacity restriction [-0.1,0.1]; knockdown, not exact knockout',
         original_PGI_bounds=original_bounds,conditioned_source=str(conditioned),conditioned_source_sha256=sha(conditioned),
         base_geometry_sha256=sha(base/'rounding/polytope.npz'), shared_geometry_sha256=sha(inherited/'rounding/polytope.npz'),
         exact_delta='[T_PGI; -T_PGI] z <= [0.1-shift_PGI; 0.1+shift_PGI]',
         strict_interior_radius=float(lp.x[-1]),shared_min_start_slack=float(np.min(augmented_b-augmented_A@start)),
         shared_assembly_seconds=assembly_seconds,shared_interior_lp_seconds=strict_lp_seconds,
         base_preprocessing_seconds=json.loads((base/'meta.json').read_text())['timings']['total_seconds']))


def prepare(model):
    import numpy as np
    sys.path.insert(0,str(B))
    from diagnostic_mask import build_mask
    root=R/model;root.mkdir(parents=True,exist_ok=True)
    done=root/'preparation.json'
    if done.exists():
        assert json.loads(done.read_text())['status']=='complete'
        return
    shared_process_seconds=call([B/'env-cpu/bin/python',Path(__file__),'shared','--model',model],root/'shared-preprocess.log')
    shared=json.loads((root/'shared-preparation.json').read_text())
    conditioned=Path(shared['conditioned_source'])
    fresh=root/'models/full_reround';inherited=root/'models/shared'
    full_preprocess_process = call([B/'env-cpu/bin/python', B/'cpu_baseline.py', 'prep', '--model', conditioned,
                                   '--out', fresh], root/'full-preprocess.log')
    for directory in (fresh, inherited):
        call([B/'env-cpu/bin/python', B/'scripts/validate_preprocessed.py', '--model', directory,
              '--raw', conditioned], root/f'validate-{directory.name}.log')
        build_mask(directory)
    # A common conservative mask: a reaction is excluded only when BOTH fresh
    # condition-specific LP masks independently exclude it. Wider/unresolved stay.
    common_active = np.load(fresh/'active_reactions.npy') | np.load(inherited/'active_reactions.npy')
    for directory in (fresh, inherited):
        report_path = directory / 'diagnostic_mask.json'
        report = json.loads(report_path.read_text())
        save(directory/'diagnostic_mask.independent.json', report)
        np.save(directory/'active_reactions.independent.npy', np.load(directory/'active_reactions.npy'))
        for record in report['reactions']:
            record['nearconstant_excluded'] = bool(not common_active[record['index']])
        report.update(active_reactions=int(common_active.sum()), excluded_nearconstant_reactions=int((~common_active).sum()),
                      common_policy='Intersection of excluded reactions from two fresh condition-specific LP masks; no base mask copied.',
                      independent_exclusion_count=report['excluded_nearconstant_reactions'])
        np.save(directory/'active_reactions.npy', common_active)
        save(report_path, report)
    fullmeta = json.loads((fresh/'meta.json').read_text())
    save(done, dict(**shared,status='complete',
                    full_geometry_sha256=sha(fresh/'rounding/polytope.npz'),
                    shared_preprocess_process_seconds=shared_process_seconds,
                    full_preprocess_process_seconds=full_preprocess_process,
                    full_polyround_seconds=fullmeta['timings']['polyround_seconds'],
                    base_cost_policy='Base preprocessing is a one-time library setup cost; per-condition totals exclude it but report it separately.',
                    excluded_common_lp_reactions=int((~common_active).sum()),script_sha256=SCRIPT_SHA,
                    condition_no_objective_constraint=True,eps_relaxation=0,created_utc=datetime.now(timezone.utc).isoformat()))


def physical_affinity():
    allowed = sorted(os.sched_getaffinity(0))
    topology, selected, seen = [], [], set()
    for cpu in allowed:
        p = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        socket, core = int((p/'physical_package_id').read_text()), int((p/'core_id').read_text())
        siblings = list(map(int, (p/'thread_siblings_list').read_text().strip().split(',')))
        topology.append(dict(cpu=cpu, socket=socket, core=core, siblings=siblings))
        if (socket, core) not in seen and len(selected)<16:
            assert set(siblings) <= set(allowed), 'Sibling reservation incomplete'
            selected.append(cpu)
            seen.add((socket, core))
    assert len(allowed)>=32 and len(selected)==16
    os.sched_setaffinity(0, selected)
    return dict(reserved_logical_cpus=int(os.environ['SLURM_CPUS_PER_TASK']), allowed=allowed,
                selected=selected, physical_cores=len(seen), topology=topology)


def run_pipeline(model, pipeline, seed):
    import numpy as np
    root = R / model
    prep = json.loads((root/'preparation.json').read_text())
    assert prep['status']=='complete'
    directory = root/'models'/('full_reround' if pipeline=='full_cpu' else 'shared')
    cpu = pipeline.endswith('cpu')
    affinity = physical_affinity() if cpu else None
    output = root/'runs'/pipeline/f'seed{seed}'
    output.mkdir(parents=True, exist_ok=True)
    evaluated=[]
    for draws in (10000, 50000):
        if datetime.now(timezone.utc)>=DEADLINE:
            save(output/'selection.json', dict(status='deadline', pipeline=pipeline, seed=seed));return
        destination = output/f'budget{draws}'
        marker = destination/'conditioned_diagnostics.json'
        if marker.exists():
            diagnostic = json.loads(marker.read_text())
        else:
            if cpu:
                proof=json.loads((B/'audit/hopsy-worker-timing-verification/iJO1366-real-parity.json').read_text())
                assert proof['status']=='passed' and proof['bitwise_equal']
                reference=json.loads((B.parent/proof['instrumented_run']/'sampling_math_timing.json').read_text())
                assert reference['proof']['instrumentation_module_sha256']==sha(B/'scripts/hopsy_worker_timing.py')
                command = [B/'env-cpu-native/bin/python', B/'scripts/timed_cpu_baseline.py', 'sample', '--model', directory,
                           '--out', destination, '--seed', seed, '--chains',16, '--draws',draws,
                           '--warmup',1000,'--thinning',2000,'--workers',16,'--chunk',draws+1000,'--build-variant','native']
                os.environ['HOPSY_CXXFLAGS']='-O3 -march=native -mtune=native'
                os.environ['HOPSY_BUILD_MANIFEST']=str(B/'audit/hopsy_native_build.json')
            else:
                command = [B/'scripts/enter.sh', B/'env-gpu/bin/python', B/'naja_runner.py', '--exe',B/'build-h100/naja',
                           '--model',directory,'--out',destination,'--seed',seed,'--chains',16,'--draws',draws,
                           '--warmup',1000,'--thinning',1000,'--tpb',512,'--barrier', 'rotate' if pipeline=='naja_rotate' else 'off',
                           '--timeout',1200]
            sampling_command_seconds=call(command, output/f'budget{draws}-sampling.log')
            if affinity:save(destination/'physical_affinity.json',affinity)
            metrics_process_seconds=call([B/'env-cpu/bin/python',B/'metrics.py','--samples',destination/'samples.npy','--model-dir',directory,
                  '--output',destination],output/f'budget{draws}-metrics.log')
            metric=json.loads((destination/'metrics.json').read_text());run=json.loads((destination/'run.json').read_text())
            assert metric['schema_version']==2 and metric['diagnostic_mask_status']=='model_lp_verified'
            assert run['status']=='complete'
            if not cpu:assert run['exe_sha256']==FROZEN
            # Complete original-reaction stoichiometric and declared PGI-bound
            # checks over every sample, independent of masked convergence cells.
            independent_check_started=time.monotonic()
            from scipy import sparse
            arrays=np.load(directory/'rounding/polytope.npz')
            S=sparse.csr_matrix(np.loadtxt(directory/'gem/A_eq.csv',delimiter=','))
            samples=np.load(destination/'samples.npy',mmap_mode='r')
            ids=(directory/'gem/reaction_ids.txt').read_text().splitlines();index=ids.index('PGI')
            max_equality=max_capacity=0.
            for chain in samples:
                for start in range(0,draws,1024):
                    flux=chain[start:start+1024]@arrays['T'].T+arrays['shift'].reshape(-1)
                    max_equality=max(max_equality,float(np.max(np.abs(S@flux.T))))
                    max_capacity=max(max_capacity,float(np.max(np.abs(flux[:,index])-.1)))
            quality=metric['split_rhat_median']<=1.05 and metric['bulk_ess_median']>=100
            bound_ok=metric['sample_bound_violation_fraction']==0 and max_capacity<=1e-7
            math_timing=json.loads((destination/'sampling_math_timing.json').read_text()) if cpu else None
            timer=math_timing['instrumented_wrapper_process_seconds'] if cpu else run['wall_including_normalization_seconds']
            compute_seconds=math_timing['worker_draw_critical_path_seconds'] if cpu else None
            phase_seconds=run.get('timings',{}).get('sample_seconds') if cpu else run['profile']['timing']['gpu_sampling_s']
            preprocessing=prep['full_preprocess_process_seconds'] if pipeline=='full_cpu' else prep['shared_preprocess_process_seconds']
            diagnostic=dict(status='complete', model=model,pipeline=pipeline,seed=seed,draws=draws,chains=16,warmup=1000,
                            thinning=2000 if cpu else 1000,quality_met=quality,bounds_met=bound_ok,
                            comparison_eligible=bool(quality and bound_ok and max_equality<=1e-7),
                            split_rhat_median=metric['split_rhat_median'],split_rhat_p95=metric['split_rhat_p95'],bulk_ess_median=metric['bulk_ess_median'],
                            sample_bound_violation_fraction=metric['sample_bound_violation_fraction'],
                            maximum_PGI_capacity_violation=max_capacity,maximum_stoichiometric_residual=max_equality,
                            sampling_process_inclusive_seconds=timer,per_condition_preprocessing_seconds=preprocessing,
                            sampling_phase_seconds=phase_seconds,
                            sampling_phase_scope=('hopsy.sample call including multiprocessing/IPC, warmup and retained sampling; excludes imports/load/chain initialization and output writes' if cpu else 'CoordinateHitAndRun phase: device sample allocation, initial slack GEMM, PRNG initialization, CHR kernel and synchronization; excludes inverse map/download/file writes/normalization'),
                            mathematical_compute_timer_status=('validated_worker_draw_call_critical_path_proxy; includes pybind/state copy and OS descheduling' if cpu else 'GPU sampler phase proxy; pure kernel timer not yet attached'),
                            worker_draw_critical_path_seconds=compute_seconds,
                            total_worker_draw_thread_cpu_seconds=math_timing['total_worker_draw_thread_cpu_seconds'] if cpu else None,
                            median_bulk_ess_per_worker_compute_second=metric['bulk_ess_median']/compute_seconds if cpu else None,
                            sampling_math_timing=math_timing,
                            sampling_subprocess_external_seconds=sampling_command_seconds,
                            primary_sampler_seconds=compute_seconds if cpu else phase_seconds,
                            primary_sampler_scope=('CPU MarkovChain.draw critical-path proxy; Python binding/state-copy included, IPC/startup/list assembly excluded' if cpu else 'GPU CoordinateHitAndRun phase proxy; device allocation/slack/RNG init included, output download/write/adapter excluded'),
                            primary_median_ess_per_sampler_second=metric['bulk_ess_median']/(compute_seconds if cpu else phase_seconds),
                            median_bulk_ess_per_sampling_phase_second=metric['bulk_ess_median']/phase_seconds,
                            measured_sampling_phase_time_to_quality_seconds=phase_seconds if quality and bound_ok and max_equality<=1e-7 else None,
                            primary_interpretation='Compare setup/correction and output-excluded selected-budget sampler phase separately; neither is an unqualified pure-kernel speedup.',
                            end_to_end_role='secondary operational timing',
                            diagnostic_process_seconds=metrics_process_seconds,independent_check_seconds=time.monotonic()-independent_check_started,
                            script_sha256=SCRIPT_SHA,
                            per_condition_total_seconds=timer+preprocessing,
                            conditioning_geometry_algorithm_seconds=(prep['full_polyround_seconds'] if pipeline=='full_cpu' else prep['shared_assembly_seconds']+prep['shared_interior_lp_seconds']),
                            conditioning_geometry_algorithm_scope=('PolyRound simplify/equality-eliminate/round phase' if pipeline=='full_cpu' else 'Inherited NPZ load/row assembly plus strict interior LP; excludes standalone Python startup and geometry writes'),
                            gpu_barrier_correction_seconds=run.get('profile',{}).get('timing',{}).get('barrier_rotation_s'),
                            correction_timing_policy='Naja correction included in sampling process; do not add it twice.',
                            geometry_sha256=sha(directory/'rounding/polytope.npz'),samples_sha256=sha(destination/'samples.npy'),
                            run_dir=str(destination),physical_affinity=affinity,criterion='median split-Rhat<=1.05 and median pooled bulk ESS>=100, independently require original+PGI bounds<=1e-7 and mass balance<=1e-7',
                            no_filtering=True,deadline_utc='2026-11-08T11:00:00+00:00')
            save(marker,diagnostic)
        evaluated.append(dict(draws=draws,record=str(marker),sampling_phase_seconds=diagnostic['sampling_phase_seconds'],sampling_process_seconds=diagnostic['sampling_process_inclusive_seconds'],comparison_eligible=diagnostic['comparison_eligible']))
        if diagnostic['comparison_eligible']:
            save(output/'selection.json',dict(status='passing',evaluated_budgets=evaluated,prior_budget_search_sampling_phase_seconds=sum(r['sampling_phase_seconds'] for r in evaluated[:-1]),**{k:v for k,v in diagnostic.items() if k!='status'}));return
    save(output/'selection.json',dict(status='no_quality_or_feasibility_within_cap',evaluated_budgets=evaluated,last=diagnostic))


def summarize():
    R.mkdir(parents=True,exist_ok=True)
    lock=(R/'summary.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX)
    rows=[]
    for p in sorted(R.glob('*/runs/*/seed*/selection.json')):
        row=json.loads(p.read_text());row['selection_file']=str(p);rows.append(row)
    save(R/'summary.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),rows=rows,
         scientific_scope='Explicit PGI capacity knockdown, not zero knockout. Same conditional target; fresh geometry vs same inherited geometry. Finite tested budgets/configurations; no assumed winner.',
         criterion='median Rhat<=1.05 and bulk ESS>=100; original/conditioned bounds and mass balance at1e-7 required separately',
         cost_policy='Primary: separately report setup/correction and selected-budget output-excluded sampler-phase ESS/sec; phase timers remain scoped proxies until CPU-worker/pure-kernel compute instrumentation. Secondary: complete process and per-condition operational wall. Base-preprocessing once; diagnostics/search separate; GPU correction already in process wall.'))
    lines=['# Conditional capacity study: saved progress', '',
           'Primary measures separate geometry setup/correction from output-excluded sampling. Primary sampler column uses CPU MarkovChain.draw critical-path proxy (pybind/state-copy included, IPC excluded) or GPU sampling-phase proxy (device allocation/slack/RNG initialization included). These are explicitly scoped, not unqualified arithmetic-only timers. All final budgets are actual executions. Criterion: median Rhat <=1.05 and median bulk ESS >=100; conditioned/original bounds and mass balance independently require1e-7. A cap/failure is not a passing comparison.', '',
           '| Pipeline | Seed | Status | Draws/chain | Geometry algorithm s | Local GPU correction s | Primary sampler proxy s | Median ESS / sampler s | Median Rhat | Bound fraction | Secondary inclusive process s |',
           '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def cell(value):
        return '—' if value is None else f'{value:.6g}'
    for selected in rows:
        r=selected.get('last',selected)
        lines.append('| '+' | '.join([r.get('pipeline',selected.get('pipeline','—')),str(r.get('seed',selected.get('seed','—'))),
            selected['status'],cell(r.get('draws')),cell(r.get('conditioning_geometry_algorithm_seconds')),
            cell(r.get('gpu_barrier_correction_seconds')),cell(r.get('primary_sampler_seconds')),
            cell(r.get('primary_median_ess_per_sampler_second')),cell(r.get('split_rhat_median')),
            cell(r.get('sample_bound_violation_fraction')),cell(r.get('sampling_process_inclusive_seconds'))])+' |')
    if not rows:lines.append('| Awaiting resource hold / preparation | | pending | | | | | | | | |')
    lines += ['', 'Standalone preparation process and all output/adapter costs remain in JSON as secondary operational measures. One-time base preparation is separately recorded; diagnostic/search costs are excluded from selected-budget phase times and preserved. Only independently passing, bound-compliant rows support quality-speed claims. [Protocol](README.md) · [structured rows](summary.json).','']
    table_temporary=R/f'.TABLE.{os.getpid()}.tmp'
    table_temporary.write_text('\n'.join(lines))
    table_temporary.replace(R/'TABLE.md')
    fcntl.flock(lock,fcntl.LOCK_UN)
    lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','shared','run','summarize']);p.add_argument('--model',default='iJO1366');p.add_argument('--pipeline',choices=['full_cpu','stale_cpu','naja_rotate','naja_off']);p.add_argument('--seed',type=int,default=101);a=p.parse_args()
    if a.action=='prepare':prepare(a.model)
    elif a.action=='shared':prepare_shared(a.model)
    elif a.action=='run':
        try:run_pipeline(a.model,a.pipeline,a.seed)
        except Exception as error:
            save(R/a.model/'runs'/a.pipeline/f'seed{a.seed}'/'selection.json',
                 dict(status='failed',model=a.model,pipeline=a.pipeline,seed=a.seed,error=repr(error)))
            raise
        finally:summarize()
    else:summarize()
