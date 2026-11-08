"""Operational owner snapshot; live state files remain authoritative."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path

B=Path(__file__).resolve().parents[1]
A=B/'audit/high-throughput'

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def process(pidfile,script,statefile=None):
    try:pid=int(pidfile.read_text())
    except (OSError,ValueError):pid=None
    alive=False
    if pid:
        try:alive=Path(f'/proc/{pid}').stat().st_uid==os.getuid() and script.name in Path(f'/proc/{pid}/cmdline').read_text()
        except OSError:pass
    return dict(pid=pid,alive_at_snapshot=alive,pid_file=str(pidfile),script=str(script),state_file=str(statefile) if statefile else None,
        restart_cwd=str(B.parent),restart_command=f"setsid /usr/bin/python3 -u '{script}' >> '{A/(script.stem+'.restart.log')}' 2>&1 < /dev/null &")

def generate():
    full=read(A/'full64-state.json',{})
    sample_chain='→'.join(a.get('sample_job','pending') for a in full.get('attempts',[]))
    diagnostic_jobs='/'.join(a.get('metrics_job','pending') for a in full.get('attempts',[]))
    controllers={
        'full64_sequence':process(A/'full64-controller.pid',B/'scripts/full64_sequence_controller.py',A/'full64-state.json'),
        'deadline_guard':process(A/'deadline-guard.pid',B/'scripts/full64_deadline_guard.py',A/'deadline-guard-status.json'),
        'Recon3D_completion':process(B/'audit/recon3d-completion/watcher.pid',B/'scripts/watch_recon3d_completion.py',B/'audit/recon3d-completion/watcher-state.json')}
    terminal={
        'old_main_dispatcher':dict(pid_file=str(A/'controller.pid'),state_file=str(A/'state.json'),expected='Stopped intentionally after CPU48 validated and full64 delegated; do not resurrect.'),
        'coordinate_dispatcher':dict(pid_file=str(A/'coordinate-controller.pid'),state_file=str(A/'coordinate-state.json'),expected='Complete3/3; do not resurrect.'),
        'prefault_dispatcher':dict(pid_file=str(A/'prefault-controller.pid'),state_file=str(A/'prefault-state.json'),expected='Stopped after current replicate when coordinate candidate adopted; two-seed history remains partial; do not resurrect.')}
    worker=B/'scripts/hopsy_worker_timing.py'
    result=dict(schema_version=1,generated_utc=datetime.now(timezone.utc).isoformat(),owner='full64_sequence_controller + independent deadline_guard',
        interactive_stop_utc='2026-11-08T05:22:37Z',autonomous_hard_deadline_utc='2026-11-08T11:00:00Z',
        root_independent_slurm_cleanup=dict(job_id='557',begin_utc='2026-11-08T11:05:00Z',scope='Workspace/prefix-scoped final cleanup and new report refresh; benign overlap with local11:00 guard.'),
        controllers=controllers,intentionally_terminal_controllers=terminal,
        full64_sequence=full,held_jobs=read(A/'full64-held-jobs.json',{}),held_manifest=str(A/'full64-held-jobs.json'),
        validated_cpu_protocol=read(A/'cpu-selected-budget.json',{}),
        readiness_note='All512-chain CPU seeds use validated12k retained +1000warmup,thin2000,native64workers pinned one thread/core;128logical reserved.451 waits afterany150_6. Original150_6 nine-hour cap ends08:23:25UTC if it does not finish earlier.',
        sequence_contract=f'{sample_chain} sampling with afterok dependencies. Firstfull64 exact-array identity to validated48core source gates success. Diagnostics{diagnostic_jobs} depend on finalsample,8CPU128GiB45min each. No diagnostics/mask can consume the128CPU sampling window. Failed identity triggers first fresh diagnostics before rebuilding follow-ons.',
        cleanup_contract='Only same-UID, same-workspace pending Recon3D mask/validator/campaign jobs are held; original deps recorded. Restore after all3samples captured. Deadline11:00 cancels remaining owned sequence jobs and held pending cases. Independent guard restarts an unexpectedly dead active full64 sequencer and separately enforces cleanup. Never restart an intentionally terminal dispatcher.',
        worker_instrumentation=dict(module=str(worker),sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
            validated_sha256='a01b0bfd384c37ad483abdc0840fc28611ce34132d654e91e1e794a09e0732ba',bootstrap=str(B/'scripts/timed_cpu_baseline.py'),
            parity_proof=str(B/'audit/hopsy-worker-timing-verification/iJO1366-real-parity.json'),
            scope='Max per-PID accumulated C++draw-call time includes binding/state-copy/descheduling; excludes startup/IPC/list/finalarray assembly. Not pure arithmetic-only. Keep these modules frozen.'),
        scientific_status=dict(
            fresh16_validation=str(B/'results/fresh16_validation/iJO1366/summary.json'),
            fresh16_fixed_result='CPU2/3 and GPU1/3 pass fixed budgets. Separate bounded extensions all pass; legacy2.73x fixed-budget estimate does not validate universally.',
            worker_math_tables=str(B/'results/WORKER_MATH_BENCHMARKS.json'),
            sampling_phase_tables=str(B/'results/SAMPLING_BENCHMARKS.json'),
            iml_budget_caps=str(B/'results/iml_time_to_target/summary.json'),
            gpu512=str(B/'results/high_throughput/iJO1366/summary.json'),
            gpu512_scope='Coordinate518546 build3/3valid, actualsampler median13.0012s. Initialsinglepaired1.29x specialization gain did not replicate in controlled3seed512audit; do not generalize it. Full64CPU result remains pending until actual jobs and diagnostics finish.',
            cpu48_scope='Only one48physical-core preliminary: fresh12k passes; workercritical222.3549s. Never substitute it for full64CPU benchmark.'),
        refresh_commands=[f"python3 '{B/'scripts/benchmark_status.py'}'",f"python3 '{B/'scripts/summarize_high_throughput.py'}'",f"python3 '{B/'scripts/new_results_report.py'}'"],
        new_report=str(B/'results/RESULTS_SINCE_UTILIZATION_REQUEST.md'),live_state_note='This is a snapshot. Read full64-state.json, full64-held-jobs.json, controller logs and root RESUME_STATUS.json for later changes. New central report auto-refreshes on meaningful sequence state changes. Historical reports and canonical run records are not rewritten.')
    temp=A/'OWNER_HANDOFF.json.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(A/'OWNER_HANDOFF.json')
    lines=['# CPU throughput owner handoff','',f"Snapshot: {result['generated_utc']}. Live state JSON files are authoritative after this snapshot.",'',
        'Interactive stop05:22:37UTC; autonomous batch deadline11:00UTC. Root cleanup557 runs11:05UTC.','',
        'The full64 sequencer and independent deadline guard are detached processes. The guard restarts only an unexpectedly absent active sequencer. It never resurrects completed main/coordinate/prefault dispatchers.','',
        f'Full sampling chain{sample_chain} uses128logical/64physical cores,512chains,12kretained,1000warmup,thin2000.451 awaits originalRecon3D150_6 completion/timeout (cap08:23:25UTC). Firstwhole-array identity proves shared per-chain streams before afterok follow-ons run; if identity fails, actual first-case diagnostics gate rebuilt follow-ons.','',
        f'Diagnostics{diagnostic_jobs} run only after the finalsample,8CPU128GiB45min each. Original453 was cancelled/replaced. Mask202_8 and any new workspace Recon3D validators/campaign jobs remain held until sample capture completes; their original dependencies are recorded in full64-held-jobs.json. Deadline cleanup cancels owned unfinished cases and held pending jobs; no holds are left deliberately.','',
        '| Controller | PID at snapshot | Alive | State / PID file |','|---|---:|---|---|']
    for name,c in controllers.items():lines.append(f"| {name} | {c['pid']} | {c['alive_at_snapshot']} | {c['state_file'] or c['pid_file']} |")
    lines += ['', 'Restart only if the relevant controller is unexpectedly dead while its state is active, from the workspace root. Exclusive locks reject duplicates. The deadline guard normally performs this recovery itself.','']
    for name,c in controllers.items():lines += [name+':','```bash',c['restart_command'],'```','']
    lines += ['Scientific interpretation:',
        '- CPU48 is a one-seed preliminary, not the full64 comparison. Full64 rows stay pending until observed.',
        '- Fresh16 fixed CPU passes2/3; fixed GPU passes1/3. Separate extensions cannot rescue a universal legacy2.73x claim.',
        '- Worker draw timing excludes IPC but includes binding/state-copy; GPU sampler phase includes allocation/slackGEMM/RNG initialization. Neither is labeled pure arithmetic-only.',
        '- iML guideline3/3passes, verifiedF64GPU3/3passes; CPUthin2000 fixed10k0/3passes (cap only, not eventual convergence).',
        '- Coordinate5123/3valid, actualphase median13.0012s. Initialsinglepaired1.29x specialization gain is exploratory and did not replicate in the controlled512audit.',
        '',f"Refresh central report: python3 {B/'scripts/new_results_report.py'}",f"Status entry point: {B/'RESUME.md'}",'']
    tmp=A/'OWNER_HANDOFF.md.tmp';tmp.write_text('\n'.join(lines));tmp.replace(A/'OWNER_HANDOFF.md')
    return result

if __name__=='__main__':
    r=generate();print(r['owner'],{k:v['alive_at_snapshot'] for k,v in r['controllers'].items()})
