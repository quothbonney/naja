"""Capture three full-node samples before diagnostics; restore owned holds."""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

B = Path(__file__).resolve().parents[1]
audit = B / "audit/high-throughput"
lock = (audit / "full64-controller.lock").open("w")
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(audit / "full64-controller.pid").write_text(str(os.getpid())+"\n")
state_path = audit / "full64-state.json"
held_path = audit / "full64-held-jobs.json"
state = json.loads(state_path.read_text()) if state_path.exists() else dict(status="active",stage="waiting_budget",
    attempts=[dict(seed=101001,draws=10000,originally_planned_draws=10000,sample_job="451",metrics_job="453",
        path=str(B / "results/high_throughput/iJO1366/cpu_native512_physical64_thin2000/seed101001"),status="pending")])
held = json.loads(held_path.read_text()) if held_path.exists() else {}
deadline = datetime(2026,11,8,11,tzinfo=timezone.utc).timestamp()
launch_end = datetime(2026,11,8,10,15,tzinfo=timezone.utc).timestamp()
terminal = {"FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","DEADLINE"}

def atomic(path,data):
    temp = path.with_suffix(".tmp");temp.write_text(json.dumps(data,indent=2)+"\n");os.replace(temp,path)

def save():
    atomic(state_path,state);atomic(held_path,held)

def log(*values):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),*values,flush=True)

def call(command,check=True):
    return subprocess.run(command,cwd=B.parent,capture_output=True,text=True,check=check)

def status(job):
    result = call(["sacct","-n","-X","-j",job,"--format=State","-P"],False)
    values = [line.strip().split("|")[0].split()[0].rstrip("+") for line in result.stdout.splitlines() if line.strip()]
    return values[0] if values else "UNKNOWN"

def info(job):
    result = call(["scontrol","show","job",job,"-o"],False)
    return dict(re.findall(r"(\w+)=([^ ]*)",result.stdout)) if result.returncode == 0 else {}

def owned(job):
    data = info(job)
    return bool(data and data.get("WorkDir") in (str(B),str(B.parent)) and f"({os.getuid()})" in data.get("UserId","")),data

def pending():
    result = call(["squeue","-h","-r","-u",str(os.getuid()),"-o","%i|%j|%T|%R"])
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split("|",3)
        if len(fields) == 4:
            rows.append(fields)
    return rows

def hold(job,reason):
    if job in held and held[job].get("status") == "held":
        return
    ours,data = owned(job)
    if not ours or data.get("JobState") != "PENDING":
        return
    if data.get("Reason") == "JobHeldUser":
        # Preserve holds established outside this sequencer.
        return
    result = call(["scontrol","hold",job],False)
    if result.returncode == 0:
        held[job] = dict(status="held",reason=reason,job_name=data.get("JobName"),original_dependency=data.get("Dependency"),
                         original_reason=data.get("Reason"),workdir=data.get("WorkDir"),held_utc=datetime.now(timezone.utc).isoformat())
        save();log("Held",job,reason)

def defer_owned_recon_work():
    for job,name,job_state,reason in pending():
        if job_state != "PENDING":
            continue
        recon_named = name in ("Recon3D-validation-publish","bench-naja-Recon3D","bench-hopsy-Recon3D","metrics-Recon3D") or (
            "recon3d" in name.lower() and (name.startswith(("bench-","metrics-","quality-")) or "validation" in name.lower()))
        mask = job == "202_8"
        if name == "diagnostic-mask" and not mask:
            ours,data = owned(job)
            mask = ours and data.get("ArrayTaskId","").split("%")[0] == "8"
        if recon_named or mask:
            hold(job,"Deferred until all three full64CPU sampling processes finish")

def release_holds(at_deadline=False):
    for job,record in held.items():
        if record.get("status") != "held":
            continue
        ours,data = owned(job)
        if not ours or data.get("JobState") != "PENDING":
            record["status"] = "gone_or_already_running";continue
        cancel = at_deadline
        if job == "202_8" and status("150_6") in terminal:
            cancel = True
        command = ["scancel",job] if cancel else ["scontrol","release",job]
        result = call(command,False)
        record.update(status=("cancelled_deadline_or_failed_dependency" if cancel else "released") if result.returncode == 0 else "held",
                      restore_utc=datetime.now(timezone.utc).isoformat(),restore_stderr=result.stderr.strip())
        log("Restored",job,record["status"])
    save()

def submit(command):
    result = call(command);log(result.stdout.strip());return result.stdout.split()[-1]

def stage_pipeline(first_dependency):
    draws = state["attempts"][0]["draws"]
    previous = state["attempts"][0]["sample_job"]
    for seed in (202001,303001):
        attempt = next((a for a in state["attempts"] if a["seed"] == seed),None)
        if attempt is None:
            attempt = dict(seed=seed,draws=draws,status="pending",path=str(B / f"results/high_throughput/iJO1366/cpu_native512_physical64_thin2000/seed{seed}/budget{draws}"))
            state["attempts"].append(attempt);save()
        if "sample_job" not in attempt:
            dep = first_dependency if seed == 202001 else "afterok"
            attempt["sample_job"] = submit(["sbatch","--dependency="+dep+":"+previous,str(B / "scripts/high_throughput_cpu.sbatch"),
                                            str(seed),str(draws),attempt["path"]]);save()
        previous = attempt["sample_job"]
    last = previous
    for index,attempt in enumerate(state["attempts"]):
        if index == 0:
            if status(attempt["metrics_job"]) == "COMPLETED":
                continue
            if not attempt.get("deferred_metrics_replaced"):
                old_job = attempt["metrics_job"]
                call(["scancel",old_job],False)
                if old_job in held:held[old_job]["status"] = "cancelled_replaced_with45minute_after_all_samples_diagnostic"
                attempt["previous_metrics_job"] = old_job
                attempt["metrics_job"] = submit(["sbatch","--time=00:45:00","--dependency=afterok:"+last,
                    str(B / "scripts/high_throughput_metrics.sbatch"),attempt["path"]])
                attempt["deferred_metrics_replaced"] = True;save()
            else:
                call(["scontrol","update","JobId="+attempt["metrics_job"],"Dependency=afterok:"+last])
        elif "metrics_job" not in attempt:
            attempt["metrics_job"] = submit(["sbatch","--time=00:45:00","--dependency=afterok:"+last,
                                             str(B / "scripts/high_throughput_metrics.sbatch"),attempt["path"]]);save()
    state.update(stage="sampling_chain",last_sample_job=last);save()

def cancel_pending_pipeline():
    for attempt in state["attempts"]:
        for key in ("sample_job","metrics_job"):
            job = attempt.get(key)
            if job and status(job) in ("PENDING","UNKNOWN"):
                ours,_ = owned(job)
                if ours:call(["scancel",job],False)

def refresh_reports(force=False):
    signature=json.dumps(dict(stage=state['stage'],status=state['status'],attempts=[
        (a['seed'],a.get('sampling_state'),a.get('metrics_state'),a['status']) for a in state['attempts']]),sort_keys=True)
    if force or signature!=state.get('last_report_signature'):
        subprocess.run([sys.executable,str(B/'scripts/summarize_high_throughput.py')],check=False,timeout=60)
        report=B/'scripts/new_results_report.py'
        if report.exists():subprocess.run([sys.executable,str(report)],check=False,timeout=60)
        handoff=B/'scripts/write_cpu_handoff.py'
        if handoff.exists():subprocess.run([sys.executable,str(handoff)],check=False,timeout=60)
        state['last_report_signature']=signature;save()

save()
try:
    while time.time() < deadline and state["status"] == "active":
        try:
            if state["stage"] in ("waiting_budget","sampling_chain","verify_first"):
                defer_owned_recon_work()
            first = state["attempts"][0]
            if state["stage"] == "waiting_budget":
                hold(first["metrics_job"],"Full64 diagnostics deferred until all samples captured")
                protocol = audit / "cpu-selected-budget.json"
                if protocol.exists():
                    selected = json.loads(protocol.read_text());assert selected["status"] == "validated"
                    first["draws"] = selected["draws"];state["selected_protocol"] = selected;save()
                    if time.time() >= launch_end:
                        state["status"] = "launch_deadline_partial";save();break
                    stage_pipeline("afterok")
                    call(["scontrol","release",first["sample_job"]])
                    state["full64_gate_released_utc"] = datetime.now(timezone.utc).isoformat();save()
            if state["stage"] == "sampling_chain":
                failed = None
                for attempt in state["attempts"]:
                    attempt["sampling_state"] = status(attempt["sample_job"])
                    if attempt["sampling_state"] in terminal:failed = attempt
                    elif attempt["sampling_state"] == "COMPLETED":attempt["status"] = "sampled"
                if failed:
                    proof_path = Path(first["path"])/"shared_stream_identity.json"
                    if failed["seed"] == 101001 and proof_path.exists() and json.loads(proof_path.read_text()).get("status") == "failed":
                        # The sampler completed but cross-worker bytes differed.
                        # Cancel the gated, not-yet-runnable follow-ons and get
                        # actual first-case diagnostics before rebuilding them.
                        for a in state["attempts"][1:]:
                            for key in ("sample_job","metrics_job"):
                                if a.get(key):call(["scancel",a[key]],False)
                        state.setdefault("cancelled_pipeline_attempts",[]).extend(state["attempts"][1:])
                        state["attempts"] = [first]
                        call(["scontrol","update","JobId="+first["metrics_job"],"Dependency=afterany:"+first["sample_job"]])
                        call(["scontrol","release",first["metrics_job"]],False)
                        state["stage"] = "verify_first";save()
                    else:
                        state.update(status="failed_sampling",failed_attempt=failed);save();break
                elif all(a["sampling_state"] == "COMPLETED" for a in state["attempts"]):
                    state.update(stage="diagnostics",all_samples_captured_utc=datetime.now(timezone.utc).isoformat());save()
                    release_holds()
            if state["stage"] == "verify_first":
                if status(first["metrics_job"]) == "COMPLETED":
                    q = json.loads((Path(first["path"])/"quality.json").read_text())
                    if q["status"] != "passed":state["status"] = "first_quality_failed";save();break
                    first.update(status="passed",quality=q);save();stage_pipeline("afterany")
            if state["stage"] == "diagnostics":
                done = True
                for attempt in state["attempts"]:
                    diagnostic = status(attempt["metrics_job"]);attempt["metrics_state"] = diagnostic
                    qpath = Path(attempt["path"])/"quality.json"
                    if diagnostic == "COMPLETED" and qpath.exists():
                        q = json.loads(qpath.read_text());attempt.update(quality=q,status="passed" if q["status"] == "passed" else "quality_failed")
                    elif diagnostic in terminal:attempt["status"] = "failed_diagnostics"
                    else:done = False
                if done:
                    state["status"] = "complete" if all(a["status"] == "passed" for a in state["attempts"]) else "incomplete_quality_or_diagnostics"
            save()
            log(state["stage"],[(a["seed"],a.get("sampling_state"),a.get("metrics_state"),a["status"]) for a in state["attempts"]])
            refresh_reports()
        except Exception as error:
            state["last_error"] = repr(error);save();log("Retryable sequence error",repr(error))
        if state["status"] == "active":time.sleep(min(15,max(0,deadline-time.time())))
finally:
    expired = time.time() >= deadline
    if expired:
        state["status"] = "deadline_partial" if state["status"] == "active" else state["status"]
        for attempt in state["attempts"]:
            for key in ("sample_job","metrics_job"):
                job = attempt.get(key)
                if job and status(job) not in terminal|{"COMPLETED"}:
                    ours,_ = owned(job)
                    if ours:call(["scancel",job],False)
    elif state["status"] != "complete":
        cancel_pending_pipeline()
    release_holds(expired)
    save()
    refresh_reports(True)
    log("Full64 sequencer stopped",state["status"])
