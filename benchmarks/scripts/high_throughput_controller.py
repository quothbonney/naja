"""Detached bounded continuation for the512-chain one-seed-first prototype."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

B = Path(__file__).resolve().parents[1]
audit = B / "audit/high-throughput"
audit.mkdir(exist_ok=True)
lock = (audit / "controller.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
(audit / "controller.pid").write_text(str(os.getpid()) + "\n")
state_path = audit / "state.json"
stop_launch = datetime(2026,11,8,10,30,tzinfo=timezone.utc).timestamp()
stop_watch = datetime(2026,11,8,11,tzinfo=timezone.utc).timestamp()
labels = dict(cpu="cpu_native512_physical64_thin2000", cpu48="cpu_native512_physical48_thin2000",
              gpu="naja512_tpb128_thin1000", gpu500="naja512_tpb128_thin500")
defaults = dict(cpu=10000, cpu48=10000, gpu=2000, gpu500=5000)
fallbacks = dict(cpu=20000, cpu48=20000, gpu=5000, gpu500=5000)
seeds = [101001,202001,303001]
terminal = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "DEADLINE", "DEPENDENCY"}

def path_for(method, seed, draws):
    path = B / f"results/high_throughput/iJO1366/{labels[method]}/seed{seed}"
    return path if draws == defaults[method] else path / f"budget{draws}"

state = json.loads(state_path.read_text()) if state_path.exists() else dict(methods={
    "cpu":dict(status="active", attempts=[dict(seed=101001,draws=10000,sample_job="451",metrics_job="453",path=str(path_for("cpu",101001,10000)),status="pending")]),
    "gpu":dict(status="active", attempts=[dict(seed=101001,draws=2000,sample_job="450",metrics_job="452",path=str(path_for("gpu",101001,2000)),status="pending")])})
state["methods"].setdefault("gpu500",dict(status="active",attempts=[dict(seed=101001,draws=5000,sample_job="472",metrics_job="473",path=str(path_for("gpu500",101001,5000)),status="pending")]))
state["methods"].setdefault("cpu48",dict(status="active",first_seed_only=True,attempts=[dict(seed=101001,draws=10000,sample_job="468",metrics_job="471",path=str(path_for("cpu48",101001,10000)),status="pending")]))

def save():
    temp = state_path.with_suffix(".tmp")
    temp.write_text(json.dumps(state,indent=2)+"\n")
    os.replace(temp,state_path)

def log(*values):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),*values,flush=True)

def status(job):
    proc = subprocess.run(["sacct","-n","-X","-j",job,"--format=State","-P"],capture_output=True,text=True)
    lines = [line.strip().split("|")[0].split()[0].rstrip("+") for line in proc.stdout.splitlines() if line.strip()]
    return lines[0] if lines else "UNKNOWN"

def submit(command):
    proc = subprocess.run(command,cwd=B.parent,capture_output=True,text=True,check=True)
    log(proc.stdout.strip())
    return proc.stdout.split()[-1]

def ensure_jobs(method, attempt):
    if "sample_job" not in attempt:
        prior = next((old for old in state["methods"][method]["attempts"][:-1]
                      if old["path"] == attempt["path"] and "sample_job" in old
                      and status(old["sample_job"]) == "COMPLETED"), None)
        if prior is not None:
            attempt["sample_job"] = prior["sample_job"]
            attempt["sampling_reused_from_attempt"] = prior["sample_job"]
            log("Reusing previously executed exact-budget sampling",method,attempt["sample_job"])
        else:
            script_method = "gpu" if method == "gpu500" else method
            command = ["sbatch",str(B / f"scripts/high_throughput_{script_method}.sbatch"),str(attempt["seed"]),str(attempt["draws"])]
            if method == "gpu500":
                command.append("500")
            if method.startswith("cpu"):
                command.append(attempt["path"])
            attempt["sample_job"] = submit(command)
        save()
    if "metrics_job" not in attempt:
        attempt["metrics_job"] = submit(["sbatch","--dependency=afterok:"+attempt["sample_job"],str(B / "scripts/high_throughput_metrics.sbatch"),attempt["path"]])
        save()

save()
while time.time() < stop_watch:
    for method, group in state["methods"].items():
        if group["status"] != "active":
            continue
        attempt = group["attempts"][-1]
        try:
            if "sample_job" not in attempt or "metrics_job" not in attempt:
                if time.time() < stop_launch:
                    ensure_jobs(method,attempt)
                else:
                    group["status"] = "launch_deadline_partial"
                    save()
                    continue
            sampling = status(attempt["sample_job"])
            diagnostic = status(attempt["metrics_job"])
            attempt["sampling_state"],attempt["metrics_state"] = sampling,diagnostic
            protocol_path = audit / "cpu-selected-budget.json"
            if method == "cpu" and sampling == "PENDING" and protocol_path.exists():
                protocol = json.loads(protocol_path.read_text())
                if protocol.get("status") == "validated" and attempt["draws"] != protocol["draws"]:
                    attempt.setdefault("originally_planned_draws",attempt["draws"])
                    attempt["draws"] = protocol["draws"]
                    attempt["selected_budget_protocol"] = str(protocol_path)
            log(method,attempt["seed"],attempt["draws"],sampling,diagnostic)
            if method == "cpu48" and sampling == "COMPLETED" and diagnostic == "TIMEOUT" and not attempt.get("diagnostic_cap_retry") and time.time() < stop_launch:
                attempt["previous_metrics_job"] = attempt["metrics_job"]
                attempt["metrics_job"] = submit(["sbatch","--time=00:45:00",str(B / "scripts/high_throughput_metrics.sbatch"),attempt["path"]])
                attempt["diagnostic_cap_retry"] = True
                save();continue
            if sampling in terminal or diagnostic in terminal:
                group["status"] = attempt["status"] = "failed_job"
                if sampling in terminal:
                    subprocess.run(["scancel",attempt["metrics_job"]],check=False)
                save()
                continue
            quality = Path(attempt["path"]) / "quality.json"
            if diagnostic != "COMPLETED" or not quality.exists():
                save()
                continue
            result = json.loads(quality.read_text())
            attempt["quality"] = result
            if result["status"] == "pilot_prefix_selected":
                attempt["status"] = "selection_pilot"
                next_seed,next_draws = attempt["seed"],result["selected_draws"]
            elif result["status"] == "passed":
                attempt["status"] = "passed"
                attempt["draws"] = result["draws_per_chain"]
                if method == "cpu48":
                    protocol = dict(status="validated",draws=attempt["draws"],seed=attempt["seed"],chains=512,
                                    warmup=1000,thinning=2000,source_run=attempt["path"],source_quality=result,
                                    note="Full64CPU first seed shares the same per-chain RNG streams and target; selected from a fresh passing48physicalCPU retiming. Its own full diagnostics remain required.")
                    temp = protocol_path.with_suffix(".tmp")
                    temp.write_text(json.dumps(protocol,indent=2)+"\n")
                    os.replace(temp,protocol_path)
                    # The full64 sequencer releases451 only after its hold/chain
                    # safeguards and deferred-diagnostics dependencies are ready.
                if group.get("first_seed_only"):
                    group["status"] = "pilot_complete"
                    save()
                    continue
                index = seeds.index(attempt["seed"])
                if method in ("gpu","gpu500") and index == 0 and not state.get("gpu_selection"):
                    group["status"] = "waiting_gpu_selection"
                    save()
                    continue
                if index == len(seeds)-1:
                    group["status"] = "complete"
                    save()
                    continue
                next_seed,next_draws = seeds[index+1],attempt["draws"]
            elif result["status"] in ("pilot_budget_failed_rhat_or_bounds", "failed_quality") and attempt["draws"] < fallbacks[method]:
                attempt["status"] = "budget_failed"
                next_seed,next_draws = attempt["seed"],fallbacks[method]
            else:
                group["status"] = attempt["status"] = "quality_cap"
                save()
                continue
            if time.time() >= stop_launch:
                group["status"] = "launch_deadline_partial"
                save()
                continue
            group["attempts"].append(dict(seed=next_seed,draws=next_draws,path=str(path_for(method,next_seed,next_draws)),status="pending"))
            save()
            ensure_jobs(method,group["attempts"][-1])
        except Exception as error:
            log("Retryable controller error",method,repr(error))
            group["last_controller_error"] = repr(error)
            save()
    gpu_groups = {key:state["methods"][key] for key in ("gpu","gpu500")}
    if not state.get("gpu_selection") and all(g["status"] != "active" for g in gpu_groups.values()):
        candidates = [(key,attempt) for key,group in gpu_groups.items() for attempt in group["attempts"]
                      if attempt["seed"] == 101001 and attempt.get("status") == "passed"]
        if candidates:
            def sampler_phase(candidate):
                return json.loads((Path(candidate[1]["path"])/"run.json").read_text())["profile"]["timing"]["gpu_sampling_s"]
            winner,first = min(candidates,key=sampler_phase)
            state["gpu_selection"] = dict(winner=winner,selection="Fastest passing first-seed GPU sampler phase; empirical selection, not a global optimum. Includes GPUsampler allocation/init; excludes outputhandling.",
                                           pilot_sampler_phase_seconds={key:sampler_phase((key,a)) for key,a in candidates},
                                           pilot_process_seconds={key:a["quality"]["process_seconds"] for key,a in candidates})
            for key,group in gpu_groups.items():
                group["status"] = "waiting_prefault" if key == winner else "pilot_not_selected"
            save()
        else:
            state["gpu_selection"] = dict(status="no_passing_pilot")
            save()
    decision_path = audit / "prefault-decision.json"
    selection = state.get("gpu_selection",{})
    if selection.get("winner") and not selection.get("prefault_decision") and decision_path.exists():
        decision = json.loads(decision_path.read_text())
        selection["prefault_decision"] = decision
        winner = selection["winner"]
        group = state["methods"][winner]
        if decision["choice"] == "frozen":
            first = next(a for a in reversed(group["attempts"]) if a.get("status") == "passed")
            group["status"] = "active"
            next_attempt = dict(seed=202001,draws=first["draws"],path=str(path_for(winner,202001,first["draws"])),status="pending")
            group["attempts"].append(next_attempt)
            save()
            if time.time() < stop_launch:
                ensure_jobs(winner,next_attempt)
        else:
            group["status"] = "prefault_superseded"
            save()
    subprocess.run([sys.executable,str(B / "scripts/summarize_high_throughput.py")],check=False)
    if all(group["status"] not in ("active","waiting_prefault") for group in state["methods"].values()):
        break
    time.sleep(30)
log("Controller stopped",{k:v["status"] for k,v in state["methods"].items()})
