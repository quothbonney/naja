"""Bounded prefault selection and disjoint-replicate continuation."""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

B = Path(__file__).resolve().parents[1]
audit = B / "audit/high-throughput"
lock = (audit / "prefault-controller.lock").open("w")
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(audit / "prefault-controller.pid").write_text(str(os.getpid())+"\n")
state_path = audit / "prefault-state.json"
state = json.loads(state_path.read_text()) if state_path.exists() else dict(status="waiting_frozen_selection",attempts=[])
launch_end = datetime(2026,11,8,10,30,tzinfo=timezone.utc).timestamp()
watch_end = datetime(2026,11,8,11,tzinfo=timezone.utc).timestamp()
terminal = {"FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","DEADLINE"}

def read(path):
    return json.loads(path.read_text())

def write(path,data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data,indent=2)+"\n")
    os.replace(temp,path)

def save():
    write(state_path,state)

def log(*values):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),*values,flush=True)

def status(job):
    result = subprocess.run(["sacct","-n","-X","-j",job,"--format=State","-P"],capture_output=True,text=True)
    lines = [line.strip().split("|")[0].split()[0].rstrip("+") for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else "UNKNOWN"

def submit(command):
    result = subprocess.run(command,cwd=B.parent,capture_output=True,text=True,check=True)
    log(result.stdout.strip())
    return result.stdout.split()[-1]

def case_path(seed,draws):
    path = B / f"results/high_throughput/iJO1366/naja512_tpb128_thin{state['thin']}_prefault8/seed{seed}"
    return path if draws == state["selected_draws"] else path / f"budget{draws}"

def ensure_jobs(attempt):
    if "sample_job" not in attempt:
        attempt["sample_job"] = submit(["sbatch",str(B / "scripts/high_throughput_prefault.sbatch"),"sample",
            "--thin",str(state["thin"]),"--draws",str(attempt["draws"]),"--seed",str(attempt["seed"]),"--out",attempt["path"]])
        save()
    if "metrics_job" not in attempt:
        attempt["metrics_job"] = submit(["sbatch","--dependency=afterok:"+attempt["sample_job"],
            str(B / "scripts/high_throughput_metrics.sbatch"),attempt["path"]])
        save()

save()
while time.time() < watch_end:
    try:
        if state["status"] == "waiting_frozen_selection":
            main = read(audit / "state.json")
            selection = main.get("gpu_selection",{})
            if selection.get("status") == "no_passing_pilot":
                state["status"] = "no_frozen_winner"
                save()
                break
            if selection.get("winner"):
                winner = selection["winner"]
                attempt = next(a for a in reversed(main["methods"][winner]["attempts"]) if a.get("status") == "passed")
                state.update(status="pair_pending",thin=500 if winner == "gpu500" else 1000,
                             selected_draws=attempt["draws"],reference=attempt["path"],frozen_winner=winner)
                save()
        if state["status"] == "pair_pending":
            if "pair_job" not in state and time.time() < launch_end:
                state["pair_job"] = submit(["sbatch",str(B / "scripts/high_throughput_prefault.sbatch"),"pair",
                    "--reference",state["reference"],"--thin",str(state["thin"]),"--draws",str(state["selected_draws"]),"--seed","101001"])
                save()
            if "pair_job" in state:
                pair_state = status(state["pair_job"])
                log("Pair",state["pair_job"],pair_state)
                pair_path = B / f"results/high_throughput/iJO1366/prefault512_pair_thin{state['thin']}/seed101001/summary.json"
                if pair_state in terminal or pair_state == "COMPLETED":
                    pair = read(pair_path) if pair_path.exists() else dict(status="failed_job",slurm_state=pair_state)
                    state["pair_result"] = pair
                    wins = pair.get("status") == "passed" and pair.get("prefault_wins")
                    decision = dict(choice="prefault8" if wins else "frozen",pair_job=state["pair_job"],
                                    pair_summary=str(pair_path),thin=state["thin"],draws=state["selected_draws"],
                                    note="Distinct experimental build; guarded byte-identical diagnostics reuse for paired seed only. No frozen/experimental timing pooling.")
                    write(audit / "prefault-decision.json",decision)
                    if not wins:
                        state["status"] = "no_prefault_gain"
                        save()
                        break
                    on = next(row for row in pair["rows"] if row["host_prefault_threads"] == 8)
                    state["attempts"] = [dict(seed=101001,draws=state["selected_draws"],path=on["run_dir"],
                        sample_job=state["pair_job"],metrics_job=state["pair_job"],status="passed",quality=on["quality"]),
                        dict(seed=202001,draws=state["selected_draws"],path=str(case_path(202001,state["selected_draws"])),status="pending")]
                    state["status"] = "active"
                    save()
        if state["status"] == "active":
            attempt = state["attempts"][-1]
            if ("sample_job" not in attempt or "metrics_job" not in attempt) and time.time() >= launch_end:
                state["status"] = "launch_deadline_partial"
                save()
                break
            ensure_jobs(attempt)
            sampling,diagnostic = status(attempt["sample_job"]),status(attempt["metrics_job"])
            attempt.update(sampling_state=sampling,metrics_state=diagnostic)
            log("Replicate",attempt["seed"],attempt["draws"],sampling,diagnostic)
            if sampling in terminal or diagnostic in terminal:
                state["status"] = attempt["status"] = "failed_job"
                if sampling in terminal:
                    subprocess.run(["scancel",attempt["metrics_job"]],check=False)
                save()
                break
            quality_path = Path(attempt["path"]) / "quality.json"
            if diagnostic == "COMPLETED" and quality_path.exists():
                quality = read(quality_path)
                attempt["quality"] = quality
                if quality["status"] == "passed":
                    attempt["status"] = "passed"
                    if state.get("stop_after_current"):
                        state["status"] = "superseded_by_coordinate_after_current"
                        save()
                        break
                    if attempt["seed"] == 303001:
                        state["status"] = "complete"
                        save()
                        break
                    next_seed,next_draws = 303001,attempt["draws"]
                elif quality["status"] in ("pilot_budget_failed_rhat_or_bounds","failed_quality") and attempt["draws"] < 5000:
                    attempt["status"] = "budget_failed"
                    next_seed,next_draws = attempt["seed"],5000
                else:
                    state["status"] = attempt["status"] = "quality_cap"
                    save()
                    break
                state["attempts"].append(dict(seed=next_seed,draws=next_draws,path=str(case_path(next_seed,next_draws)),status="pending"))
                save()
        save()
        subprocess.run([sys.executable,str(B / "scripts/summarize_high_throughput.py")],check=False)
    except Exception as error:
        log("Retryable prefault controller error",repr(error))
        state["last_error"] = repr(error)
        save()
    time.sleep(30)
save()
subprocess.run([sys.executable,str(B / "scripts/summarize_high_throughput.py")],check=False)
log("Prefault controller stopped",state["status"])
