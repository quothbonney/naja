"""Bounded distinct three-seed coordinate-specialized512-chain group."""
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
lock = (audit / "coordinate-controller.lock").open("w")
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(audit / "coordinate-controller.pid").write_text(str(os.getpid())+"\n")
state_path = audit / "coordinate-state.json"
state = json.loads(state_path.read_text()) if state_path.exists() else dict(status="active",thin=500,selected_draws=5000,
    attempts=[dict(seed=101001,draws=5000,path=str(B / "results/coordinate_kernel/paired/chains512/coordinate"),
                   sample_job="494",metrics_job="504",status="pending")])
launch_end = datetime(2026,11,8,10,30,tzinfo=timezone.utc).timestamp()
watch_end = datetime(2026,11,8,11,tzinfo=timezone.utc).timestamp()
terminal = {"FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","DEADLINE"}

def save():
    temp = state_path.with_suffix(".tmp")
    temp.write_text(json.dumps(state,indent=2)+"\n");os.replace(temp,state_path)

def log(*values):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),*values,flush=True)

def status(job):
    result = subprocess.run(["sacct","-n","-X","-j",job,"--format=State","-P"],capture_output=True,text=True)
    values = [line.strip().split("|")[0].split()[0].rstrip("+") for line in result.stdout.splitlines() if line.strip()]
    return values[0] if values else "UNKNOWN"

def submit(command):
    result = subprocess.run(command,cwd=B.parent,capture_output=True,text=True,check=True)
    log(result.stdout.strip());return result.stdout.split()[-1]

save()
while time.time() < watch_end and state["status"] == "active":
    try:
        attempt = state["attempts"][-1]
        if "sample_job" not in attempt:
            if time.time() >= launch_end:
                state["status"] = "launch_deadline_partial";save();break
            attempt["sample_job"] = submit(["sbatch","--gres=gpu:1",str(B / "scripts/coordinate512.sbatch"),"sample",
                                            "--seed",str(attempt["seed"]),"--out",attempt["path"]])
            save()
        if "metrics_job" not in attempt:
            attempt["metrics_job"] = submit(["sbatch","--dependency=afterok:"+attempt["sample_job"],
                str(B / "scripts/high_throughput_metrics.sbatch"),attempt["path"]]);save()
        sample,diagnostic = status(attempt["sample_job"]),status(attempt["metrics_job"])
        attempt.update(sampling_state=sample,metrics_state=diagnostic)
        log(attempt["seed"],sample,diagnostic)
        if sample in terminal or diagnostic in terminal:
            state["status"] = attempt["status"] = "failed_job"
            if sample in terminal:subprocess.run(["scancel",attempt["metrics_job"]],check=False)
            save();break
        quality_path = Path(attempt["path"])/"quality.json"
        if diagnostic == "COMPLETED" and quality_path.exists():
            quality = json.loads(quality_path.read_text());attempt["quality"] = quality
            if quality["status"] != "passed":
                state["status"] = attempt["status"] = "quality_cap";save();break
            attempt["status"] = "passed"
            if attempt["seed"] == 303001:
                state["status"] = "complete";save();break
            next_seed = 202001 if attempt["seed"] == 101001 else 303001
            state["attempts"].append(dict(seed=next_seed,draws=5000,status="pending",
                path=str(B / f"results/high_throughput/iJO1366/naja512_coordinate_tpb128_thin500_prefault8/seed{next_seed}")))
            save()
        subprocess.run([sys.executable,str(B / "scripts/summarize_high_throughput.py")],check=False)
    except Exception as error:
        state["last_error"] = repr(error);save();log("Retryable controller error",repr(error))
    time.sleep(30)
save()
subprocess.run([sys.executable,str(B / "scripts/summarize_high_throughput.py")],check=False)
log("Coordinate controller stopped",state["status"])
