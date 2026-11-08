"""Persistent stdlib-only bounded Recon3D readiness watcher."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

B = Path(__file__).resolve().parents[1]
audit = B / "audit/recon3d-completion"
audit.mkdir(exist_ok=True)
lock = (audit / "watcher.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
(audit / "watcher.pid").write_text(str(os.getpid()) + "\n")
state_path = audit / "watcher-state.json"
state = json.loads(state_path.read_text()) if state_path.exists() else {"submitted": {}}
deadline = datetime(2026, 11, 8, 11, tzinfo=timezone.utc).timestamp()
sources = [("original", "150_6", B / "data/models/Recon3D"),
           ("private", "401", B / "data/models/Recon3D_zero_fva_faces_fast")]
while time.time() < deadline:
    marker = B / "data/models/Recon3D/validation.json"
    if marker.exists() and json.loads(marker.read_text()).get("status") == "passed":
        print("Canonical validated; watcher complete", flush=True)
        break
    for kind, job, source in sources:
        if kind in state["submitted"]:
            validator_job = state["submitted"][kind]["submission"].split()[-1]
            validation_state = subprocess.run(["sacct", "-n", "-X", "-j", validator_job,
                                                "--format=JobIDRaw,State,ExitCode", "-P"],
                                               capture_output=True, text=True)
            print("Validation attempt status", kind, validation_state.stdout.strip(), flush=True)
            continue
        result = subprocess.run(["sacct", "-n", "-X", "-j", job, "--format=JobIDRaw,State", "-P"],
                                capture_output=True, text=True)
        if result.returncode:
            print("Accounting query failed; retry next poll", result.stderr, flush=True)
            continue
        print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), kind, result.stdout.strip(), flush=True)
        completed = any(line.split("|")[1] == "COMPLETED" for line in result.stdout.splitlines() if "|" in line)
        if (completed and (source / "meta.json").exists() and (source / "rounding/polytope.npz").exists()
                and time.time() < deadline - 1800):
            submitted = subprocess.run(["sbatch", str(B / "scripts/recon3d_completion.sbatch"), kind],
                                       cwd=B.parent, capture_output=True, text=True)
            if submitted.returncode:
                print("Validation submission failed; retry next poll", submitted.stderr, flush=True)
                continue
            state["submitted"][kind] = {"submission": submitted.stdout.strip(), "source_job": job,
                                         "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            temp = state_path.with_suffix(".tmp")
            temp.write_text(json.dumps(state, indent=2) + "\n")
            os.replace(temp, state_path)
            print("Submitted validation", state["submitted"][kind], flush=True)
    time.sleep(40)
else:
    print("11:00UTC hard deadline reached; watcher stopped", flush=True)
