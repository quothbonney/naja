"""Validate a finished Recon3D bundle and publish it under an exclusive lock."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

B = Path(__file__).resolve().parents[1]
kind = sys.argv[1]
assert kind in ("original", "private")
canonical = B / "data/models/Recon3D"
source = canonical if kind == "original" else B / "data/models/Recon3D_zero_fva_faces_fast"
audit = B / "audit/recon3d-completion"
audit.mkdir(exist_ok=True)
lock = (audit / "publication.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX)

def passed(path):
    try:
        return json.loads(path.read_text()).get("status") == "passed"
    except (FileNotFoundError, json.JSONDecodeError):
        return False

def atomic_json(path, data):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temp, path)

def log(message):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), message, flush=True)

if passed(canonical / "validation.json"):
    log("Canonical already passed; publication unchanged")
    sys.exit(0)
assert (source / "meta.json").exists()
assert (source / "rounding/polytope.npz").exists()
selected = None
for tolerance, label in ((1e-7, "strict1e7"), (1e-6, "gate1e6")):
    result = audit / f"{kind}-{label}.json"
    command = [str(B / "env-cpu/bin/python"), str(B / "scripts/validate_preprocessed.py"),
               "--model", str(source), "--raw", str(B / "data/raw/Recon3D.json"),
               "--output", str(result), "--tolerance", str(tolerance)]
    log(f"Validating {kind} tolerance={tolerance}")
    completed = subprocess.run(command)
    if completed.returncode == 0 and passed(result):
        selected = json.loads(result.read_text())
        break
if selected is None:
    log("Both validation gates failed; no publication")
    sys.exit(1)

if kind == "private":
    # Stop every original writer before moving or replacing its directory.
    subprocess.run(["scancel", "150_6"], check=False)
    for attempt in range(60):
        jobs = subprocess.run(["squeue", "-h", "-u", str(os.getuid()), "-o", "%i"], capture_output=True, text=True, check=True)
        steps = subprocess.run(["squeue", "-h", "--steps", "-j", "150_6"], capture_output=True, text=True, check=True)
        if "150_6" not in jobs.stdout.split() and "150" not in jobs.stdout.split() and not steps.stdout.strip():
            # Slurm no longer lists the allocation or any step; confirm again.
            time.sleep(2)
            confirm = subprocess.run(["squeue", "-h", "--steps", "-j", "150_6"], capture_output=True, text=True, check=True)
            if not confirm.stdout.strip():
                break
        time.sleep(5)
    else:
        raise RuntimeError("Original steps did not stop; refusing publication")
    if passed(canonical / "validation.json"):
        raise RuntimeError("Canonical passed while waiting; refusing replacement")

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

bundle = {str(p.relative_to(source)): sha(p) for p in source.rglob("*")
          if p.is_file() and p.name != "validation.json"}
manifest = dict(published_from=str(source), kind=kind, source_job="150_6" if kind == "original" else "401",
                validator_job=os.environ.get("SLURM_JOB_ID"), bundle_sha256=bundle,
                raw_source_sha256=sha(B / "data/raw/Recon3D.json"),
                strict_pass=selected["tolerance"] == 1e-7,
                precision_caveat=None if selected["tolerance"] == 1e-7 else "Strict1e-7 validation failed; preprocessing accepted only at1e-6. Keep strict failed QC and validate result samples at declared tolerance.",
                published_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
if kind == "private":
    stage = canonical.with_name(f".Recon3D-publish-{os.environ.get('SLURM_JOB_ID', os.getpid())}")
    shutil.copytree(source, stage, ignore=shutil.ignore_patterns("validation.json"))
    for name, digest in bundle.items():
        assert sha(stage / name) == digest, name
    archive = canonical.with_name("Recon3D-original-partial-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    if canonical.exists():
        canonical.rename(archive)
    stage.rename(canonical)
    manifest["archived_original"] = str(archive)
selected["model"] = str(canonical)
selected["published_from"] = str(source)
selected["strict_1e_minus_7_pass"] = manifest["strict_pass"]
selected["precision_caveat"] = manifest["precision_caveat"]
atomic_json(canonical / "publication.json", manifest)
atomic_json(audit / "published.json", manifest)
# This readiness marker is deliberately the final bundle write.
atomic_json(canonical / "validation.json", selected)
log(f"Published {kind} with tolerance={selected['tolerance']}")
if kind == "original":
    subprocess.run(["scancel", "401"], check=False)
else:
    subprocess.run(["scancel", "202_8"], check=False)
    job = subprocess.run(["sbatch", "--array=8", "--output=" + str(B / "logs/diagnostic-mask-recon3d-%A_%a.log"),
                          str(B / "scripts/diagnostic_mask.sbatch")], cwd=B, capture_output=True, text=True, check=True)
    atomic_json(audit / "replacement-diagnostic-job.json", {"submission": job.stdout.strip()})
