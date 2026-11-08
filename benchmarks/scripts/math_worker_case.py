"""Run one opt-in16-physical-core case and prove unchanged sample bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

B = Path(__file__).resolve().parents[1]

def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",type=int,required=True)
    args = parser.parse_args()
    task = json.loads((B / "audit/math_worker_tasks.json").read_text())[args.task]
    command = [str(B / "env-cpu-native/bin/python"),str(B / "scripts/native_physical_pilot.py"),
        "--worker-timing","--model",task["model"],"--build","native","--label",task["label"],
        "--seed",str(task["seed"]),"--draws",str(task["draws"]),"--thinning",task["thinning"]]
    subprocess.run(command,check=True)
    out = B / f"results/native_cpu/{task['model']}/{task['label']}/seed{task['seed']}"
    current,reference = sha(out / "samples.npy"),sha(Path(task["reference"]) / "samples.npy")
    result = dict(status="passed" if current == reference else "failed",bitwise_equal=current == reference,
                  samples_sha256=current,reference_samples_sha256=reference,reference_run=task["reference"],
                  instrumented_run=str(out),seed=task["seed"],model=task["model"])
    (out / "worker_timing_parity.json").write_text(json.dumps(result,indent=2)+"\n")
    assert current == reference
    subprocess.run([str(B / "env-cpu-native/bin/python"),str(B / "scripts/summarize_worker_math.py")],check=True)

if __name__ == "__main__":
    main()
