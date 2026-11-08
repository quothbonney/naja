"""One-seed native pilot with recorded, physically distinct worker cores."""
import json
import os
import argparse
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--model", default="iJO1366")
parser.add_argument("--build", choices=["native", "wheel"], default="native")
parser.add_argument("--draws", type=int, default=10000)
parser.add_argument("--thinning", default="2000")
parser.add_argument("--label", default="physical16_thin2000")
parser.add_argument("--selected-task", type=int)
parser.add_argument("--worker-timing",action="store_true")
parser.add_argument("--reuse-diagnostics-from",type=Path)
args = parser.parse_args()
task = None
if args.selected_task is not None:
    task = json.loads((root / "audit/physical_selected_tasks.json").read_text())[args.selected_task]
    for key in ("model", "build", "draws", "thinning", "label", "seed"):
        setattr(args, key, task[key])
out = root / f"results/native_cpu/{args.model}/{args.label}/seed{args.seed}"
out.mkdir(parents=True, exist_ok=False)
allowed = sorted(os.sched_getaffinity(0))
topology = []
for cpu in allowed:
    base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    topology.append({"cpu": cpu,
                     "socket": int((base / "physical_package_id").read_text()),
                     "core": int((base / "core_id").read_text()),
                     "siblings": (base / "thread_siblings_list").read_text().strip()})
unique = {}
for row in topology:
    unique.setdefault((row["socket"], row["core"]), row["cpu"])
by_socket = {}
for (socket, core), cpu in sorted(unique.items()):
    by_socket.setdefault(socket, []).append(cpu)
selected = []
while len(selected) < 16 and any(by_socket.values()):
    for socket in sorted(by_socket):
        if by_socket[socket] and len(selected) < 16:
            selected.append(by_socket[socket].pop(0))
metadata = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "reserved_logical_cpus": int(os.environ["SLURM_CPUS_PER_TASK"]),
            "allowed_logical_cpus": allowed,
            "allowed_physical_core_count": len(unique), "topology": topology,
            "selected_logical_cpus": sorted(selected),
            "selected_physical_core_count": len(selected),
            "selection": "one hardware thread per unique socket/core, round robin sockets",
            "pilot_only": True, "status": "affinity_recorded"}
metadata["opt_in_worker_draw_timing"] = args.worker_timing
if task is not None:
    metadata["selected_budget_provenance"] = task
path = out / "physical_affinity.json"
path.write_text(json.dumps(metadata, indent=2) + "\n")
assert len(allowed) >= 32, f"Expected >=32 reserved logical CPUs, got {allowed}"
assert len(selected) == 16, f"Only {len(selected)} distinct physical cores available"
os.sched_setaffinity(0, selected)
metadata["actual_python_affinity"] = sorted(os.sched_getaffinity(0))
metadata["status"] = "running"
path.write_text(json.dumps(metadata, indent=2) + "\n")
python = str(root / ("env-cpu-native/bin/python" if args.build == "native" else "env-cpu/bin/python"))
if args.build == "native":
    os.environ["HOPSY_CXXFLAGS"] = "-O3 -march=native -mtune=native"
    os.environ["HOPSY_BUILD_MANIFEST"] = str(root / "audit/hopsy_native_build.json")
else:
    os.environ["HOPSY_CXXFLAGS"] = "wheel defaults"
    os.environ.pop("HOPSY_BUILD_MANIFEST", None)
baseline_script = root / ("scripts/timed_cpu_baseline.py" if args.worker_timing else "cpu_baseline.py")
subprocess.run([python, str(baseline_script), "sample", "--model",
                str(root / "data/models" / args.model), "--out", str(out), "--seed", str(args.seed),
                "--chains", "16", "--draws", str(args.draws), "--warmup", "1000",
                "--thinning", str(args.thinning), "--workers", "16", "--chunk", str(args.draws + 1000),
                "--build-variant", args.build], check=True)
metric_command = [python, str(root / "metrics.py"), "--samples", str(out / "samples.npy"),
                  "--model-dir", str(root / "data/models" / args.model), "--output", str(out)]
if args.worker_timing:
    full_seconds = json.loads((out / "sampling_math_timing.json").read_text())["instrumented_wrapper_process_seconds"]
    metric_command += ["--wall-seconds",str(full_seconds)]
reused = False
if args.reuse_diagnostics_from is not None:
    wall = full_seconds if args.worker_timing else json.loads((out/'run.json').read_text())['process_wall_seconds']
    reuse_command = [python,str(root/'scripts/strict_diagnostic_reuse.py'),'--samples',str(out/'samples.npy'),
        '--model',str(root/'data/models'/args.model),'--out',str(out),'--reference',str(args.reuse_diagnostics_from),'--wall',str(wall)]
    reused = subprocess.run(reuse_command,check=False).returncode == 0
if not reused:
    subprocess.run(metric_command,check=True)
metadata["status"] = "completed"
path.write_text(json.dumps(metadata, indent=2) + "\n")
