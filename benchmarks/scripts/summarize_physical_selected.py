"""Audit selected-budget physical runs without mixing groups or prior timings."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parents[1]
tasks = json.loads((root / "audit/physical_selected_tasks.json").read_text())

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

groups = {}
for task in tasks:
    directory = root / "results/native_cpu" / task["model"] / task["label"]
    rows = groups.setdefault(directory, [])
    path = directory / f"seed{task['seed']}"
    if not (path / "metrics.json").exists():
        continue
    affinity = json.loads((path / "physical_affinity.json").read_text())
    if affinity["status"] != "completed":
        continue
    run = json.loads((path / "run.json").read_text())
    prior = json.loads((Path(task["prior_run"]) / "run.json").read_text())
    metrics = json.loads((path / "metrics.json").read_text())
    assert metrics["schema_version"] == 2 and metrics["diagnostic_mask_status"] == "model_lp_verified"
    for field in ("seed", "workers", "chains", "draws", "warmup", "thinning"):
        assert run[field] == prior[field], (task, field)
    if "hopsy_module_sha256" in prior:
        assert run["hopsy_module_sha256"] == prior["hopsy_module_sha256"]
    else:
        assert task["build"] == "wheel" and run["versions"]["hopsy"] == prior["versions"]["hopsy"]
    assert run["build_variant"] == task["build"]
    assert run["draws"] == task["draws"] and run["thinning"] == task["expected_thinning"]
    allowed = set(affinity["allowed_logical_cpus"])
    selected = set(affinity["actual_python_affinity"])
    topo = [r for r in affinity["topology"] if r["cpu"] in selected]
    assert len(selected) == len({(r["socket"], r["core"]) for r in topo}) == 16
    assert len(allowed) == affinity["reserved_logical_cpus"] == 32
    assert all(set(map(int, r["siblings"].split(","))) <= allowed for r in topo)
    sample_sha = sha(path / "samples.npy")
    prior_sha = sha(Path(task["prior_run"]) / "samples.npy")
    assert sample_sha == prior_sha, f"Different trajectory for {task}"
    passed = (metrics["split_rhat_median"] <= 1.01 and metrics["bulk_ess_median"] >= 100
              and metrics["sample_bound_violation_fraction"] == 0 and metrics["bounds_tolerance"] == 1e-7)
    rows.append(dict(seed=task["seed"], model=task["model"], build=task["build"],
                     draws=run["draws"], thinning=run["thinning"], physical_cores=16,
                     reserved_logical_cpus=32, workers=16, process_seconds=run["process_wall_seconds"],
                     sample_seconds=run["timings"]["sample_seconds"],
                     split_rhat_median=metrics["split_rhat_median"], bulk_ess_median=metrics["bulk_ess_median"],
                     bound_violation_fraction=metrics["sample_bound_violation_fraction"],
                     quality_met=passed, samples_sha256=sample_sha, prior_same_seed_sha256=prior_sha,
                     bitwise_same_as_prior=True, bitwise_same_as_canonical=True,
                     run_dir=str(path), prior_run=task["prior_run"]))
for directory, rows in groups.items():
    directory.mkdir(parents=True, exist_ok=True)
    status = "complete" if len(rows) == 3 and all(r["quality_met"] for r in rows) else "partial"
    summary = dict(status=status, model=directory.parent.name, variant=directory.name,
                   completed_seeds=[r["seed"] for r in rows], required_seeds=[101,202,303],
                   physical_cores=16, reserved_logical_cpus=32, workers=16, rows=rows,
                   draws=rows[0]["draws"] if rows else None,
                   thinning=rows[0]["thinning"] if rows else None,
                   timing_note="Actual selected-budget process executions including warmup/startup/write. Fresh diagnostics; no prior timings reused.")
    if status == "complete":
        summary["measured_process_seconds_median"] = statistics.median(r["process_seconds"] for r in rows)
        summary["measured_sample_seconds_median"] = statistics.median(r["sample_seconds"] for r in rows)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if rows:
        with (directory / "table.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(directory.parent.name, directory.name, status, summary.get("measured_process_seconds_median"), summary["completed_seeds"])
