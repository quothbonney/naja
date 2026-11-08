"""Validate and summarize the distinct physical-core native CPU group."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parents[1]
group = root / "results/native_cpu/iJO1366/physical16_thin2000"
manifest = json.loads((root / "audit/hopsy_native_build.json").read_text())

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

rows = []
for seed in (101, 202, 303):
    path = group / f"seed{seed}"
    run = json.loads((path / "run.json").read_text())
    metrics = json.loads((path / "metrics.json").read_text())
    affinity = json.loads((path / "physical_affinity.json").read_text())
    assert affinity["status"] == "completed"
    assert run["hopsy_module_sha256"] == manifest["module_sha256"]
    assert run["hopsy_cxxflags"] == manifest["requested_flags"]
    assert (run["seed"], run["workers"], run["chains"], run["draws"], run["warmup"], run["thinning"]) == (seed, 16, 16, 10000, 1000, 2000)
    allowed = set(affinity["allowed_logical_cpus"])
    selected = set(affinity["actual_python_affinity"])
    topology = [r for r in affinity["topology"] if r["cpu"] in selected]
    assert len(selected) == len({(r["socket"], r["core"]) for r in topology}) == 16
    assert len(allowed) == affinity["reserved_logical_cpus"] == 32
    assert all(set(map(int, r["siblings"].split(","))) <= allowed for r in topology)
    old = root / f"results/quality/iJO1366/cpu_native2000/seed{seed}/final-budget10000/samples.npy"
    sample_sha = sha(path / "samples.npy")
    old_sha = sha(old)
    assert sample_sha == old_sha, f"Seed{seed}: differs from canonical same-seed native trajectory"
    passed = (metrics["split_rhat_median"] <= 1.01 and metrics["bulk_ess_median"] >= 100
              and metrics["sample_bound_violation_fraction"] == 0
              and metrics["bounds_tolerance"] == 1e-7)
    rows.append(dict(seed=seed, workers=16, physical_cores=16, reserved_logical_cpus=32,
                     sockets=",".join(map(str, sorted({r["socket"] for r in topology}))),
                     process_seconds=run["process_wall_seconds"],
                     sample_seconds=run["timings"]["sample_seconds"],
                     split_rhat_median=metrics["split_rhat_median"],
                     bulk_ess_median=metrics["bulk_ess_median"],
                     bound_violation_fraction=metrics["sample_bound_violation_fraction"],
                     bounds_tolerance=metrics["bounds_tolerance"], quality_met=passed,
                     samples_sha256=sample_sha, canonical_same_seed_sha256=old_sha,
                     bitwise_same_as_canonical=True, run_dir=str(path)))
summary = dict(status="complete" if all(r["quality_met"] for r in rows) else "quality_failed",
               model="iJO1366", variant="native_physical16_thin2000", seeds=[101,202,303],
               physical_cores=16, reserved_logical_cpus=32, workers=16,
               samples_space="reduced", build_module_sha256=manifest["module_sha256"],
               measured_process_seconds_median=statistics.median(r["process_seconds"] for r in rows),
               measured_sample_seconds_median=statistics.median(r["sample_seconds"] for r in rows),
               note="Three independent RNG seeds. Each exact budget was actually executed; diagnostics computed fresh. Single-socket placement per run; actual socket may differ. Earlier logical-affinity controls do not establish isolated placement causality.", rows=rows)
(group / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (group / "table.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(summary, indent=2))
