"""Summarize complete selected-budget thinning groups without pilot timing reuse."""
import csv
import json
from pathlib import Path
import statistics

B = Path(__file__).resolve().parents[1]
base = B / "results/gpu_thinning/iJO1366"
baseline = [json.loads((B / f"results/quality/iJO1366/naja512/seed{s}/quality.json").read_text())["measured"]["wall_seconds"] for s in (101,202,303)]
cpu = json.loads((B / "results/native_cpu/iJO1366/physical16_thin2000/summary.json").read_text())["measured_process_seconds_median"]
groups = []
for thin in (200,500,1000):
    rows = []
    for seed in (101,202,303):
        quality = base / f"thin{thin}/seed{seed}/quality.json"
        if not quality.exists():
            rows.append(dict(seed=seed, status="pending"))
            continue
        q = json.loads(quality.read_text())
        row = dict(seed=seed, status=q["status"], pilot_process_seconds=q.get("pilot_process_seconds"),
                   pilot_and_selection_seconds=q.get("pilot_and_selection_seconds"), selected_draws=q.get("selected_draws"))
        row["preserved_prior_attempts"] = [str(p) for p in quality.parent.glob("prior-attempt-*.json")]
        precheck = quality.parent / "pilot/rhat-precheck.json"
        if precheck.exists():
            checks = json.loads(precheck.read_text())["prefixes"]
            row["pilot_rhat_precheck_seconds"] = sum(c["precheck_wall_seconds"] for c in checks)
        pilot_metrics = quality.parent / "pilot/metrics.json"
        if pilot_metrics.exists():
            row["pilot_full_metrics_seconds"] = json.loads(pilot_metrics.read_text()).get("metrics_wall_seconds")
        if "measured" in q:
            measured = q["measured"]
            final = Path(measured["run_dir"])
            run = json.loads((final / "run.json").read_text())
            metrics = json.loads((final / "metrics.json").read_text())
            assert run["profile"]["config"]["thinning"] == thin
            assert run["profile"]["config"]["tpb_ss"] == 512
            assert run["seed"] == seed and run["chains"] == 16 and run["warmup"] == 1000
            assert metrics["schema_version"] == 2 and metrics["diagnostic_mask_status"] == "model_lp_verified"
            row.update(measured)
            phase = run["profile"]["timing"]["gpu_sampling_s"]
            transitions = run["chains"]*(run["draws"]+run["warmup"])*thin
            row.update(sampler_phase_seconds=phase,pooled_bulk_ess_per_sampler_phase_second=metrics["bulk_ess_median"]/phase,
                       transitions_including_warmup=transitions,pooled_bulk_ess_per_total_transition=metrics["bulk_ess_median"]/transitions)
        rows.append(row)
    complete = all(r["status"] == "complete" and r.get("quality_met") for r in rows)
    group = dict(thinning=thin, tpb=512, status="complete" if complete else "incomplete", rows=rows)
    if complete:
        median = statistics.median(r["wall_seconds"] for r in rows)
        group.update(measured_process_seconds_median=median,
                     sampler_phase_seconds_median=statistics.median(r["sampler_phase_seconds"] for r in rows),
                     pooled_bulk_ess_per_sampler_phase_second_median=statistics.median(r["pooled_bulk_ess_per_sampler_phase_second"] for r in rows),
                     baseline_auto96_process_seconds_median=statistics.median(baseline),
                     speedup_vs_gpu_auto96=statistics.median(baseline)/median,
                     physical_native_cpu_process_seconds_median=cpu,
                     speedup_vs_physical_native_cpu=cpu/median)
    groups.append(group)
summary = dict(model="iJO1366", groups=groups,
               primary_scope="Sampling phase and ESS/s. GPU gpu_sampling_s includes sampler allocation/slackGEMM/RNG initialization, not isolated CHRRkernel time. Existing process-speedup fields are secondary operational measurements.",
               timing_note="Each final is a fresh exact selected-budget process. Pilots, prefix search and diagnostics are excluded and recorded separately. Resume selection elapsed excludes the preserved failed attempt and reused pilot/precheck work; component costs remain recorded. Warmup1000 retained draws means warmup moves change with thinning. Selection is empirical on three seeds; no general optimality claim.")
(base / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
fields = ["thinning", "seed", "status", "selected_draws", "wall_seconds", "split_rhat_median",
          "bulk_ess_median", "quality_met", "sample_bound_violation_fraction", "pilot_process_seconds",
          "pilot_rhat_precheck_seconds", "pilot_full_metrics_seconds", "run_dir"]
with (base / "table.csv").open("w") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for group in groups:
        for row in group["rows"]:
            writer.writerow(dict(thinning=group["thinning"], **row))
print(json.dumps(summary, indent=2))
