"""Summarize byte-verified opt-in worker draw timings separately from phases."""
import csv
import fcntl
import json
from pathlib import Path
import statistics

B = Path(__file__).resolve().parents[1]

def generate():
    lock=(B/'results/.worker_math_benchmarks.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    specs = [("iAB_RBC_283","physical16_mathinstrumented_native_jade"),("iJO1366","physical16_mathinstrumented_thin2000"),("iML1515","physical16_mathinstrumented_native_jade")]
    groups = []
    for model,label in specs:
        parent = B / f"results/native_cpu/{model}/{label}"
        rows = []
        for seed in (101,202,303):
            path = parent / f"seed{seed}"
            proof = path / "worker_timing_parity.json"
            if not proof.exists():
                continue
            parity = json.loads(proof.read_text())
            if parity.get("status") != "passed" or not parity.get("bitwise_equal"):
                continue
            run = json.loads((path / "run.json").read_text())
            m = json.loads((path / "metrics.json").read_text())
            t = json.loads((path / "sampling_math_timing.json").read_text())
            assert t["status"] == "complete" and t["worker_count"] == 16
            assert m["schema_version"] == 2 and m["diagnostic_mask_status"] == "model_lp_verified"
            passed = m["split_rhat_median"] <= 1.01 and m["bulk_ess_median"] >= 100 and m["sample_bound_violation_fraction"] == 0
            total = run["chains"]*(run["draws"]+run["warmup"])*run["thinning"]
            rows.append(dict(seed=seed,model=model,quality_met=passed,run_dir=str(path),
                draws=run["draws"],thinning=run["thinning"],physical_cores=16,reserved_logical_cpus=32,
                worker_draw_critical_path_seconds=t["worker_draw_critical_path_seconds"],
                worker_draw_cpu_critical_path_seconds=t["worker_draw_cpu_critical_path_seconds"],
                total_worker_draw_thread_cpu_seconds=t["total_worker_draw_thread_cpu_seconds"],
                hopsy_sample_phase_inclusive_seconds=t["hopsy_sample_phase_inclusive_seconds"],
                pooled_bulk_ess_median=m["bulk_ess_median"],
                pooled_bulk_ess_per_worker_draw_second=m["bulk_ess_median"]/t["worker_draw_critical_path_seconds"],
                transitions_including_warmup=total,pooled_bulk_ess_per_total_transition=m["bulk_ess_median"]/total,
                split_rhat_median=m["split_rhat_median"],bounds_tolerance=m["bounds_tolerance"],sample_bound_violation_fraction=m["sample_bound_violation_fraction"],
                geometry_sha256=m.get("diagnostic_mask_geometry_sha256"),
                instrumentation_sha256=t["proof"]["instrumentation_module_sha256"],bitwise_same_as_reference=True))
        complete = len(rows) == 3 and all(r["quality_met"] for r in rows)
        group = dict(model=model,variant=label,status="complete" if complete else "partial",rows=rows,
                     critical_path_scope="Maximum per-worker accumulated C++MarkovChain.draw-call elapsed time. Includes Python binding/state-copy and OSdescheduling; excludes pool startup, IPC, Python list bookkeeping and final array assembly. Raw timer overhead not subtracted; not pure arithmetic-only.")
        if complete:
            group.update(worker_draw_critical_path_seconds_median=statistics.median(r["worker_draw_critical_path_seconds"] for r in rows),
                         pooled_bulk_ess_per_worker_draw_second_median=statistics.median(r["pooled_bulk_ess_per_worker_draw_second"] for r in rows),
                         total_worker_draw_thread_cpu_seconds_median=statistics.median(r["total_worker_draw_thread_cpu_seconds"] for r in rows))
        parent.mkdir(parents=True,exist_ok=True)
        (parent / "worker_math_summary.json").write_text(json.dumps(group,indent=2)+"\n")
        groups.append(group)
    result = dict(schema_version=1,groups=groups,note="Separate instrumented variant. Full retained-output SHA parity required for every case. Three-seed groups only; one-seed results remain partial. GPU sampler phases include GPUinit; genuine CHRRkernel-only comparisons require matching trace evidence.")
    comparison_source = B / "results/SAMPLING_BENCHMARKS.json"
    comparisons = []
    if comparison_source.exists():
        baseline = json.loads(comparison_source.read_text())
        for group in groups:
            if group["status"] != "complete":
                continue
            gpu = next((g for g in baseline["fastest_complete_groups"] if g["model"] == group["model"] and g["method"] == "GPU"),None)
            if not gpu:
                continue
            for cpu_row,gpu_row in zip(group["rows"],gpu["rows"]):
                assert cpu_row["seed"] == gpu_row["seed"] and cpu_row["geometry_sha256"] == gpu_row["geometry_sha256"]
            comparisons.append(dict(model=group["model"],cpu_variant=group["variant"],gpu_variant=gpu["config"],
                cpu_worker_draw_critical_path_seconds_median=group["worker_draw_critical_path_seconds_median"],
                gpu_sampler_phase_seconds_median=gpu["sampler_phase_seconds_median"],
                ratio_cpu_worker_draw_over_gpu_sampler_phase=group["worker_draw_critical_path_seconds_median"]/gpu["sampler_phase_seconds_median"],
                cpu_pooled_ess_per_worker_draw_second_median=group["pooled_bulk_ess_per_worker_draw_second_median"],
                gpu_pooled_ess_per_sampler_phase_second_median=gpu["pooled_bulk_ess_per_sampler_phase_second_median"],
                median_paired_gpu_over_cpu_ess_rate=statistics.median(gr["pooled_bulk_ess_per_sampler_phase_second"]/cr["pooled_bulk_ess_per_worker_draw_second"] for cr,gr in zip(group["rows"],gpu["rows"])),
                scope="CPUdraw-call criticalpath includes binding/state-copy but excludes IPC/bookkeeping; GPU samplerphase includes allocation/slackGEMM/RNG init. These are explicit samplercompute proxies, not identical pure-arithmetic scope."))
    result["gpu_sampler_phase_comparisons"] = comparisons
    cap_path=B/'results/iml_time_to_target/summary.json'
    if cap_path.exists():result['budget_cap_studies']=[json.loads(cap_path.read_text())]
    target=B/'results/WORKER_MATH_BENCHMARKS.json';temp=target.with_suffix('.json.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(target)
    fields = ["model","seed","quality_met","row_scope","worker_draw_critical_path_seconds","worker_draw_cpu_critical_path_seconds","total_worker_draw_thread_cpu_seconds","hopsy_sample_phase_inclusive_seconds","pooled_bulk_ess_median","pooled_bulk_ess_per_worker_draw_second","transitions_including_warmup","pooled_bulk_ess_per_total_transition","run_dir"]
    csv_target=B/'results/WORKER_MATH_BENCHMARKS.csv';csv_temp=csv_target.with_suffix('.csv.tmp')
    with csv_temp.open("w") as handle:
        writer = csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore");writer.writeheader()
        for group in groups:
            writer.writerows(dict(row_scope='byte_verified_passing_timed_group',**row) for row in group['rows'])
        for study in result.get('budget_cap_studies',[]):
            for row in study['cpu_thin2000_fixed_cap']['rows']:
                writer.writerow(dict(model=study['model'],seed=row['seed'],quality_met=row['quality_met'],row_scope='fixed10000_cap_control_no_passing_timing_claim',run_dir=row['run_dir']))
    csv_temp.replace(csv_target)
    md=['# Worker draw timing comparisons','',result['note'],'','| Model | CPU worker draw critical path(s) | GPU sampler phase(s) | Time ratioCPU/GPU | Paired ESS/s ratioGPU/CPU |','|---|---:|---:|---:|---:|']
    for c in comparisons:md.append(f"| {c['model']} | {c['cpu_worker_draw_critical_path_seconds_median']:.6g} | {c['gpu_sampler_phase_seconds_median']:.6g} | {c['ratio_cpu_worker_draw_over_gpu_sampler_phase']:.3f} | {c['median_paired_gpu_over_cpu_ess_rate']:.3f} |")
    md += ['', 'CPU draw timing includes Python binding/state-copy and OS descheduling. GPU sampler phase includes allocation/slack GEMM/RNG initialization. Neither is labeled pure arithmetic-only; no IPC/output cost is hidden in the CPU draw counter.','']
    for study in result.get('budget_cap_studies',[]):
        c=study['cpu_thin2000_fixed_cap'];md.append(f"{study['model']} native thin2000 fixed10k control: {c['passing_count']}/{c['tested_count']} actually tested seeds pass; missing {c['missing_seeds'] or 'none'}. Cap failure does not mean eventual nonconvergence. This control is separate from timing comparisons.")
    md_target=B/'results/WORKER_MATH_BENCHMARKS.md';md_temp=md_target.with_suffix('.md.tmp');md_temp.write_text('\n'.join(md)+'\n');md_temp.replace(md_target)
    print([(g["model"],g["status"],[r["seed"] for r in g["rows"]]) for g in groups])
    return result

if __name__ == "__main__":
    generate()
