"""Publish only validated512-chain pilot/replicate measurements."""
import json
from pathlib import Path
import statistics

B = Path(__file__).resolve().parents[1]
state_path = B / "audit/high-throughput/state.json"
state = json.loads(state_path.read_text())
full64_path = B / "audit/high-throughput/full64-state.json"
if full64_path.exists():
    full64 = json.loads(full64_path.read_text())
    state["methods"]["cpu"] = full64
prefault_path = B / "audit/high-throughput/prefault-state.json"
prefault = json.loads(prefault_path.read_text()) if prefault_path.exists() else {}
if prefault.get("attempts"):
    state["methods"]["gpu_prefault"] = prefault
coordinate_path = B / "audit/high-throughput/coordinate-state.json"
coordinate = json.loads(coordinate_path.read_text()) if coordinate_path.exists() else {}
if coordinate.get("attempts"):
    state["methods"]["gpu_coordinate"] = coordinate
required = [101001,202001,303001]
groups = {}
for method, group in state["methods"].items():
    passing = {}
    for attempt in group["attempts"]:
        if attempt.get("status") != "passed":
            continue
        path = Path(attempt["path"])
        q = json.loads((path / "quality.json").read_text())
        m = json.loads((path / "metrics.json").read_text())
        run = json.loads((path / "run.json").read_text())
        assert q["status"] == "passed" and run["status"] == "complete"
        assert m["schema_version"] == 2 and m["diagnostic_mask_status"] == "model_lp_verified"
        assert m["split_rhat_median"] <= 1.01 and m["bulk_ess_median"] >= 100
        assert m["bounds_tolerance"] == 1e-7 and m["sample_bound_violation_fraction"] == 0
        assert run["chains"] == 512 and run["seed"] in required
        if method.startswith("cpu"):
            affinity = json.loads((path / "physical_affinity.json").read_text())
            cores = 48 if method == "cpu48" else 64
            assert len(affinity["allowed_logical_cpus"]) == cores * 2 and len(affinity["selected_logical_cpus"]) == cores
            assert len({(r["socket"],r["core"]) for r in affinity["topology"] if r["cpu"] in affinity["selected_logical_cpus"]}) == cores
            manifest = json.loads((B / "audit/hopsy_native_build.json").read_text())
            assert run["hopsy_module_sha256"] == manifest["module_sha256"]
            assert run["workers"] == cores and run["thinning"] == 2000
        else:
            config = run["profile"]["config"]
            thin = prefault["thin"] if method == "gpu_prefault" else (500 if method in ("gpu500","gpu_coordinate") else 1000)
            assert config["thinning"] == thin and config["tpb_ss"] == 128
            expected_sha = ("51854688114677b14de3c110c262d210281de1f351f1ae6df6be994c2cc67e82" if method == "gpu_coordinate" else
                            "a656a1f93489af051acd540d08369092567dd0673af2b0e1ac922dc58a910603" if method == "gpu_prefault" else "c7baf4f0ed282b936ac0dbac4b896bfbfc040eb06da6f84aff6605fddf4b0fc6")
            assert run["exe_sha256"] == expected_sha
            if method in ("gpu_prefault","gpu_coordinate"):
                assert run["host_prefault_threads"] == 8
            assert run["preflight"]["native_matrix_elements"] < 2**31
        phase = run["timings"]["sample_seconds"] if method.startswith("cpu") else run["profile"]["timing"]["gpu_sampling_s"]
        thin = run["thinning"] if method.startswith("cpu") else run["profile"]["config"]["thinning"]
        transitions = run["chains"]*(run["draws"]+run["warmup"])*thin
        derived = dict(q,run_dir=str(path),sampler_phase_seconds=phase,
                       pooled_bulk_ess_per_sampler_phase_second=m["bulk_ess_median"]/phase,
                       transitions_including_warmup=transitions,pooled_bulk_ess_per_total_transition=m["bulk_ess_median"]/transitions)
        timing_path = path / "sampling_math_timing.json"
        if timing_path.exists():
            timing = json.loads(timing_path.read_text())
            derived.update(worker_draw_critical_path_seconds=timing["worker_draw_critical_path_seconds"],
                           total_worker_draw_thread_cpu_seconds=timing["total_worker_draw_thread_cpu_seconds"],
                           pooled_bulk_ess_per_worker_draw_second=m["bulk_ess_median"]/timing["worker_draw_critical_path_seconds"])
        passing[q["seed"]] = derived
    rows = [passing[seed] for seed in required if seed in passing]
    summary = dict(status="complete" if len(rows) == 3 else "partial", controller_status=group["status"],
                   required_seeds=required, passing_seeds=sorted(passing), rows=rows,
                   attempts=group["attempts"])
    if method == "cpu48" and rows:
        summary["status"] = "one_seed_preliminary"
    if rows:
        summary["process_seconds_median"] = statistics.median(r["process_seconds"] for r in rows)
        summary["pooled_bulk_ess_median_per_process_second_median"] = statistics.median(r["pooled_bulk_ess_median_per_process_second"] for r in rows)
        summary["draws_per_chain_by_seed"] = {str(r["seed"]):r["draws_per_chain"] for r in rows}
        summary["sampler_phase_seconds_median"] = statistics.median(r["sampler_phase_seconds"] for r in rows)
        summary["pooled_bulk_ess_per_sampler_phase_second_median"] = statistics.median(r["pooled_bulk_ess_per_sampler_phase_second"] for r in rows)
        if all("worker_draw_critical_path_seconds" in r for r in rows):
            summary["worker_draw_critical_path_seconds_median"] = statistics.median(r["worker_draw_critical_path_seconds"] for r in rows)
            summary["pooled_bulk_ess_per_worker_draw_second_median"] = statistics.median(r["pooled_bulk_ess_per_worker_draw_second"] for r in rows)
    groups[method] = summary
result = dict(model="iJO1366", chains=512, groups=groups,
              note="Best-use-case throughput experiment. Disjoint replicate seed ranges. GPU/CPU retain independently selected, potentially different draw counts; quality target is shared. ESS/s is median reaction pooled rank-normalized bulkESS divided by measured sampling process wall. ESS/512 is a pooled equivalent per-chain average, not independent per-chain estimates. Diagnostics, pilot selection and queue costs excluded and separately logged. Partial groups support no three-seed comparison.")
result["gpu_selection"] = state.get("gpu_selection")
result["prefault_selection_status"] = prefault.get("status")
result["prefault_pair"] = prefault.get("pair_result")
result["primary_timing_scope"] = "Samplingphase and ESS/s lead. CPUinclusivehopsy.sample includes IPC; workerdrawsidecar (whenpresent) excludes IPC/bookkeeping but includes binding/state-copy. GPU samplerphase includes allocation/slackGEMM/RNG init; not pureCHRRkernel. Process/output timings secondary, separate. Totaltransitions include warmup."
complete_gpu = {key:value for key,value in groups.items() if key.startswith("gpu") and value["status"] == "complete"}
if groups["cpu"]["status"] == "complete" and complete_gpu:
    gpu_key,gpu_group = min(complete_gpu.items(),key=lambda item:item[1]["sampler_phase_seconds_median"])
    result["complete_gpu_variant"] = gpu_key
    result["gpu_vs_cpu_ess_per_second_ratio"] = gpu_group["pooled_bulk_ess_median_per_process_second_median"] / groups["cpu"]["pooled_bulk_ess_median_per_process_second_median"]
    result["cpu_vs_gpu_process_seconds_ratio"] = groups["cpu"]["process_seconds_median"] / gpu_group["process_seconds_median"]
    result["cpu_vs_gpu_sampler_phase_ratio"] = groups["cpu"]["sampler_phase_seconds_median"] / gpu_group["sampler_phase_seconds_median"]
    result["gpu_vs_cpu_ess_per_sampler_phase_second_ratio"] = gpu_group["pooled_bulk_ess_per_sampler_phase_second_median"] / groups["cpu"]["pooled_bulk_ess_per_sampler_phase_second_median"]
    if "worker_draw_critical_path_seconds_median" in groups["cpu"]:
        result["cpu_worker_draw_vs_gpu_sampler_phase_ratio"] = groups["cpu"]["worker_draw_critical_path_seconds_median"] / gpu_group["sampler_phase_seconds_median"]
preliminary = groups.get("cpu48",{}).get("rows",[])
first_gpu = [(key,row) for key,group in groups.items() if key.startswith("gpu") for row in group["rows"] if row["seed"] == 101001]
if preliminary and first_gpu:
    gpu_key,row = min(first_gpu,key=lambda item:item[1]["sampler_phase_seconds"])
    cpu_row = preliminary[0]
    result["one_seed_physical48_preliminary"] = dict(gpu_variant=gpu_key,seed=101001,
        cpu_process_seconds=cpu_row["process_seconds"],gpu_process_seconds=row["process_seconds"],
        cpu_vs_gpu_process_seconds_ratio=cpu_row["process_seconds"] / row["process_seconds"],
        gpu_vs_cpu_pooled_ess_per_second_ratio=row["pooled_bulk_ess_median_per_process_second"] / cpu_row["pooled_bulk_ess_median_per_process_second"],
        cpu_sampler_phase_seconds=cpu_row["sampler_phase_seconds"],gpu_sampler_phase_seconds=row["sampler_phase_seconds"],
        cpu_vs_gpu_sampler_phase_ratio=cpu_row["sampler_phase_seconds"] / row["sampler_phase_seconds"],
        gpu_vs_cpu_ess_per_sampler_phase_second_ratio=row["pooled_bulk_ess_per_sampler_phase_second"] / cpu_row["pooled_bulk_ess_per_sampler_phase_second"],
        note="One seed and48physicalCPU cores only; cannot substitute for the queued64physicalCPU comparison.")
out = B / "results/high_throughput/iJO1366/summary.json"
out.parent.mkdir(parents=True,exist_ok=True)
temp = out.with_suffix(".tmp")
temp.write_text(json.dumps(result,indent=2)+"\n")
temp.replace(out)
print({k:(v["status"],v["passing_seeds"]) for k,v in groups.items()})
