"""One full-budget common diagnostic evaluation, separately bounded by Slurm."""
import argparse
import json
from pathlib import Path
import sys

B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B))
import metrics
from quality_diagnostics import rhat_precheck,save_identity,reuse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    path = args.run
    run = json.loads((path / "run.json").read_text())
    assert run["status"] == "complete" and run["chains"] == 512
    model = B / "data/models/iJO1366"
    worker_timing_path = path / "sampling_math_timing.json"
    worker_timing = json.loads(worker_timing_path.read_text()) if worker_timing_path.exists() else None
    wall_override = worker_timing["instrumented_wrapper_process_seconds"] if worker_timing else None
    cached = False
    protocol_path = path / "selected_budget_protocol.json"
    if run["method"] == "hopsy_chrr" and protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if run["seed"] == protocol["seed"] and run["draws"] == protocol["draws"]:
            cached = reuse(path / "samples.npy",model,path,wall_override or run["process_wall_seconds"],Path(protocol["source_run"]))
    if run["method"] == "hopsy_chrr" and run["draws"] == 20000:
        for budget in (12000,15000):
            prior_quality = path.parent / f"budget{budget}/quality.json"
            if prior_quality.exists():
                continue
            prefix = rhat_precheck(path / "samples.npy",model,draws_limit=budget)
            (path / f"prefix{budget}-rhat-precheck.json").write_text(json.dumps(metrics.json_safe(prefix),indent=2)+"\n")
            if prefix["split_rhat_median"] <= 1.01 and prefix["sample_bound_violation_fraction"] == 0:
                (path / "quality.json").write_text(json.dumps(dict(status="pilot_prefix_selected",selected_draws=budget,
                    precheck=prefix,note="20k selection pilot; requires fresh exact selected-budget sampling and full common diagnostics. No20k performance headline."),indent=2)+"\n")
                return
    check = None
    skip_duplicate_scan = ((run.get("method") == "hopsy_chrr" and run["draws"] >= 12000)
                           or run.get("exe_sha256") == "51854688114677b14de3c110c262d210281de1f351f1ae6df6be994c2cc67e82")
    if not cached and not skip_duplicate_scan:
        check = rhat_precheck(path / "samples.npy", model)
        (path / "rhat-precheck.json").write_text(json.dumps(metrics.json_safe(check), indent=2) + "\n")
    else:
        (path / "precheck-skipped.json").write_text(json.dumps(dict(reason="Previously selected CPU retime or validated coordinate configuration goes directly to full common metrics, or strict exact-byte cache proof succeeded. Avoids a redundant whole-data scan within the30-minute cap; full diagnostics still check Rhat/ESS/bounds."), indent=2) + "\n")
    if check is not None and (check["split_rhat_median"] > 1.0100000001 or check["sample_bound_violation_fraction"]):
        (path / "quality.json").write_text(json.dumps(dict(status="pilot_budget_failed_rhat_or_bounds", precheck=check,
            note="No pooledESS claim. Larger budget requires a new sampling process; frozen GPU signed32 index cap limits512chains to <=6219 retained draws with1000warmup."), indent=2) + "\n")
        return
    selected_path = path.parent / "budget3000/quality.json"
    if run.get("method") == "naja" and run.get("exe_sha256") == "c7baf4f0ed282b936ac0dbac4b896bfbfc040eb06da6f84aff6605fddf4b0fc6" and run.get("requested_thinning") == 1000 and run["draws"] == 5000 and not selected_path.exists():
        shorter = rhat_precheck(path / "samples.npy", model, draws_limit=3000)
        (path / "prefix3000-rhat-precheck.json").write_text(json.dumps(metrics.json_safe(shorter), indent=2) + "\n")
        if shorter["split_rhat_median"] <= 1.01 and shorter["sample_bound_violation_fraction"] == 0:
            (path / "quality.json").write_text(json.dumps(dict(status="pilot_prefix_selected", selected_draws=3000,
                pilot_process_seconds=run["wall_including_normalization_seconds"], precheck=shorter,
                note="This5k run is a selection pilot. Fresh3k sampling and full common metrics required; no selected-budget performance or ESS claim yet."), indent=2) + "\n")
            return
    if not cached:
        metrics.evaluate(path / "samples.npy", model, path, prefixes=[run["draws"]], workers=4,wall_seconds=wall_override)
    save_identity(path / "samples.npy",model,path)
    m = json.loads((path / "metrics.json").read_text())
    wall = m["wall_seconds"]
    passed = m["split_rhat_median"] <= 1.01 and m["bulk_ess_median"] >= 100 and m["sample_bound_violation_fraction"] == 0
    summary = dict(status="passed" if passed else "failed_quality", seed=run["seed"], chains=512,
                   draws_per_chain=run["draws"], warmup_per_chain=1000, process_seconds=wall,
                   split_rhat_median=m["split_rhat_median"], pooled_bulk_ess_median=m["bulk_ess_median"],
                   pooled_bulk_ess_median_per_chain=m["bulk_ess_median"] / 512,
                   pooled_bulk_ess_median_per_process_second=m["bulk_ess_median"] / wall,
                   bounds_tolerance=m["bounds_tolerance"], sample_bound_violation_fraction=m["sample_bound_violation_fraction"],
                   schema_version=m["schema_version"], diagnostic_mask_status=m["diagnostic_mask_status"],
                   diagnostics_reused_from=m.get("diagnostics_reused_from"),
                   count_note="GPU and CPU retained counts differ; both use the same pooled quality criterion. ESS/512 is an equivalent average from pooledESS, not independent single-chain ESS estimates.",
                   timing_note="Sampling process wall includes warmup and output; diagnostics are separate and excluded. One-seed pilot until distinct seeds202001/303001 complete.")
    sampler_phase = run["timings"]["sample_seconds"] if run["method"] == "hopsy_chrr" else run["profile"]["timing"]["gpu_sampling_s"]
    thin = run["thinning"] if run["method"] == "hopsy_chrr" else run["profile"]["config"]["thinning"]
    transitions = run["chains"]*(run["draws"]+run["warmup"])*thin
    summary.update(sampler_phase_seconds=sampler_phase,pooled_bulk_ess_per_sampler_phase_second=m["bulk_ess_median"]/sampler_phase,
                   transitions_including_warmup=transitions,pooled_bulk_ess_per_total_transition=m["bulk_ess_median"]/transitions,
                   sampler_phase_scope="Inclusive hopsy.sample: pool startup, pickle, IPC and array assembly included." if run["method"] == "hopsy_chrr" else "GPU sampler phase includes allocation, initial slackGEMM/RNG init and CHRRsampling; excludes download/output; not isolated kernel-only time.")
    if worker_timing:
        summary.update(worker_draw_critical_path_seconds=worker_timing["worker_draw_critical_path_seconds"],
                       total_worker_draw_thread_cpu_seconds=worker_timing["total_worker_draw_thread_cpu_seconds"],
                       pooled_bulk_ess_per_worker_draw_second=m["bulk_ess_median"]/worker_timing["worker_draw_critical_path_seconds"],
                       worker_draw_scope=worker_timing["critical_path_definition"])
    (path / "quality.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
