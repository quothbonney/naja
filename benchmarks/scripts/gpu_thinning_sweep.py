"""Frozen-binary GPU thinning pilot and fresh selected-budget retiming."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B))
import metrics
from quality_diagnostics import rhat_precheck, reuse, save_identity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    thinning = [200, 500, 1000][args.task // 3]
    seed = [101, 202, 303][args.task % 3]
    out = B / f"results/gpu_thinning/iJO1366/thin{thinning}/seed{seed}"
    out.mkdir(parents=True, exist_ok=args.resume)
    if args.resume and (out / "quality.json").exists():
        previous = json.loads((out / "quality.json").read_text())
        (out / ("prior-attempt-" + str(time.time_ns()) + ".json")).write_text(json.dumps(previous, indent=2) + "\n")
    model = B / "data/models/iJO1366"
    exe = B / "build-h100/naja"
    exe_sha = hashlib.file_digest(exe.open("rb"), "sha256").hexdigest()
    assert exe_sha == "c7baf4f0ed282b936ac0dbac4b896bfbfc040eb06da6f84aff6605fddf4b0fc6"
    result = dict(model="iJO1366", thinning=thinning, tpb=512, seed=seed, chains=16,
                  warmup=1000, status="running", binary_sha256=exe_sha,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  pilot_draws=10000, selection_grid=[100,250,500,1000,2000,5000,10000],
                  note="Pilot and diagnostic costs excluded from fresh selected-budget process timing. Three complete passing seeds required for a group comparison.")
    
    def write(path, data):
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(metrics.json_safe(data), indent=2, allow_nan=False) + "\n")
        os.replace(temp, path)
    
    def read(path):
        return json.loads(path.read_text())
    
    def qualifies(m):
        return (m.get("schema_version") == 2 and m.get("diagnostic_mask_status") == "model_lp_verified"
                and m.get("split_rhat_median") is not None and m["split_rhat_median"] <= 1.01
                and m.get("bulk_ess_median") is not None and m["bulk_ess_median"] >= 100
                and m["sample_bound_violation_fraction"] == 0 and m["bounds_tolerance"] == 1e-7)
    
    def sample(phase, draws):
        path = out / phase
        if args.resume and (path / "run.json").exists():
            existing = read(path / "run.json")
            if existing.get("status") == "complete":
                assert existing["requested_thinning"] == thinning and existing["exe_sha256"] == exe_sha
                return path, existing
        command = ["/usr/bin/env", "bash", str(B / "scripts/enter.sh"), "python", str(B / "naja_runner.py"),
                   "--exe", str(exe), "--model", str(model), "--out", str(path), "--seed", str(seed),
                   "--chains", "16", "--draws", str(draws), "--warmup", "1000", "--tpb", "512",
                   "--thinning", str(thinning), "--timeout", "1200", "--barrier", "rotate", "--pair-prob", "0"]
        with (out / f"{phase}-sampling.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1250)
        run = read(path / "run.json")
        assert run["status"] == "complete" and run["exe_sha256"] == exe_sha
        assert run["requested_thinning"] == thinning
        return path, run
    
    write(out / "quality.json", result)
    try:
        begun = time.perf_counter()
        pilot, pilot_run = sample("pilot", 10000)
        checks = (read(pilot / "rhat-precheck.json")["prefixes"] if args.resume and (pilot / "rhat-precheck.json").exists() else
                  [rhat_precheck(pilot / "samples.npy", model, draws_limit=n) for n in result["selection_grid"]])
        write(pilot / "rhat-precheck.json", {"prefixes":checks})
        eligible = [n for n, c in zip(result["selection_grid"], checks)
                    if c["split_rhat_median"] is not None and c["split_rhat_median"] <= 1.0100000001
                    and c["sample_bound_violation_fraction"] == 0]
        result["pilot_process_seconds"] = pilot_run["wall_including_normalization_seconds"]
        result["pilot_prechecks"] = checks
        if not eligible:
            result["status"] = "pilot_cap_no_passing_rhat"
        else:
            metrics.evaluate(pilot / "samples.npy", model, pilot, prefixes=eligible)
            save_identity(pilot / "samples.npy", model, pilot)
            pm = read(pilot / "metrics.json")
            passing = [row["draws_per_chain"] for row in pm["prefix_quality"] if row["quality_met"]]
            if not passing:
                result["status"] = "pilot_cap_no_passing_quality"
            else:
                selected = min(passing)
                result["selected_draws"] = selected
                result["pilot_and_selection_seconds"] = time.perf_counter() - begun
                write(out / "quality.json", result)
                final, final_run = sample("final", selected)
                wall = final_run["wall_including_normalization_seconds"]
                cached = selected == 10000 and reuse(final / "samples.npy", model, final, wall, pilot)
                if not cached:
                    metrics.evaluate(final / "samples.npy", model, final, prefixes=[selected])
                save_identity(final / "samples.npy", model, final)
                fm = read(final / "metrics.json")
                result["status"] = "complete" if qualifies(fm) else "selected_budget_failed_revalidation"
                result["measured"] = dict(draws_per_chain=selected, wall_seconds=wall,
                                          split_rhat_median=fm["split_rhat_median"], bulk_ess_median=fm["bulk_ess_median"],
                                          quality_met=qualifies(fm), sample_bound_violation_fraction=fm["sample_bound_violation_fraction"],
                                          diagnostics_reused_from=fm.get("diagnostics_reused_from"), run_dir=str(final))
    except Exception as error:
        result.update(status="failed", error=repr(error))
        write(out / "quality.json", result)
        raise
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write(out / "quality.json", result)
    print(json.dumps(result, indent=2), flush=True)

if __name__ == "__main__":
    main()
