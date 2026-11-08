"""Held-out fixed-budget validation; bounded extensions remain separate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

B = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B))
import metrics

def read(path):
    return json.loads(path.read_text())

def run_case(method,seed,draws,path):
    if method == "cpu":
        command = [str(B / "env-cpu-native/bin/python"),str(B / "scripts/timed_cpu_baseline.py"),"sample",
            "--model",str(B / "data/models/iJO1366"),"--out",str(path),"--seed",str(seed),"--chains","16",
            "--draws",str(draws),"--warmup","1000","--thinning","2000","--workers","16","--chunk",str(draws+1000),"--build-variant","native"]
    else:
        exe = B / "build-h100/naja"
        assert hashlib.file_digest(exe.open("rb"),"sha256").hexdigest() == "c7baf4f0ed282b936ac0dbac4b896bfbfc040eb06da6f84aff6605fddf4b0fc6"
        command = ["/usr/bin/env","bash",str(B / "scripts/enter.sh"),"python",str(B / "naja_runner.py"),
            "--exe",str(exe),"--model",str(B / "data/models/iJO1366"),"--out",str(path),"--seed",str(seed),
            "--chains","16","--draws",str(draws),"--warmup","1000","--tpb","512","--thinning","1000",
            "--barrier","rotate","--pair-prob","0","--timeout","600"]
    subprocess.run(command,check=True,timeout=900)
    run = read(path / "run.json")
    assert run["status"] == "complete"
    timing = read(path / "sampling_math_timing.json") if method == "cpu" else None
    wall = timing["instrumented_wrapper_process_seconds"] if timing else run["wall_including_normalization_seconds"]
    metrics.evaluate(path / "samples.npy",B / "data/models/iJO1366",path,prefixes=[draws],workers=4,wall_seconds=wall)
    m = read(path / "metrics.json")
    passed = (m["schema_version"] == 2 and m["diagnostic_mask_status"] == "model_lp_verified"
              and m["split_rhat_median"] <= 1.01 and m["bulk_ess_median"] >= 100
              and m["bounds_tolerance"] == 1e-7 and m["sample_bound_violation_fraction"] == 0)
    phase = run["timings"]["sample_seconds"] if method == "cpu" else run["profile"]["timing"]["gpu_sampling_s"]
    thin = 2000 if method == "cpu" else 1000
    total = 16*(draws+1000)*thin
    result = dict(status="PASS" if passed else "FAIL",quality_met=passed,method=method,seed=seed,chains=16,
        draws_per_chain=draws,warmup=1000,thinning=thin,sampler_phase_seconds=phase,process_seconds_secondary=wall,
        split_rhat_median=m["split_rhat_median"],pooled_bulk_ess_median=m["bulk_ess_median"],
        pooled_bulk_ess_per_sampler_phase_second=m["bulk_ess_median"]/phase,transitions_including_warmup=total,
        pooled_bulk_ess_per_total_transition=m["bulk_ess_median"]/total,bounds_tolerance=m["bounds_tolerance"],
        sample_bound_violation_fraction=m["sample_bound_violation_fraction"],schema_version=m["schema_version"],
        diagnostic_mask_status=m["diagnostic_mask_status"],run_dir=str(path))
    if timing:
        result.update(worker_draw_critical_path_seconds=timing["worker_draw_critical_path_seconds"],
            total_worker_draw_thread_cpu_seconds=timing["total_worker_draw_thread_cpu_seconds"],
            pooled_bulk_ess_per_worker_draw_second=m["bulk_ess_median"]/timing["worker_draw_critical_path_seconds"])
    (path / "quality.json").write_text(json.dumps(result,indent=2)+"\n")
    return result

def main():
    parser = argparse.ArgumentParser();parser.add_argument("--task",type=int,required=True);args = parser.parse_args()
    seed = [101001,202001,303001][args.task//2]
    method = ["cpu","gpu"][args.task%2]
    label = "cpu_native_physical16_thin2000" if method == "cpu" else "naja_frozen_tpb512_thin1000"
    out = B / f"results/fresh16_validation/iJO1366/{label}/seed{seed}"
    out.mkdir(parents=True,exist_ok=False)
    allowed = sorted(os.sched_getaffinity(0));topology=[];unique={}
    for cpu in allowed:
        top = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        row = dict(cpu=cpu,socket=int((top / "physical_package_id").read_text()),core=int((top / "core_id").read_text()),siblings=(top / "thread_siblings_list").read_text().strip())
        topology.append(row);unique.setdefault((row["socket"],row["core"]),cpu)
    assert len(allowed) == 32 and len(unique) == 16
    os.sched_setaffinity(0,sorted(unique.values()))
    (out / "physical_affinity.json").write_text(json.dumps(dict(reserved_logical_cpus=32,selected_physical_core_count=16,
        allowed_logical_cpus=allowed,selected_logical_cpus=sorted(os.sched_getaffinity(0)),topology=topology),indent=2)+"\n")
    fixed_draws = 10000 if method == "cpu" else 2000
    result = dict(method=method,seed=seed,status="running",fixed_draws=fixed_draws,
        note="Fresh fixed selected configuration, no tuning. Cross-replicate16-chain RNG ranges disjoint; overlap with512study intentional. Fixed failures remain failures even if the separately labeled bounded extension passes.")
    try:
        fixed = run_case(method,seed,fixed_draws,out / "fixed")
        result.update(fixed=fixed,fixed_status=fixed["status"])
        (out / "validation.json").write_text(json.dumps(result,indent=2)+"\n")
        if fixed["status"] == "FAIL":
            extension_draws = 15000 if method == "cpu" else 3000
            result["extension"] = run_case(method,seed,extension_draws,out / f"bounded_extension{extension_draws}")
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed_backend_or_diagnostics",error=repr(error))
        raise
    finally:
        (out / "validation.json").write_text(json.dumps(result,indent=2)+"\n")
        subprocess.run([sys.executable,str(B / "scripts/summarize_fresh16.py")],check=False)

if __name__ == "__main__":
    main()
