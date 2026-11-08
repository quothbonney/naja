"""Opt-in prefault512-chain pair and subsequent distinct experimental runs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

B = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B))
import metrics
from quality_diagnostics import reuse,save_identity

def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()

def read(path):
    return json.loads(path.read_text())

def run_case(path,thin,draws,seed,threads):
    proof = B / "results/prefault-build-verification/verification.json"
    verification = read(proof)
    exe = B / "build-h100-prefault/naja"
    assert verification["host_prefault_verified"] and verification["gpu_correctness_tests_passed"] == 5
    assert sha(exe) == verification["binary_sha256"] == "a656a1f93489af051acd540d08369092567dd0673af2b0e1ac922dc58a910603"
    assert 512*(draws+1000)*581 < 2**31
    command = ["/usr/bin/env","bash",str(B / "scripts/enter.sh"),"python",str(B / "naja_runner.py"),
               "--exe",str(exe),"--model",str(B / "data/models/iJO1366"),"--out",str(path),
               "--seed",str(seed),"--chains","512","--draws",str(draws),"--warmup","1000",
               "--tpb","128","--thinning",str(thin),"--barrier","rotate","--pair-prob","0",
               "--host-prefault-threads",str(threads),"--wide-indices-proof",str(proof),"--timeout","1200","--resume"]
    subprocess.run(command,check=True,timeout=1250)
    run = read(path / "run.json")
    assert run["status"] == "complete"
    return run

def quality(path,run,m):
    wall = run["wall_including_normalization_seconds"]
    passed = (m["schema_version"] == 2 and m["diagnostic_mask_status"] == "model_lp_verified"
              and m["split_rhat_median"] <= 1.01 and m["bulk_ess_median"] >= 100
              and m["bounds_tolerance"] == 1e-7 and m["sample_bound_violation_fraction"] == 0)
    result = dict(status="passed" if passed else "failed_quality",seed=run["seed"],chains=512,
                  draws_per_chain=run["draws"],warmup_per_chain=1000,process_seconds=wall,
                  split_rhat_median=m["split_rhat_median"],pooled_bulk_ess_median=m["bulk_ess_median"],
                  pooled_bulk_ess_median_per_chain=m["bulk_ess_median"]/512,
                  pooled_bulk_ess_median_per_process_second=m["bulk_ess_median"]/wall,
                  bounds_tolerance=m["bounds_tolerance"],sample_bound_violation_fraction=m["sample_bound_violation_fraction"],
                  schema_version=m["schema_version"],diagnostic_mask_status=m["diagnostic_mask_status"],
                  experimental_binary_sha256=run["exe_sha256"],host_prefault_threads=run["host_prefault_threads"],
                  diagnostics_reused_from=m.get("diagnostics_reused_from"),
                  count_note="PooledESS/512 is an equivalent average, not independent single-chain ESS estimates.")
    phase = run["profile"]["timing"]["gpu_sampling_s"]
    transitions = 512*(run["draws"]+run["warmup"])*run["profile"]["config"]["thinning"]
    result.update(sampler_phase_seconds=phase,pooled_bulk_ess_per_sampler_phase_second=m["bulk_ess_median"]/phase,
                  transitions_including_warmup=transitions,pooled_bulk_ess_per_total_transition=m["bulk_ess_median"]/transitions,
                  sampler_phase_scope="GPU sampler phase includes allocation, initial slackGEMM/RNG init and sampling. Not isolated CHRRkernel-only time; download/output excluded.")
    (path / "quality.json").write_text(json.dumps(result,indent=2)+"\n")
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=["pair","sample"])
    parser.add_argument("--reference",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--thin",type=int,choices=[500,1000],required=True)
    parser.add_argument("--draws",type=int,choices=[2000,3000,5000],required=True)
    parser.add_argument("--seed",type=int,default=101001)
    args = parser.parse_args()
    if args.mode == "sample":
        assert args.out is not None
        run_case(args.out,args.thin,args.draws,args.seed,8)
        return
    assert args.reference is not None
    reference = read(args.reference / "run.json")
    reference_metrics = read(args.reference / "metrics.json")
    assert reference_metrics["schema_version"] == 2 and reference_metrics["diagnostic_mask_status"] == "model_lp_verified"
    assert reference["exe_sha256"] == "c7baf4f0ed282b936ac0dbac4b896bfbfc040eb06da6f84aff6605fddf4b0fc6"
    assert reference["seed"] == args.seed and reference["draws"] == args.draws
    parent = B / "results/high_throughput/iJO1366"
    pair = parent / f"prefault512_pair_thin{args.thin}/seed{args.seed}"
    pair.mkdir(parents=True,exist_ok=True)
    reference_sha = sha(args.reference / "samples.npy")
    save_identity(args.reference / "samples.npy",B / "data/models/iJO1366",args.reference)
    rows = []
    for threads in (0,8):
        path = parent / f"naja512_tpb128_thin{args.thin}_prefault{threads}/seed{args.seed}"
        run = run_case(path,args.thin,args.draws,args.seed,threads)
        digest = sha(path / "samples.npy")
        row = dict(host_prefault_threads=threads,process_seconds=run["wall_including_normalization_seconds"],
                   samples_sha256=digest,equals_frozen_reference=digest == reference_sha,run_dir=str(path),
                   download_including_prefault_seconds=run["profile"]["timing"]["download_from_gpu_s"],
                   normalization_seconds=run["normalization_seconds"])
        rows.append(row)
        if digest == reference_sha:
            reused = reuse(path / "samples.npy",B / "data/models/iJO1366",path,row["process_seconds"],args.reference)
            if not reused:
                metrics.evaluate(path / "samples.npy",B / "data/models/iJO1366",path,prefixes=[args.draws],workers=4)
            save_identity(path / "samples.npy",B / "data/models/iJO1366",path)
            row["quality"] = quality(path,run,read(path / "metrics.json"))
        (pair / "progress.json").write_text(json.dumps(rows,indent=2)+"\n")
    passed = all(row["equals_frozen_reference"] and row.get("quality",{}).get("status") == "passed" for row in rows)
    off,on = rows
    result = dict(status="passed" if passed else "identity_or_quality_failed",thin=args.thin,draws=args.draws,
                  seed=args.seed,experimental_binary_sha256="a656a1f93489af051acd540d08369092567dd0673af2b0e1ac922dc58a910603",
                  reference_run=str(args.reference),reference_samples_sha256=reference_sha,
                  reference_process_seconds=reference["wall_including_normalization_seconds"],rows=rows,
                  prefault_wins=passed and on["process_seconds"] < off["process_seconds"]
                                and on["process_seconds"] < reference["wall_including_normalization_seconds"],
                  paired_process_speedup=off["process_seconds"]/on["process_seconds"],
                  note="One fresh off/on pair, same experimental build and seed, off then on. All retained-array bytes must equal frozen reference before guarded diagnostics reuse. Prefault/allocation costs included in measured process wall. This pair is a selection pilot, not independent RNG replicates.")
    (pair / "summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)

if __name__ == "__main__":
    main()
