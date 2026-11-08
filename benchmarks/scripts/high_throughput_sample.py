"""Bounded512-chain pilot using unchanged samplers and explicit resources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

B = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--seed", type=int, default=101001)
    parser.add_argument("--draws", type=int)
    parser.add_argument("--physical-cores", type=int, choices=[48,64], default=64)
    parser.add_argument("--thinning", type=int, choices=[500,1000], default=1000)
    parser.add_argument("--out",type=Path)
    args = parser.parse_args()
    label = f"cpu_native512_physical{args.physical_cores}_thin2000" if args.method == "cpu" else f"naja512_tpb128_thin{args.thinning}"
    out = B / f"results/high_throughput/iJO1366/{label}/seed{args.seed}"
    default_draws = 10000 if args.method == "cpu" else (5000 if args.thinning == 500 else 2000)
    selected_protocol = None
    protocol_path = B / "audit/high-throughput/cpu-selected-budget.json"
    if args.method == "cpu" and args.physical_cores == 64 and protocol_path.exists():
        selected_protocol = json.loads(protocol_path.read_text())
        assert selected_protocol["status"] == "validated"
        default_draws = selected_protocol["draws"]
    draws = args.draws or default_draws
    assert draws in ([10000,12000,15000,20000] if args.method == "cpu" else [2000,3000,5000])
    if draws != default_draws:
        out = out / f"budget{draws}"
    if args.out is not None:
        out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.exists(), f"Refusing existing output {out}"
    if args.method == "cpu":
        out.mkdir()
        if selected_protocol:
            (out / "selected_budget_protocol.json").write_text(json.dumps(selected_protocol,indent=2)+"\n")
        allowed = sorted(os.sched_getaffinity(0))
        topology = []
        for cpu in allowed:
            path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            topology.append(dict(cpu=cpu, socket=int((path / "physical_package_id").read_text()),
                                 core=int((path / "core_id").read_text()),
                                 siblings=(path / "thread_siblings_list").read_text().strip()))
        unique = {}
        for row in topology:
            unique.setdefault((row["socket"], row["core"]), row["cpu"])
        selected = sorted(unique.values())
        assert len(allowed) == args.physical_cores * 2 and len(selected) == args.physical_cores
        assert int(os.environ["SLURM_CPUS_PER_TASK"]) == args.physical_cores * 2
        assert all(set(map(int, row["siblings"].split(","))) <= set(allowed) for row in topology)
        os.sched_setaffinity(0, selected)
        record = dict(job_id=os.environ.get("SLURM_JOB_ID"), allowed_logical_cpus=allowed,
                      selected_logical_cpus=sorted(os.sched_getaffinity(0)), topology=topology,
                      physical_cores=args.physical_cores, reserved_logical_cpus=args.physical_cores * 2, workers=args.physical_cores,
                      created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        (out / "physical_affinity.json").write_text(json.dumps(record, indent=2) + "\n")
        os.environ["HOPSY_CXXFLAGS"] = "-O3 -march=native -mtune=native"
        os.environ["HOPSY_BUILD_MANIFEST"] = str(B / "audit/hopsy_native_build.json")
        command = [str(B / "env-cpu-native/bin/python"), str(B / "scripts/timed_cpu_baseline.py"), "sample",
                   "--model", str(B / "data/models/iJO1366"), "--out", str(out), "--seed", str(args.seed),
                   "--chains", "512", "--draws", str(draws), "--warmup", "1000", "--thinning", "2000",
                   "--workers", str(args.physical_cores), "--chunk", str(draws + 1000), "--build-variant", "native"]
    else:
        exe = B / "build-h100/naja"
        assert hashlib.file_digest(exe.open("rb"), "sha256").hexdigest() == "c7baf4f0ed282b936ac0dbac4b896bfbfc040eb06da6f84aff6605fddf4b0fc6"
        assert 512 * (draws + 1000) * 581 < 2**31
        command = ["/usr/bin/env", "bash", str(B / "scripts/enter.sh"), "python", str(B / "naja_runner.py"),
                   "--exe", str(exe), "--model", str(B / "data/models/iJO1366"), "--out", str(out),
                   "--seed", str(args.seed), "--chains", "512", "--draws", str(draws), "--warmup", "1000",
                   "--tpb", "128", "--thinning", str(args.thinning), "--timeout", "1500", "--barrier", "rotate", "--pair-prob", "0"]
    subprocess.run(command, check=True)
    if args.method == "cpu" and args.physical_cores == 64 and selected_protocol and args.seed == selected_protocol["seed"] and draws == selected_protocol["draws"]:
        begun = time.perf_counter()
        source = Path(selected_protocol["source_run"])
        def sha(path):
            with path.open("rb") as handle:
                return hashlib.file_digest(handle,"sha256").hexdigest()
        reference_sha,current_sha = sha(source / "samples.npy"),sha(out / "samples.npy")
        proof = dict(status="passed" if reference_sha == current_sha else "failed",source_run=str(source),
                     source_samples_sha256=reference_sha,samples_sha256=current_sha,
                     post_sampling_identity_seconds=time.perf_counter()-begun,
                     note="Whole retained-file bytes match the validated48physicalCPU source. Independent full64 process timing; shared same-seed trajectory. This proof gates afterok pipeline execution and is excluded from sampler timing.")
        (out / "shared_stream_identity.json").write_text(json.dumps(proof,indent=2)+"\n")
        assert reference_sha == current_sha,"Shared stream mismatch: require fresh first-case diagnostics before pipelining"

if __name__ == "__main__":
    main()
