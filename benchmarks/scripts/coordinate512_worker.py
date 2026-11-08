"""Separate coordinate-specialized512-chain sampling and guarded first reuse."""
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
from high_throughput_prefault import quality

def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=["validate","sample"])
    parser.add_argument("--seed",type=int,default=101001)
    parser.add_argument("--out",type=Path)
    args = parser.parse_args()
    proof = B / "results/coordinate_kernel/verification.json"
    verification = json.loads(proof.read_text())
    assert verification["status"] == "passed" and verification["gpu_correctness_tests_passed"] == 5
    assert verification["binary_sha256"] == sha(B / "build-h100-coordinate/naja") == "51854688114677b14de3c110c262d210281de1f351f1ae6df6be994c2cc67e82"
    if args.mode == "sample":
        assert args.out is not None
        command = ["/usr/bin/env","bash",str(B / "scripts/enter.sh"),"python",str(B / "naja_runner.py"),
            "--exe",str(B / "build-h100-coordinate/naja"),"--model",str(B / "data/models/iJO1366"),
            "--out",str(args.out),"--seed",str(args.seed),"--chains","512","--draws","5000","--warmup","1000",
            "--thinning","500","--tpb","128","--host-prefault-threads","8","--barrier","rotate","--pair-prob","0",
            "--wide-indices-proof",str(proof),"--timeout","1200"]
        subprocess.run(command,check=True,timeout=1250)
        return
    path = B / "results/coordinate_kernel/paired/chains512/coordinate"
    reference = B / "results/high_throughput/iJO1366/naja512_tpb128_thin500/seed101001"
    sample_sha,reference_sha = sha(path / "samples.npy"),sha(reference / "samples.npy")
    assert sample_sha == reference_sha
    save_identity(reference / "samples.npy",B / "data/models/iJO1366",reference)
    run = json.loads((path / "run.json").read_text())
    if not reuse(path / "samples.npy",B / "data/models/iJO1366",path,run["wall_including_normalization_seconds"],reference):
        metrics.evaluate(path / "samples.npy",B / "data/models/iJO1366",path,prefixes=[5000],workers=4)
    save_identity(path / "samples.npy",B / "data/models/iJO1366",path)
    q = quality(path,run,json.loads((path / "metrics.json").read_text()))
    q.update(samples_sha256=sample_sha,reference_samples_sha256=reference_sha,
             scope="Existing independently executed paired seed101001 timing; guarded diagnostics identity against frozen512thin500 output. Distinct coordinate-specialized build, no history pooling.")
    (path / "quality.json").write_text(json.dumps(q,indent=2)+"\n")
    print(json.dumps(q,indent=2))

if __name__ == "__main__":
    main()
