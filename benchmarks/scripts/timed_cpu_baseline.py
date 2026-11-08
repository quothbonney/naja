"""Run the unchanged CPU CLI with opt-in worker draw-call timing sidecar."""
import json
from pathlib import Path
import runpy
import sys
import time

def main():
    started = time.perf_counter()
    B = Path(__file__).resolve().parents[1]
    arguments = sys.argv[1:]
    assert arguments[0] == "sample"
    out = Path(arguments[arguments.index("--out")+1]).resolve()
    import hopsy_worker_timing
    proof = hopsy_worker_timing.install(out / "worker_math_logs")
    sys.argv = [str(B / "cpu_baseline.py")]+arguments
    runpy.run_path(str(B / "cpu_baseline.py"),run_name="__main__")
    run = json.loads((out / "run.json").read_text())
    result = hopsy_worker_timing.summarize(out / "worker_math_logs",run["chains"],run["draws"]+run["warmup"])
    result.update(proof=proof,instrumented_wrapper_process_seconds=time.perf_counter()-started,
                  unmodified_baseline_process_seconds=run["process_wall_seconds"],
                  hopsy_sample_phase_inclusive_seconds=run["timings"]["sample_seconds"],
                  timing_note="Wrapper process includes instrumentation/hopsy import setup. Unmodified baseline PROCESS_STARTED begins after those imports in this opt-in variant; use wrapper process for its full-process secondary timing.")
    (out / "sampling_math_timing.json").write_text(json.dumps(result,indent=2)+"\n")

if __name__ == "__main__":
    main()
