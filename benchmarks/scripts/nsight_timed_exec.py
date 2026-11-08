"""Measure only the target process wall time, inside/outside Nsight alike."""
import json
import subprocess
import sys
import time
from pathlib import Path

record = Path(sys.argv[1])
command = sys.argv[2:]
start_epoch_ns = time.time_ns()
start = time.perf_counter()
result = subprocess.run(command)
record.write_text(json.dumps(dict(command=command, returncode=result.returncode,
    target_wall_seconds=time.perf_counter()-start, start_epoch_ns=start_epoch_ns,
    end_epoch_ns=time.time_ns()), indent=2)+'\n')
raise SystemExit(result.returncode)
