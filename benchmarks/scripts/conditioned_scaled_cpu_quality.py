"""Separately time CPU on the independently validated scaled-coordinate MVE."""
import argparse,json,sys,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
B=Path(__file__).resolve().parents[1];sys.path[:0]=[str(B),str(B/'scripts')]
import conditioned_quality as quality
from conditional_model_api import sha,write_json

def bounded_call(command,log):
 remaining=(datetime.fromisoformat('2026-11-08T22:00:00+00:00')-datetime.now(timezone.utc)).total_seconds()
 if remaining<=0:raise TimeoutError('New extension deadline22UTC')
 log=Path(log);log.parent.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 with log.open('w') as f:subprocess.run(list(map(str,command)),check=True,stdout=f,stderr=subprocess.STDOUT,timeout=min(2700,remaining))
 return time.perf_counter()-start
quality.call=bounded_call
root=B/'results/conditioned_extensions/b0031_scaled_global/b0031'
a=argparse.Namespace(condition_root=root,variant='global_cpu',seed=101,pilot_source=None,allow_200k=False,sample_only=False)
quality.main(a)
p=root/'quality/global_cpu/seed101/quality.json';r=json.loads(p.read_text());r.update(extension_wrapper_sha256=sha(Path(__file__)),baseline_note='Separate full PolyRound rerounding after uniform initial coordinate scale1e-6. Default PolyRound thresholds unchanged; original failed unscaled reference retained. Same exact target/inherited reaction mask.');write_json(p,r)
