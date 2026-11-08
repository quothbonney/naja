#!/usr/bin/env python3
"""Three paired fresh-process timings of default versus eight-thread prefault.

Order is off/on, on/off, off/on; each case uses the same seed and trajectory.
Full SHA256 verifies retained output bytes. This is a performance retiming,
not six independent MCMC replicates.
"""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

B=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
proof=B/'results/prefault-build-verification/verification.json'
verification=json.loads(proof.read_text());assert verification['host_prefault_verified']
reference_sha=None;rows=[]
for order,(pair,threads) in enumerate([(1,0),(1,8),(2,8),(2,0),(3,0),(3,8)],start=1):
    name=f'pair{pair}_threads{threads}';out=a.out/name
    command=[sys.executable,str(B/'naja_runner.py'),'--exe',str(B/'build-h100-prefault/naja'),
       '--model',str(B/'data/models/iJO1366'),'--out',str(out),'--seed','101','--chains','512',
       '--draws','10000','--warmup','1000','--tpb','128','--host-prefault-threads',str(threads),
       '--wide-indices-proof',str(proof),'--timeout','600','--resume']
    subprocess.run(command,check=True)
    run=json.loads((out/'run.json').read_text());assert run['status']=='complete'
    with (out/'samples.npy').open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    if reference_sha is None:reference_sha=digest
    assert digest==reference_sha,'Same-seed output bytes changed across off/on cases'
    timing=run['profile']['timing']
    row=dict(pair=pair,order=order,host_prefault_threads=threads,
       native_process_seconds=run['wall_seconds'],download_including_prefault_seconds=timing['download_from_gpu_s'],
       gpu_sampling_seconds=timing['gpu_sampling_s'],write_seconds=timing['write_output_s'],
       normalization_seconds=run['normalization_seconds'],
       end_to_end_seconds=run['wall_including_normalization_seconds'],samples_sha256=digest,
       samples_bitwise_equal_to_first=True,run_dir=str(out))
    rows.append(row)
    (a.out/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(row),flush=True)
with (a.out/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
pairs=[]
for pair in [1,2,3]:
    off=next(r for r in rows if r['pair']==pair and r['host_prefault_threads']==0)
    on=next(r for r in rows if r['pair']==pair and r['host_prefault_threads']==8)
    pairs.append({'pair':pair,'download_speedup_including_first_touch':off['download_including_prefault_seconds']/on['download_including_prefault_seconds'],
      'native_process_speedup':off['native_process_seconds']/on['native_process_seconds'],
      'end_to_end_speedup':off['end_to_end_seconds']/on['end_to_end_seconds']})
medians={str(mode):{field:statistics.median(r[field] for r in rows if r['host_prefault_threads']==mode)
    for field in ['download_including_prefault_seconds','native_process_seconds','normalization_seconds','end_to_end_seconds']}
    for mode in [0,8]}
summary={'model':'iJO1366','chains':512,'draws_per_chain':10000,'warmup':1000,'tpb':128,'seed':101,
 'binary_sha256':verification['binary_sha256'],'dtype':'float32',
 'scope':'Opt-in host first-touch only, same binary, same seed, fresh processes; all retained arrays SHA256-identical.',
 'order':'off/on, on/off, off/on','download_timing':'Includes Eigen allocation, first-touch threads, all joins, final-byte touch, and cudaMemcpy.',
 'fixed_resources':'One H100, eight CPUs,128GiB requested memory; no allocation/prefault cost omitted.',
 'medians':medians,'pairs':pairs,'rows':rows,
 'median_paired_end_to_end_speedup':statistics.median(r['end_to_end_speedup'] for r in pairs),
 'median_paired_native_process_speedup':statistics.median(r['native_process_speedup'] for r in pairs),
 'median_paired_download_speedup':statistics.median(r['download_speedup_including_first_touch'] for r in pairs),
 'finished_utc':datetime.now(timezone.utc).isoformat()}
(a.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2),flush=True)
