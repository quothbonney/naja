#!/usr/bin/env python3
"""Sequential chain/TPB sweep inside a SLURM GPU allocation.

Emits normal naja_runner.py run.json records; default sweep changes one factor
at a time around 16 chains and TPB128. GPU and CPU comparisons should run in
separate allocations with reserved CPU counts to avoid timing contention.
"""
import argparse
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--exe',type=Path)
p.add_argument('--seed',type=int,default=101)
p.add_argument('--draws',type=int,default=10000)
p.add_argument('--warmup',type=int,default=1000)
p.add_argument('--timeout',type=float,default=3600)
p.add_argument('--axis',choices=['chains','tpb','both'],default='both')
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
exe=a.exe or root/'build-h100/naja'
cases=[]
if a.axis in ('chains','both'):
    cases.extend((f'chains{c}',c,128) for c in [8,16,32,64,128,256])
if a.axis in ('tpb','both'):
    cases.extend((f'tpb{t}',16,t) for t in [64,128,256,512])
for name,chains,tpb in cases:
    subprocess.run([sys.executable,str(root/'naja_runner.py'),'--exe',str(exe),
                    '--model',str(a.model),'--out',str(a.out/name),'--seed',str(a.seed),
                    '--chains',str(chains),'--tpb',str(tpb),'--draws',str(a.draws),
                    '--warmup',str(a.warmup),'--timeout',str(a.timeout),'--resume'],check=True)
