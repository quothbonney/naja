#!/usr/bin/env python3
"""Exploratory 512-chain H100 throughput at TPB128 and TPB512."""
import argparse
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--exe',type=Path)
p.add_argument('--wide-indices-proof',type=Path)
a=p.parse_args()
b=Path(__file__).resolve().parents[1]
for tpb in [128,512]:
    command=[sys.executable,str(b/'naja_runner.py'),'--exe',str(a.exe or b/'build-h100/naja'),
        '--model',str(a.model),'--out',str(a.out/f'chains512_tpb{tpb}'),'--seed','101',
        '--chains','512','--tpb',str(tpb),'--draws','10000','--warmup','1000',
        '--timeout','360','--resume']
    if a.wide_indices_proof: command+=['--wide-indices-proof',str(a.wide_indices_proof)]
    subprocess.run(command,check=True)
