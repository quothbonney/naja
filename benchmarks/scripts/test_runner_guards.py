#!/usr/bin/env python3
"""Regress silent CUDA failures, oversized matrices, shape and finite checks."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import numpy as np

B=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('naja_runner',B/'naja_runner.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
with tempfile.TemporaryDirectory(dir=B/'tmp') as directory:
    root=Path(directory);model=root/'BOX';(model/'rounding').mkdir(parents=True)
    np.savez(model/'rounding/polytope.npz',A=np.eye(3),b=np.ones(3))
    fake=root/'fake_naja';fake.write_text('#!/bin/bash\necho "CUDA Error: out of memory" >&2\nexit 0\n');fake.chmod(0o700)
    out=root/'error_out'
    result=subprocess.run([sys.executable,str(B/'naja_runner.py'),'--exe',str(fake),'--model',str(model),
        '--out',str(out),'--seed','1','--chains','2','--draws','4','--warmup','1'],capture_output=True,text=True)
    record=json.loads((out/'run.json').read_text())
    assert result.returncode!=0 and record['status']=='failed' and record['backend_errors']
    assert not (out/'samples.npy').exists(),'Zero-exit CUDA failure must never normalize output'
    np.savez(model/'rounding/polytope.npz',A=np.ones((1,581)),b=np.ones(1))
    try: runner.preflight_dimension(model,512,10000,1000)
    except ValueError as error: assert 'INT_MAX' in str(error)
    else: raise AssertionError('512-chain overflow must be blocked for original binary')
    raw=root/'raw.npy';target=root/'normalized.npy'
    x=np.ones((10,3),dtype=np.float32);x[0,0]=np.nan;np.save(raw,x)
    try: runner.normalize_samples(raw,target,2,4,1,expected_dimension=3)
    except ValueError as error: assert 'Nonfinite' in str(error)
    else: raise AssertionError('Nonfinite warmup must fail')
    x[0,0]=1;x[-1,0]=np.inf;np.save(raw,x)
    try: runner.normalize_samples(raw,target,2,4,1,expected_dimension=3)
    except ValueError as error: assert 'Nonfinite' in str(error)
    else: raise AssertionError('Nonfinite retained sample must fail')
    np.save(raw,np.ones((10,2),dtype=np.float32))
    try: runner.normalize_samples(raw,target,2,4,1,expected_dimension=3)
    except ValueError as error: assert 'shape' in str(error)
    else: raise AssertionError('Unexpected reduced dimension must fail')
print('PASS: zero-exit CUDA error rejected; INT_MAX overflow blocked; nonfinite warmup/retained and wrong dimension rejected.')
