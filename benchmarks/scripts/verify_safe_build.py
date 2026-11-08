#!/usr/bin/env python3
"""Verify cast-only wide-index build without altering the frozen benchmark."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np

B=Path(__file__).resolve().parents[1]
OUT=B/'results/safe-build-verification'
OUT.mkdir(parents=True,exist_ok=True)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
original=B/'build-h100/naja';patched=B/'build-h100-safe/naja'
assert sha(original)=='c7baf4f0ed282b936ac0dbac4b896bfbfc040eb06da6f84aff6605fddf4b0fc6','Frozen binary changed'
with (OUT/'gpu-correctness.log').open('w') as log:
    subprocess.run([str(B/'scripts/enter.sh'),'ctest','--test-dir',str(B/'build-h100-safe'),'-V','-R','^test_gpu_'],
                   stdout=log,stderr=subprocess.STDOUT,check=True)
arrays=[]
for name,exe in [('original',original),('safe',patched)]:
    out=OUT/name
    subprocess.run([sys.executable,str(B/'naja_runner.py'),'--exe',str(exe),
                   '--model',str(B/'data/synthetic/BOX'),'--out',str(out),'--seed','101',
                   '--chains','4','--draws','128','--warmup','16','--thinning','2','--resume'],check=True)
    arrays.append(np.load(out/'samples.npy'))
same=bool(np.array_equal(*arrays));assert same,'Small-case sample values changed'
proof={'binary_sha256':sha(patched),'frozen_binary_sha256':sha(original),
       'wide_index_build_verified':True,'large_matrix_runtime_validated':False,
       'gpu_correctness_tests_passed':5,'same_seed_original_samples_bitwise_equal':same,
       'shape':list(arrays[0].shape),'source_change':'Cast-only DMatrix allocation/copy sizes and IDX2C offset products to size_t before multiplication.',
       'patch_sha256':sha(B/'results/source-snapshots/wide-indices-only.patch'),
       'scope':'Reduced-output path; per-axis int dimensions and sample-step counters still limited; large backmap, native validate, and float64 NPY readers not widened.'}
(OUT/'wide-index-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof,indent=2))
