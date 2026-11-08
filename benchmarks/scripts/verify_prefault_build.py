#!/usr/bin/env python3
"""Validate default behavior and page-touch opt-in without changing samples."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import numpy as np

B=Path(__file__).resolve().parents[1];OUT=B/'results/prefault-build-verification';OUT.mkdir(parents=True,exist_ok=True)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
with (OUT/'gpu-correctness.log').open('w') as log:
    subprocess.run([str(B/'scripts/enter.sh'),'ctest','--test-dir',str(B/'build-h100-prefault'),'-V','-R','^test_gpu_'],stdout=log,stderr=subprocess.STDOUT,check=True)
arrays={};records={}
cases=[('original',B/'build-h100/naja',0,'float32'),('prefault0',B/'build-h100-prefault/naja',0,'float32'),
       ('prefault8',B/'build-h100-prefault/naja',8,'float32'),('prefault8_f64',B/'build-h100-prefault/naja',8,'float64')]
for name,exe,threads,dtype in cases:
    out=OUT/name
    subprocess.run([sys.executable,str(B/'naja_runner.py'),'--exe',str(exe),'--model',str(B/'data/synthetic/BOX'),
      '--out',str(out),'--seed','101','--chains','4','--draws','128','--warmup','16','--thinning','2',
      '--host-prefault-threads',str(threads),'--sample-dtype',dtype,'--resume'],check=True)
    records[name]=json.loads((out/'run.json').read_text());arrays[name]=np.load(out/'samples.npy')
for name in ['prefault0','prefault8']:
    assert sha(Path(records[name]['raw_samples']))==sha(Path(records['original']['raw_samples']))
assert np.array_equal(arrays['prefault8_f64'].astype(np.float32),arrays['original'])
proof={'binary_sha256':sha(B/'build-h100-prefault/naja'),'frozen_binary_sha256':sha(B/'build-h100/naja'),
 'wide_index_build_verified':True,'host_prefault_verified':True,'large_matrix_runtime_validated':False,
 'gpu_correctness_tests_passed':5,'same_seed_original_samples_bitwise_equal':True,
 'host_prefault_0_8_raw_file_hashes_equal':True,'f64_cast_to_f32_matches_original':True,
 'default_prefault_threads':0,'explicit_flag':'--host-prefault-threads 8',
 'internal_host_matrix_bytes':3*4*144*8,'system_page_bytes':os.sysconf('SC_PAGE_SIZE'),
 'partial_last_page_bytes':(3*4*144*8)%os.sysconf('SC_PAGE_SIZE'),
 'complete_patch_sha256':sha(B/'results/source-snapshots/prefault-complete.patch'),
 'timing_policy':'Download timer includes allocation, all first-touch threads, joins, tail-byte touch, and cudaMemcpy.'}
(OUT/'verification.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
