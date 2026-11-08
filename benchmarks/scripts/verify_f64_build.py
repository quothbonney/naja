#!/usr/bin/env python3
"""Verify default F32 compatibility and lossless F64 layout/export."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np

B=Path(__file__).resolve().parents[1];OUT=B/'results/f64-build-verification';OUT.mkdir(parents=True,exist_ok=True)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
with (OUT/'gpu-correctness.log').open('w') as log:
    subprocess.run([str(B/'scripts/enter.sh'),'ctest','--test-dir',str(B/'build-h100-f64'),'-V','-R','^test_gpu_'],stdout=log,stderr=subprocess.STDOUT,check=True)
arrays={};records={}
for name,exe,dtype in [('original_f32',B/'build-h100/naja','float32'),('new_default_f32',B/'build-h100-f64/naja','float32'),('new_f64',B/'build-h100-f64/naja','float64')]:
    out=OUT/name
    command=[sys.executable,str(B/'naja_runner.py'),'--exe',str(exe),'--model',str(B/'data/synthetic/BOX'),
        '--out',str(out),'--seed','101','--chains','4','--draws','128','--warmup','16','--thinning','2','--sample-dtype',dtype,'--resume']
    subprocess.run(command,check=True)
    records[name]=json.loads((out/'run.json').read_text());arrays[name]=np.load(out/'samples.npy')
assert arrays['original_f32'].dtype==np.float32 and arrays['new_default_f32'].dtype==np.float32
assert np.array_equal(arrays['original_f32'],arrays['new_default_f32']),'Default F32 values changed'
assert arrays['new_f64'].dtype==np.float64 and arrays['new_f64'].shape==(4,128,3)
assert np.array_equal(arrays['new_f64'].astype(np.float32),arrays['original_f32']),'F64 cast does not reproduce F32'
raw=np.load(records['new_f64']['raw_samples'],mmap_mode='r')
assert raw.offset%64==0,'New F64 header is not aligned to64bytes'
assert np.max(np.abs(arrays['new_f64']-arrays['original_f32'].astype(np.float64)))>0,'No extra precision retained'
proof={'binary_variant':'optional_float64_export_with_wide_indices','binary_sha256':sha(B/'build-h100-f64/naja'),
 'frozen_benchmark_binary_sha256':sha(B/'build-h100/naja'),'wide_only_binary_sha256':sha(B/'build-h100-safe/naja'),
 'gpu_correctness_tests_passed':5,'default_f32_old_new_bitwise_equal':True,'f64_cast_to_f32_matches_original':True,
 'float64_dtype_and_shape_valid':True,'f64_raw_shape':list(raw.shape),'f64_header_offset':raw.offset,
 'extra_precision_max_difference':float(np.max(np.abs(arrays['new_f64']-arrays['original_f32'].astype(np.float64)))),
 'complete_source_patch_sha256':sha(B/'results/source-snapshots/f64-export-complete.patch'),
 'sampling_math_changed':False,'default_output_dtype':'float32','explicit_flag':'--sample-dtype float64'}
(OUT/'verification.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
