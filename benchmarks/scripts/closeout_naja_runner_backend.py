#!/usr/bin/env python3
"""Run one reproducible Naja benchmark; normalize native interleaved draws.

Example: python naja_runner.py --exe build/naja --model data/models/e_coli_core \
    --out results/naja/e_coli_core/seed101 --seed 101
Requires the localized explicit --seed patch recorded in results/source.patch.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def backend_errors(directory):
    pattern=re.compile(r'(?:cuda|cublas|cusolver|curand)\s+(?:runtime\s+)?error\b',re.IGNORECASE)
    return [line for name in ['stdout.log','stderr.log']
            for line in (directory/name).read_text(errors='replace').splitlines() if pattern.search(line)]


def preflight_dimension(model,chains,draws,warmup,max_elements=2**31-1):
    import numpy as np
    bundle=model/'rounding/polytope.npz'
    if bundle.exists():
        with np.load(bundle) as p: dimension=int(p['A'].shape[1])
    else:
        dimension=np.loadtxt(model/'rounding'/f'{model.name}_rounding_A.csv',delimiter=',',ndmin=2).shape[1]
    columns=chains*(draws+warmup)
    if dimension>2**31-1 or columns>2**31-1:
        raise ValueError('Current Naja per-axis dimensions must not exceed INT_MAX')
    elements=dimension*columns
    if elements>max_elements:
        raise ValueError(f'Current Naja binary uses signed32-bit matrix products: requested {elements} elements exceeds INT_MAX=2147483647; case blocked before GPU allocation')
    return dimension,elements


def normalize_samples(raw_path, destination, chains, draws, warmup, expected_dimension=None, expected_dtype="float32"):
    """Convert by small blocks, retaining native raw output for auditing."""
    import numpy as np
    raw = np.load(raw_path, mmap_mode='r')
    if raw.ndim != 2 or raw.shape[0] != chains * (draws + warmup) or (expected_dimension is not None and raw.shape[1]!=expected_dimension):
        raise ValueError(f'Unexpected native shape {raw.shape}')
    if raw.dtype!=np.dtype(expected_dtype):
        raise ValueError(f'Unexpected native dtype {raw.dtype}; requested {expected_dtype}')
    source = raw.reshape(draws + warmup, chains, raw.shape[1])
    for start in range(0,warmup,256):
        if not np.isfinite(source[start:min(start+256,warmup)]).all():
            raise ValueError('Nonfinite native warmup samples')
    target = np.lib.format.open_memmap(destination, mode='w+', dtype=raw.dtype,
                                     shape=(chains, draws, raw.shape[1]))
    for start in range(0, draws, 256):
        end = min(start + 256, draws)
        block=source[warmup+start:warmup+end]
        if not np.isfinite(block).all():
            raise ValueError('Nonfinite native retained samples')
        target[:, start:end, :] = block.transpose(1,0,2)
    target.flush()
    return {'native_shape': list(raw.shape), 'native_dtype': str(raw.dtype),
            'native_layout': 'draw-major interleaved chains; columns are reduced coordinates',
            'analysis_shape': list(target.shape), 'analysis_layout': 'chain, draw, reduced_coordinate',
            'warmup_retained_draws_discarded_per_chain': warmup}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--model', '--model-dir', dest='model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--sample-dtype',choices=['float32','float64'],default='float32')
    p.add_argument('--host-prefault-threads',type=int,default=0)
    p.add_argument('--wide-indices-proof',type=Path,help='Experimental wide build opt-in; binary hash and verification evidence must match')
    p.add_argument('--chains', type=int, default=16)
    p.add_argument('--draws', type=int, default=10000)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--tpb', type=int, default=128)
    p.add_argument('--thinning', type=int, default=0)
    p.add_argument('--pair-prob', type=float, default=0.0)
    p.add_argument('--pair-schedule-method', choices=['barrier'])
    p.add_argument('--barrier', choices=['off', 'rotate', 'whiten'], default='rotate')
    p.add_argument('--barrier-rotate-backend',choices=['cpu','gpu'])
    p.add_argument('--timeout', type=float, default=3600)
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.host_prefault_threads<0:
        p.error('--host-prefault-threads must be nonnegative')
    if not 0 <= args.pair_prob <= 1:
        p.error('--pair-prob must be in [0,1]')
    if args.seed == 0:
        p.error('--seed must be nonzero (zero selects legacy path-hash behavior)')
    if min(args.chains, args.draws) <= 0 or args.warmup < 0:
        p.error('chains/draws must be positive and warmup nonnegative')
    args.exe, args.model, args.out = (v.resolve() for v in (args.exe, args.model, args.out))
    record_path = args.out / 'run.json'
    if record_path.exists() and args.resume:
        previous = json.loads(record_path.read_text())
        if previous.get('status') == 'complete':
            print(record_path)
            return 0
    if args.out.exists() and any(args.out.iterdir()):
        p.error(f'Output directory is not empty: {args.out}; use a fresh directory')
    args.out.mkdir(parents=True, exist_ok=True)
    command = [str(args.exe), 'sample', 'run', '--model-dir', str(args.model),
               '--out-root', str(args.out / 'native'), '--gpu', str(args.gpu),
               '--n-chains', str(args.chains), '--n-samples', str(args.draws + args.warmup),
               '--seed', str(args.seed), '--tpb', str(args.tpb), '--thinning', str(args.thinning),
               '--constraint-eps', '0', '--extra-constraint-eps', '0',
               '--pair-prob', str(args.pair_prob),
               '--bounds-policy', 'ignore', '--write-npy']
    if args.barrier != 'off':
        command.append('--barrier-' + args.barrier)
        if args.barrier_rotate_backend:
            command += ['--barrier-rotate-backend',args.barrier_rotate_backend]
    if args.host_prefault_threads:
        command += ['--host-prefault-threads',str(args.host_prefault_threads)]
    if args.sample_dtype=='float64':
        command += ['--sample-dtype','float64']
    if args.pair_schedule_method:
        command += ['--pair-schedule-method', args.pair_schedule_method]
    if args.verbose:
        command.append('--verbose')
    data = {'method': 'naja', 'status': 'running', 'command': command,
            'started_utc': datetime.now(timezone.utc).isoformat(),
            'seed': args.seed, 'model': args.model.name, 'model_dir': str(args.model),
            'chains': args.chains, 'draws': args.draws, 'warmup': args.warmup,
            'barrier': args.barrier, 'barrier_rotate_backend':args.barrier_rotate_backend, 'sample_dtype':args.sample_dtype, 'host_prefault_threads':args.host_prefault_threads, 'pair_probability': args.pair_prob, 'pair_schedule_method': args.pair_schedule_method, 'requested_thinning': args.thinning,
            'exe_sha256': hashlib.file_digest(args.exe.open('rb'), 'sha256').hexdigest(),
            'adapter_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'samples': str(args.out / 'samples.npy'),
            'notes': ['All wall sampling time includes discarded warmup.',
                      'Reduced-space native path; no GPU backmap; external reaction-space diagnostics.',
                      'Dikin may auto-enable when extra constraints exist; unconditioned models have none.',
                      'Auto thinning is a pre-sampling geometry heuristic, not online adaptation.']}
    dump(record_path, data)
    started = time.perf_counter()
    try:
        checked=time.perf_counter()
        limit=2**31-1
        if args.wide_indices_proof:
            proof=json.loads(args.wide_indices_proof.read_text())
            if not (proof.get('binary_sha256')==data['exe_sha256'] and proof.get('wide_index_build_verified') and proof.get('gpu_correctness_tests_passed')==5 and proof.get('same_seed_original_samples_bitwise_equal')):
                raise ValueError('Wide-index verification proof does not match this binary')
            limit=(2**63-1)//8
            data['wide_index_proof']=proof
            data['wide_index_proof_path']=str(args.wide_indices_proof.resolve())
        dimension,elements=preflight_dimension(args.model,args.chains,args.draws,args.warmup,max_elements=limit)
        data['preflight']={'dimension':dimension,'native_matrix_elements':elements,'matrix_element_limit':limit,'per_axis_signed_int_limit':2**31-1}
        data['preflight_seconds']=time.perf_counter()-checked
        started=time.perf_counter()
        with (args.out / 'stdout.log').open('w') as stdout, (args.out / 'stderr.log').open('w') as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr,
                                    timeout=args.timeout, cwd=args.out, check=False)
        data['returncode'] = result.returncode
        data['wall_seconds'] = time.perf_counter() - started
        if result.returncode:
            raise RuntimeError(f'Naja exited with status {result.returncode}; see logs')
        errors=backend_errors(args.out)
        if errors:
            data['backend_errors']=errors
            raise RuntimeError(f'GPU backend reported errors despite zero exit status: {errors[0]}')
        profiles = list((args.out / 'native').rglob('profile.json'))
        if len(profiles) != 1:
            raise RuntimeError(f'Expected one profile.json; found {len(profiles)}')
        profile = json.loads(profiles[0].read_text())
        if profile['config'].get('host_prefault_threads',0)!=args.host_prefault_threads:
            raise RuntimeError('Requested/exported host-prefault profile mismatch')
        if profile['config'].get('sample_dtype','float32')!=args.sample_dtype:
            raise RuntimeError('Requested/exported dtype profile mismatch')
        if profile['config']['seed'] != args.seed:
            raise RuntimeError('Actual seed metadata does not match requested seed')
        data['profile'] = profile
        data['profile_path'] = str(profiles[0])
        data['raw_samples'] = str(profiles[0].parent / 'samples.npy')
        convert_started = time.perf_counter()
        data.update(normalize_samples(Path(data['raw_samples']), args.out / 'samples.npy',
                                      args.chains, args.draws, args.warmup, expected_dimension=dimension, expected_dtype=args.sample_dtype))
        data['normalization_seconds'] = time.perf_counter() - convert_started
        data['wall_including_normalization_seconds'] = data['wall_seconds'] + data['normalization_seconds']
        data['status'] = 'complete'
    except Exception as exc:
        data['wall_seconds'] = time.perf_counter() - started
        data['status'] = 'timeout' if isinstance(exc, subprocess.TimeoutExpired) else 'failed'
        data['error'] = str(exc)
    data['finished_utc'] = datetime.now(timezone.utc).isoformat()
    dump(record_path, data)
    print(record_path)
    return 0 if data['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
