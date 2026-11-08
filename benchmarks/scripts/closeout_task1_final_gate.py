"""Final compatibility-default re-gate; immutable FCC export evidence stays separate."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import os
import subprocess
import closeout_task1_verify as audit

def main():
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    old=audit.R;out=old/'final_compatibility_default';out.mkdir(parents=True,exist_ok=True)
    audit.R=out;audit.EXE=a.build.resolve()/'naja'
    actual=subprocess.check_output(['git','-C',str(audit.B/'closeout-release'),'rev-parse','HEAD'],text=True).strip()
    assert actual.startswith(a.commit),(actual,a.commit)
    command=[str(audit.B/'env-gpu/bin/ctest'),'--test-dir',str(a.build.resolve()),'--output-on-failure',
        '-R','^(test_gpu_(sampling_feasibility|sampling_cube_moments|sampling_simplex_moments|backmap_correctness|thin_polytope|index_width)|test_large_sizes_and_npy)$']
    with (out/'gpu-correctness-tests.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    (out/'tests-passed.txt').write_text(datetime.now(timezone.utc).isoformat()+'\n')
    audit.negative_probes()
    reference=audit.read(old/'iJO_reference_seed101/verified_layout.json')
    previous_gpu=audit.read(old/'iJO_default_gpu_rotation/verified_layout.json')
    default=audit.run_case('iJO_default_cpu_rotation','iJO1366',binary=audit.EXE)
    explicit_gpu=audit.run_case('iJO_explicit_gpu_rotation','iJO1366',binary=audit.EXE,backend='gpu')
    assert default['profile']['config']['barrier_rotate_backend']=='cpu'
    assert explicit_gpu['profile']['config']['barrier_rotate_backend']=='gpu'
    default_parity=audit.parity(reference,default)
    gpu_parity=audit.parity(previous_gpu,explicit_gpu)
    assert default_parity['bitwise_equal'],'Final default must preserve legacy F32 trajectory'
    assert gpu_parity['bitwise_equal'],'Explicit GPU backend changed from the verified FCC implementation'
    short=audit.run_case('host_prefault_auto64','iJO1366',binary=audit.EXE,dtype='float64',chains=64,samples=4,thin=1)
    allowed=sorted(os.sched_getaffinity(0));effective=short['profile']['config']['host_prefault_threads']
    assert 0<effective<=min(8,len(allowed))
    result=dict(status='passed',git_commit=actual,binary_sha256=audit.sha(audit.EXE),
        correctness_tests=7,negative_probes=5,default_cpu_parity=default_parity,explicit_gpu_parity=gpu_parity,
        auto_prefault=dict(effective_threads=effective,allowed_cpu_ids=allowed),
        reference_scope='Public23873f0 plus saved one-line constant-seed101 shim; CPU-default path is compared exactly. Explicit GPU backend is compared with immutable FCC GPU results, not claimed identical to CPU reference.',
        unchanged_export_evidence=str(old/'summary.json'),
        source_sha256=audit.sha(Path(__file__)),utc=datetime.now(timezone.utc).isoformat())
    audit.save(out/'summary.json',result)
    (out/'FINAL_GATE.md').write_text('\n'.join(['# Final compatibility-default gate','',
        f"Commit `{actual}`, binary SHA256 `{result['binary_sha256']}`.",'',
        '- All seven correctness/index/FP64 tests passed.',
        '- All five real failure probes returned nonzero without writing samples.',
        '- Default CPU rotation preserves every iJO1366 F32 bit against the public reference plus seed-only shim.',
        '- Explicit GPU rotation preserves every bit of the prior FCC GPU trajectory; it remains a distinct numerical path from CPU rotation.',
        f'- Automatic prefault resolves {effective} threads within the {len(allowed)}-CPU allocation on the short64-chain F64 smoke.',
        '', 'The separately saved FCC RECON1 four-export study remains applicable to the explicit GPU path. At1e-7 it records nonzero original-bound violations from the cached geometry; F64 export alone does not repair them.', '',
        '[Full summary](summary.json) · [CTest output](gpu-correctness-tests.log) · [Failure probes](negative_probes.json).','']))
    print(result)

if __name__=='__main__':main()
