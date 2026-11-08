#!/usr/bin/env python3
"""Persist fingerprints and observed compiler commands for the native wheel."""
import hashlib
import argparse
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--variant',choices=('native','source_generic'),default='native')
args=parser.parse_args()
native=args.variant=='native'
env='env-cpu-native' if native else 'env-cpu-sdist'
wheels='wheels-native' if native else 'wheels-source-generic'
src='source' if native else 'source-generic'
prefix='hopsy_native' if native else 'hopsy_source_generic'
flags='-O3 -march=native -mtune=native' if native else '-O3'
wheel=next((root/wheels).glob('hopsy-1.7.0-*.whl'))
module=next((root/env/'lib/python3.12/site-packages/hopsy').glob('core*.so'))
source=json.loads((root/'audit/hopsy_source.json').read_text())
cube=json.loads((root/'audit'/f'{prefix}_cube.json').read_text())
commands=list((root/src/'hopsy-1.7.0/build').rglob('compile_commands.json'))
observed=[]
if commands:
    data=json.loads(commands[0].read_text())
    observed=[row.get('command',' '.join(row.get('arguments',[]))) for row in data]
    (root/'audit'/f'{prefix}_compile_commands.json').write_text(json.dumps(data,indent=2)+'\n')
    assert all(('-march=native' in row)==native for row in observed)
    assert all('-ffast-math' not in row for row in observed)
result=dict(build_variant=args.variant,hopsy_version='1.7.0',hopsy_build=cube['hopsy_build'],
            hostname=cube['hostname'],source=source,requested_flags=flags,
            compiler='GNU 11.4.0',compiler_target_file=f'audit/{prefix}_compiler_target.txt',
            observed_compile_commands=len(observed),wheel=str(wheel.relative_to(root)),
            wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
            module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
            analytic_cube_status=cube['status'])
(root/'audit'/f'{prefix}_build.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
