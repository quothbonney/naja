#!/usr/bin/env python3
"""Analytic uniform-cube smoke check for a separately compiled hopsy module."""
import json
import argparse
import os
from pathlib import Path
import platform
import hopsy
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--output')
args=parser.parse_args()

dim=3
problem=hopsy.Problem(np.vstack((np.eye(dim),-np.eye(dim))),np.ones(2*dim))
chains=[hopsy.MarkovChain(problem,proposal=hopsy.UniformCoordinateHitAndRunProposal,
                         starting_point=np.zeros(dim)) for _ in range(4)]
_,samples=hopsy.sample(chains,[hopsy.RandomNumberGenerator(9321+i) for i in range(4)],
                       n_samples=11000,thinning=10,n_procs=1)
samples=samples[:,1000:,:]
means=samples.mean(axis=(0,1)); variances=samples.var(axis=(0,1))
passed=bool(np.max(np.abs(means))<.03 and np.max(np.abs(variances-1/3))<.03
            and np.max(np.abs(samples))<=1+1e-12)
result=dict(status='passed' if passed else 'failed',hostname=platform.node(),
            hopsy_version=hopsy.__version__,hopsy_build=hopsy.__build__,
            flags=os.environ.get('HOPSY_CXXFLAGS'),chains=4,draws=10000,
            warmup=1000,thinning=10,means=means.tolist(),variances=variances.tolist(),
            expected_mean=0,expected_variance=1/3,moment_absolute_tolerance=.03,
            max_absolute_coordinate=float(np.max(np.abs(samples))))
root=Path(__file__).resolve().parents[1]
output=Path(args.output) if args.output else root/'audit/hopsy_native_cube.json'
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
raise SystemExit(not passed)
