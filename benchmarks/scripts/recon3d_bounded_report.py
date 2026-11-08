#!/usr/bin/env python3
"""Human-readable state/result for the one-hour-bounded Recon3D attempt."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

B=Path(__file__).resolve().parents[1]
R=B/'results/closeout/recon3d_bounded'
def read(path):return json.loads(path.read_text()) if path.exists() else {}
p=read(R/'protocol.json');c=read(R/'capture.json');q=read(R/'summary.json');decision=read(R/'pilot_decision.json');jobs=read(R/'jobs.json')
if jobs:
    accounting=subprocess.run(['sacct','-n','-X','-P','-j',jobs['capture_job']+','+jobs['metrics_job'],
        '--format=JobID,JobName,State,ElapsedRaw,AllocCPUS,AllocTRES'],capture_output=True,text=True)
    if accounting.returncode==0:(R/'slurm_accounting.psv').write_text(accounting.stdout)
status=q.get('status',c.get('status','pending'))
lines=['# Bounded Recon3D sampling attempt','',f"Updated {datetime.now(timezone.utc).isoformat()}. Status: **{status}**.",'',
    'One H100 allocation, limited to59minutes, containing one pilot and one main run; no retries. '
    'CPU normalization, convergence diagnostics and all-point feasibility checks run after GPU release. '
    'The cached target has n=10,600 reactions,d=4,861 dimensions,m=18,477 inequalities. '
    'It uses PolyRound-library equality elimination but **no iterative MVE rounding**. '
    'No matched-quality CPU sampling speedup or global time-to-quality optimum is claimed.','',
    '| Configuration | Value |','|---|---|',
    '| Sampler |512 chains,F64,barrier whitening,GPU A×Q,TPB128,seed101|',
    '| Pilot |64warmup+64retained draws/chain,thinning810|',
    '| Main |1,000warmup+800retained draws/chain|',
    f"| Main thinning |{decision.get('selected_thinning','pending')}|",
    '| Quality gate |Median classical split-Rhat≤1.01,median pooled bulkESS≥100,zero bound violations at1e-7,mass residual≤1e-7|','',
    'All10,600 reactions are active in the assay; no nearconstant/LP exclusions are used. '
    'Only exactly constant sampled reactions are omitted by the frozen estimator. '
    'Original-target FVA has minimum reaction width0.2. The2,723 pruned objects were strictly inactive '
    'inequality rows, not reactions. Cached equality elimination took128.8417s; MVE0s; '
    'total original cached preprocessing287.7203s includes FVA134.2945s.','',
    f"Source commit: `{p['source_commit']}`; binary SHA256: `{p['binary_sha256']}`. "
    'This FCC executable is immutable. It explicitly selects the optimized GPU rotation backend; '
    'the finalC9release changes the compatibility default, not this recorded run. '
    'Production wide-index/FP64 tests were passed before launch; no legacy experimental bit-parity flag was fabricated.','']
if decision:
    lines += [f"Pilot GPU sampler: {decision['pilot_sampler_seconds']:.4f}s; coordinate lag1 median "
        f"{decision['lag1']['median']:.6f}. The frozen rule requested thinning{decision['desired_thinning']:,} "
        f"and capped it at{decision['selected_thinning']:,} to fit the GPU budget. "
        'This is a pilot-trained attempt, not an independent confirmatory replicate.','']
if 'median_split_rhat' in q:
    endpoint=q.get('endpoint','main')
    lines += [f"| Measured {endpoint} endpoint | Value |",'|---|---:|',
        f"| GPU sampler s |{q['sampler_seconds']:.6f}|",f"| Barrier phase s |{q['barrier_seconds']:.6f}|",
        f"| Median split-Rhat |{q['median_split_rhat']}|",f"|95th percentile split-Rhat |{q['p95_split_rhat']}|",
        f"| Median pooled bulkESS |{q['median_bulk_ess']}|",f"| Sample bound-violation fraction |{q['sample_bound_violation_fraction']}|",
        f"| Maximum original-bound violation |{q['maximum_absolute_bound_violation']}|",
        f"| Maximum mass-balance residual |{q['mass_balance']['maximum_stoichiometric_residual']}|",
        f"| Reactions diagnosed |{q['diagnosed_reactions']:,}/10,600|",
        f"| Exactly constant sampled reactions |{q['exactly_constant_sampled_reactions']:,}|",
        f"| Retained points checked |{q['mass_balance']['points_checked']:,}|",'']
    lines += [f"All values above come from this actual {endpoint} execution. The sampler includes discarded warmup. "
        'Barrier, output, normalization and CPU diagnostics are separate; no prefix wall-time interpolation is used.','']
    if not q.get('all_reactions_diagnosed',False):
        lines += ['The endpoint does not establish all-reaction convergence: the frozen estimator excluded exactly constant sampled reactions whose original-target FVA ranges are positive.','']
    if endpoint=='pilot':
        lines += ['The main run did not complete. These fallback diagnostics use only the saved64-retained-draw pilot with64 warmup draws, thinning810; they do not describe the planned800-draw endpoint.','']
elif q:
    lines += ['No completed, valid convergence endpoint is available. See the captured failure/budget reason in summary.json.','']
else:
    lines += ['Main-run quality remains pending; no convergence claim is assigned until CPU diagnostics finish.','']
lines += ['[Frozen protocol](protocol.json) · [Main result](summary.json) · [Capture state](capture.json) · '
    '[Pilot decision](pilot_decision.json) · [Exact jobs/commands](jobs.json) · [Resource accounting](slurm_accounting.psv).','']
(R/'RECON3D_ATTEMPT.md').write_text('\n'.join(lines))
print(R/'RECON3D_ATTEMPT.md')
