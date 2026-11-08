#!/usr/bin/env python3
"""Summarize every predeclared rescue control, retaining incomplete/failed cases."""
from collections import Counter
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics

B=Path(__file__).resolve().parents[1]
R=B/'results/conditioned_rescue_controls'
def read(path):return json.loads(path.read_text()) if path.exists() else {}
p=read(R/'protocol.json');closed=read(R/'closeout_stop.json');rows=[];pairs=[];states=[];counts=Counter()
for case in p['cases']:
    folder=R/case['gene'];capture=read(folder/'capture.json');pair=read(folder/'paired_quality.json')
    state=dict(gene=case['gene'],draws=case['draws'],capture_status=capture.get('status','pending'),
        metrics_status=pair.get('status','pending'),order=case['order'])
    if closed and pair.get('status')!='complete':
        state.update(closeout_status='cancelled_by_user_directive',metrics_status='cancelled_closeout')
    states.append(state)
    for mode in ('off','rotate'):
        row=read(folder/mode/'quality.json')
        if row:rows.append(row)
    if pair.get('status')=='complete':
        modes={row['mode']:row for row in pair['rows']};off,rot=modes['off'],modes['rotate']
        label='both_pass' if off['quality_met'] and rot['quality_met'] else 'rotate_only' if rot['quality_met'] else 'off_only' if off['quality_met'] else 'neither_pass'
        counts[label]+=1
        pairs.append(dict(gene=case['gene'],draws=case['draws'],category=label,
            off_rhat=off['median_split_rhat'],rotate_rhat=rot['median_split_rhat'],
            off_ess=off['median_bulk_ess'],rotate_ess=rot['median_bulk_ess'],
            off_sampler_seconds=off['sampler_seconds'],rotate_sampler_seconds=rot['sampler_seconds'],
            rotate_correction_seconds=rot['correction_seconds'],
            off_bound_violation_fraction=off['sample_bound_violation_fraction'],rotate_bound_violation_fraction=rot['sample_bound_violation_fraction'],
            off_mass_balance_max=off['maximum_stoichiometric_residual'],rotate_mass_balance_max=rot['maximum_stoichiometric_residual'],
            gpu_uuid=rot['gpu_uuid'],order=case['order']))
summary=dict(generated_utc=datetime.now(timezone.utc).isoformat(),declared_count=len(p['cases']),
    completed_pairs=len(pairs),complete=len(pairs)==len(p['cases']),
    pass_contingency=dict(counts),states=states,rows=rows,pairs=pairs,protocol=str(R/'protocol.json'),
    closed_by_user=bool(closed),closeout=closed,
    scope=p['scope'],note='Posthoc original-category controls at original selected budgets. Same GPU/binary/target/seed per pair; no retuning, no representative-population or globally minimized time-to-quality claim.')
(R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if rows:
    with (R/'source.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
if pairs:
    with (R/'paired.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
lines=['# Targeted same-GPU off/rotate controls', '',
    f"Updated {summary['generated_utc']}; completed pairs **{len(pairs)}/13**.", '',
    'All 13 original conditions where the Naja+rotation pipeline passed and inherited CPU failed '
    'the 50k cap were declared before these controls. These are targeted posthoc category controls, '
    'not a representative panel or new independent validation seeds. Each pair uses the original '
    'selected Naja retained budget, seed101, 16 chains, 1,000 warmup, thinning1,000, TPB512 and '
    'the same verified F64 binary and conditional model/mask. Only the barrier flag changes. '
    'Pair order alternates; CUDA physical UUID is checked within each allocation. '
    'All cases, including new off passes and rotation failures, remain in the denominator.', '',
    f"Completed-pair contingency: {dict(counts)}. Unfinished: {13-len(pairs)}.", '',
    '| Gene KO | Retained/chain | Off Rhat | Rotate Rhat | Off ESS | Rotate ESS | Outcome |',
    '|---|---:|---:|---:|---:|---:|---|']
for case in p['cases']:
    row=next((v for v in pairs if v['gene']==case['gene']),None)
    if row:
        lines.append(f"| {row['gene']} | {row['draws']:,} | {row['off_rhat']:.5f} | {row['rotate_rhat']:.5f} | {row['off_ess']:.1f} | {row['rotate_ess']:.1f} | {row['category']} |")
    else:lines.append(f"| {case['gene']} | {case['draws']:,} | — | — | — | — | {'cancelled closeout' if closed else 'unfinished'} |")
if closed:
    lines += ['', 'The user closed this extension after its first completed pilot. The remaining 12 planned '
        'pairs were cancelled, preserving partial files. They are not sampler failures and no full13-case '
        'conclusion is supported. No further controls will launch from these arrays.', '']
lines += ['', 'Pass gate: median classical split-Rhat≤1.05, pooled median bulkESS≥100, '
    'zero original/conditional-bound violations at1e-7 and maximum mass-balance residual≤1e-7. '
    'Compute timings use native GPU sampler phases; correction and secondary process times are separate. '
    'Only full-budget endpoints are diagnosed; these controls do not estimate time-to-quality.', '',
    '[Frozen selection/protocol](protocol.json) · [Complete source/status JSON](summary.json) · '
    '[Per-mode CSV](source.csv) · [Paired CSV](paired.csv) · [Exact Slurm commands](jobs.json).', '']
(R/'README.md').write_text('\n'.join(lines))
print(R/'README.md')
