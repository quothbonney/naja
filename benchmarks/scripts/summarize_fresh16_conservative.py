#!/usr/bin/env python3
"""Summarize the frozen conservative16-chain protocol and its new seeds."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

B=Path(__file__).resolve().parents[1]
R=B/'results/fresh16_conservative/iJO1366'
protocol=json.loads((R/'protocol.json').read_text())
rows=[]
for seed in protocol['seeds']:
    for method in ('cpu','gpu'):
        path=R/f'seed{seed}'/method/'quality.json'
        if path.exists():rows.append(json.loads(path.read_text()))
groups={}
for method in ('cpu','gpu'):
    selected=[r for r in rows if r['method']==method]
    groups[method]=dict(tested_seeds=len(selected),passing_seeds=sum(r['quality_met'] for r in selected),
        compute_seconds_median=statistics.median(r['sampler_compute_seconds'] for r in selected) if selected else None,
        ess_per_compute_second_median=statistics.median(r['median_ess_per_compute_second'] for r in selected) if selected else None)
paired=[]
for seed in protocol['seeds']:
    selected={r['method']:r for r in rows if r['seed']==seed}
    if len(selected)==2:
        c,g=selected['cpu'],selected['gpu']
        paired.append(dict(seed=seed,both_pass=c['quality_met'] and g['quality_met'],
            cpu_gpu_compute_time_ratio=c['sampler_compute_seconds']/g['sampler_compute_seconds'],
            gpu_cpu_effective_sampling_rate_ratio=g['median_ess_per_compute_second']/c['median_ess_per_compute_second']))
summary=dict(generated_utc=datetime.now(timezone.utc).isoformat(),protocol=protocol,rows=rows,groups=groups,paired=paired,
    complete=len(rows)==6,all_six_pass=len(rows)==6 and all(r['quality_met'] for r in rows),
    scope='Newfixed15kCPU/3kGPUbudgets, noextensions. Conservative478cachedtargetassay. Doesnotretroactivelyrescueold10k/2kfixedfailuresorlegacy2.73x.')
if len(paired)==3:
    summary.update(paired_compute_time_ratio_median=statistics.median(p['cpu_gpu_compute_time_ratio'] for p in paired),
        paired_effective_sampling_rate_ratio_median=statistics.median(p['gpu_cpu_effective_sampling_rate_ratio'] for p in paired))
(R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if rows:
    with (R/'table.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
lines=['# Conservative 16-chain held-out validation', '',
    f"Updated {summary['generated_utc']}. Complete: **{summary['complete']}**; all six pass: **{summary['all_six_pass']}**.", '',
    'Budgets were fixed before these new seeds: CPU 15,000 retained draws/chain at thinning 2,000; '
    'Naja 3,000 retained at thinning 1,000, rotation, TPB 512. Both use 16 chains + 1,000 warmup. '
    'CPU 16 physical workers reserve 32 logical CPUs. New seeds: 444001, 555001, 666001. '
    'The same exact cached 581-D target and conservative 478-exclusion assay apply to both. '
    'No extensions or seed-dependent retuning are allowed.', '',
    '| Method | Seed | Status | Compute s | Median split-Rhat | Median bulk ESS | ESS/compute s |',
    '|---|---:|---|---:|---:|---:|---:|']
for row in rows:
    lines.append(f"| {row['method']} | {row['seed']} | {row['status']} | {row['sampler_compute_seconds']:.4f} | "
        f"{row['median_split_rhat']:.6f} | {row['median_bulk_ess']:.1f} | {row['median_ess_per_compute_second']:.2f} |")
if len(paired)==3:
    lines += ['',f"Median paired CPU/GPU compute-time ratio: **{summary['paired_compute_time_ratio_median']:.3f}×**; "
        f"GPU/CPU ESS-rate ratio: **{summary['paired_effective_sampling_rate_ratio_median']:.3f}×**."]
lines += ['', 'Quality criterion:median classical split-Rhat≤1.01,median pooled rank-normalized bulkESS≥100, '
    'zero unmasked bound violations at1e-7. CPU timing is max accumulated draw-call critical path '
    '(binding/state-copy/descheduling included;IPC/output excluded). GPU uses native sampler phase '
    '(allocation/slack/RNG setup included;download/write excluded).', '',
    'These larger fixed budgets were chosen from earlier bounded-extension evidence. '
    'Original10kCPU/2kGPUfixed failures remain visible separately; this study does not rescue '
    'their fixed-budget generalization or retroactively validate the legacy2.73× claim.', '',
    '[Frozen protocol](protocol.json) · [Source runs and comparisons](summary.json) · [CSV](table.csv).', '']
(R/'VALIDATION.md').write_text('\n'.join(lines))
print(R/'VALIDATION.md')
