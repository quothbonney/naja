"""Plot finite-cap success and paired preparation-reuse costs, with all denominators."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = Path(__file__).resolve().parents[1]
O = B/'results/knockout_panel/overnight_audit'
data = json.loads((O/'summary.json').read_text())
groups = {g['variant']: g for g in data['groups']}
pairs = [p for p in data['paired_comparisons'] if p['reference'] == 'global_cpu']
variants = ['global_cpu', 'stale_cpu', 'naja_rotate']
labels = ['Full reround\n+ CPU', 'Inherited\n+ CPU', 'Corrected\nNaja']
fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.1), layout='constrained')
counts = [groups[v]['passed'] for v in variants]
left.bar(range(3), counts, color=['#777777', '#8eaca6', '#2365a3'], width=.6)
left.set(xticks=range(3), xticklabels=labels, ylim=(0, 55),
         ylabel='Targets passing / 50')
for i, n in enumerate(counts):
    left.text(i, n+1, str(n), ha='center', va='bottom', fontsize=9)
left.axhline(50, color='#cccccc', ls=':', lw=1)
x = np.array([p['cpu_incremental_preparation_plus_sampler_seconds'] for p in pairs])
y = np.array([p['naja_incremental_preparation_plus_sampler_seconds'] for p in pairs])
low, high = min(x.min(), y.min())*.75, max(x.max(), y.max())*1.3
right.plot([low, high], [low, high], color='#999999', lw=1, ls='--')
right.scatter(x, y, s=19, color='#2365a3', alpha=.75, edgecolors='white', linewidth=.35)
right.set(xscale='log', yscale='log', xlim=(low, high), ylim=(low, high),
          xlabel='Full reround + CPU sampler (s)', ylabel='Correction + Naja sampler (s)')
right.set_aspect('equal', adjustable='box')
for ax in (left, right):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
for suffix in ('svg', 'pdf', 'png'):
    fig.savefig(O/f'conditioned_panel_summary.{suffix}', dpi=200)
with (O/'figure_source.csv').open('w') as f:
    fields = ['gene', 'cpu_incremental_preparation_plus_sampler_seconds',
              'naja_incremental_preparation_plus_sampler_seconds',
              'cpu_over_gpu_incremental_preparation_plus_sampler_time',
              'cpu_over_gpu_time_with_common_conditional_geometry']
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader();w.writerows(pairs)
(O/'figure_definitions.json').write_text(json.dumps(dict(
    source='summary.json', protocol=data['protocol'], denominator=data['denominator_note'],
    success_counts={v: groups[v] for v in variants}, paired_target_count=len(pairs),
    paired_median_incremental_preparation_and_sampler_ratio=float(np.median(x/y)),
    right_panel_scope='Common passing subset. Starts from shared ready conditional geometry; CPU MVE algorithm plus worker draw critical path versus barrier correction plus GPU sampler. Excludes transfers/output, imports, validation, masks and one-time base preparation. Common geometry cost separately reported in figure_source.csv. No population inference or sampling-kernel speedup.'), indent=2)+'\n')
