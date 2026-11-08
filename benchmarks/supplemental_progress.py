"""Render exploratory progress without adding runs to the canonical table."""
import json
import statistics


def sections(root):
    lines = []
    path = root / 'results/larger_quality/summary.json'
    if path.exists():
        data = json.loads(path.read_text())
        lines += ['', '## Additional models: measured passing budgets', '',
                  '[Source CSV](larger_quality/table.csv) · [Groups and attempts](larger_quality/summary.json). '
                  'Separate exploratory groups: 16 chains, 1,000 discarded retained warmup draws; '
                  'GPU TPB512/thinning1,000; native CPU on 16 physical cores with 32 logical CPUs reserved. '
                  'Sampling walls include startup, load, warmup, output and GPU normalization; '
                  'preprocessing, pilot search and diagnostics are recorded separately. '
                  'The shared quality target is median split-Rhat ≤1.01 and median bulk ESS ≥100. '
                  'Bound compliance is reported independently. Partial groups are not three-seed speedup evidence.', '',
                  '| Model | Method | Passing seeds | Median wall s | Median Rhat | Median bulk ESS | Max sample bound fraction at 1e-7 | Max bound residual |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for group in data.get('groups', []):
            rows = group.get('rows', [])
            values = [group['model'], group['method'], f"{len(rows)}/3"]
            if rows:
                values += [f"{statistics.median(r[key] for r in rows):.6g}" for key in
                           ['wall_seconds', 'split_rhat_median', 'bulk_ess_median']]
                values += [f"{max(r[key] for r in rows):.6g}" for key in
                           ['sample_bound_violation_fraction', 'max_absolute_bound_violation']]
            else:
                values += ['—'] * 5
            lines.append('| ' + ' | '.join(values) + ' |')
        lines += ['', 'Medians for partial groups describe only their completed seeds. '
                  'In particular, small iMM904 numerical bound residuals must not be described as '
                  'zero violations at 1e-7, even when convergence diagnostics pass.', '']
        for pair in data.get('comparisons', []):
            lines += [f"{pair['model']}: GPU median {pair['gpu_median_seconds']:.6g} s, "
                      f"fastest completed CPU group {pair['cpu_best_complete_group']} median "
                      f"{pair['cpu_group_median_seconds']:.6g} s; median paired fastest-tested "
                      f"CPU/GPU ratio {pair['median_paired_fastest_tested_cpu_ratio']:.4g}×. "
                      'Three seeds per included group; bounds remain a separate column.', '']
    path = root / 'results/high_throughput/iJO1366/summary.json'
    if path.exists():
        data = json.loads(path.read_text())
        lines += ['', '## 512-chain quality throughput: independent seeds', '',
                  '[Validated results and all attempts](high_throughput/iJO1366/summary.json). '
                  'GPU and CPU choose retained budgets independently under the same quality target. '
                  'The primary CPU group reserves all 128 logical CPUs and uses 64 distinct physical cores. '
                  'Replicate seed ranges do not overlap. Failed pilots remain in the source; '
                  'no CPU/GPU ratio is inferred from incomplete groups.', '',
                  '| Method | Passing seeds | Group status | Median wall s | Median pooled bulk ESS/s |',
                  '|---|---:|---|---:|---:|']
        for method, group in data.get('groups', {}).items():
            rows = group.get('rows', [])
            wall = group.get('process_seconds_median')
            ess_rate = group.get('pooled_bulk_ess_median_per_process_second_median')
            lines.append('| ' + ' | '.join([method, f'{len(rows)}/3', group['status'],
                f'{wall:.6g}' if wall is not None else '—',
                f'{ess_rate:.6g}' if ess_rate is not None else '—']) + ' |')
        lines += ['', data['note'], '']
    return lines
