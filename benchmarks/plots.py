#!/usr/bin/env python3
"""Regenerate SVG/PNG scientific figures from saved measured benchmark records."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/figures'
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

SEEDS = (101, 202, 303)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 180,
                     'svg.fonttype': 'none'})


def source(path):
    try:
        raw = path.read_bytes()
        return json.loads(raw), {'path': str(path.relative_to(ROOT)),
                                'sha256': hashlib.sha256(raw).hexdigest()}
    except (OSError, json.JSONDecodeError) as exc:
        print(f'Skipping pending/updating source {path}: {exc}')
        return None, None


def write_csv(name, rows):
    if rows:
        with (OUT / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def save(fig, stem):
    for extension in ('svg', 'png'):
        fig.savefig(OUT / f'{stem}.{extension}', bbox_inches='tight')
    plt.close(fig)


def scaling_figure():
    value, provenance = source(ROOT / 'results/scaling/iJO1366/summary.json')
    if not value:
        return None
    rows = sorted((r for r in value['rows'] if r['axis'] == 'chains'), key=lambda r: r['chains'])
    if not rows:
        return None
    extracted = [{k: r[k] for k in ('chains', 'tpb', 'draws_per_chain', 'warmup', 'seed',
                 'thinning', 'sampling_seconds', 'native_process_wall_seconds', 'normalization_seconds',
                 'retained_samples_per_sampling_second', 'retained_samples_per_end_to_end_second')} for r in rows]
    write_csv('chain_scaling_source.csv', extracted)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    x = [r['chains'] for r in rows]
    ax.plot(x, [r['retained_samples_per_sampling_second'] for r in rows],
            'o-', color='#0072B2', label='GPU sampling phase')
    ax.plot(x, [r['retained_samples_per_end_to_end_second'] for r in rows],
            's-', color='#D55E00', label='Whole run + adapter')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(x, [str(v) for v in x])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v/1000:g}k'))
    ax.set_yticks([20000, 50000, 100000, 200000, 500000])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('Parallel chains')
    ax.set_ylabel('Retained samples / second')
    ax.set_title('iJO1366 · one H100 · 10,000 retained draws per chain')
    ax.grid(axis='y', which='both', color='#DDDDDD', linewidth=.6)
    ax.legend(frameon=False, loc='upper left', ncol=2, fontsize=8)
    fig.subplots_adjust(bottom=.24, top=.88)
    fig.text(.12, .025, 'Both numerators exclude warmup; both times include 1,000 warmup draws per chain.\n'
             'TPB 128, seed 101, automatic thinning. Whole-run timing includes output adapter overhead.',
             fontsize=8, color='#444444')
    save(fig, 'h100_chain_scaling')
    return {'figure': 'h100_chain_scaling', 'source': provenance, 'complete_cases': len(rows),
            'scope': 'Single-seed throughput sweep; no convergence or saturation claim.'}


def quality_figure():
    value, provenance = source(ROOT / 'results/quality/summary.json')
    if not value:
        return None
    best_cpu, gpus = {}, {}
    for row in value.get('measured_rows', []):
        if row['seed'] not in SEEDS:
            continue
        key = (row['model'], row['seed'])
        if row['variant'].startswith('cpu_'):
            if key not in best_cpu or row['measured_process_wall_seconds'] < best_cpu[key]['measured_process_wall_seconds']:
                best_cpu[key] = row
        elif row['variant'] in ('naja128', 'naja512'):
            gpus[(row['model'], row['seed'], row['variant'])] = row
    selected = [('CPU quality-search variants', r) for r in best_cpu.values()]
    selected += [('Naja TPB ' + r['variant'][4:], r) for r in gpus.values()]
    physical_provenance, physical_best = [], {}
    for summary_path in sorted((ROOT / 'results/native_cpu').glob('*/physical16*/summary.json')):
        physical, physical_source = source(summary_path)
        if not physical or physical.get('status') != 'complete':
            continue
        physical_provenance.append(physical_source)
        for row in physical.get('rows', []):
            if row.get('quality_met') and row.get('bitwise_same_as_canonical'):
                run = json.loads((Path(row['run_dir']) / 'run.json').read_text())
                record = dict(row, model=physical['model'], variant=physical['variant'],
                    measured_process_wall_seconds=row['process_seconds'], draws_per_chain=run['draws'],
                    warmup=run['warmup'], chains=run['chains'], thinning=run['thinning'])
                key = (physical['model'], row['seed'])
                if key not in physical_best or row['process_seconds'] < physical_best[key]['process_seconds']:
                    physical_best[key] = record
    selected += [('CPU 16 physical cores', r) for r in physical_best.values()]
    thinning, thinning_provenance = source(ROOT / 'results/gpu_thinning/iJO1366/summary.json')
    if thinning:
        for group in thinning.get('groups', []):
            if group['thinning'] != 1000 or group.get('status') != 'complete':
                continue
            for row in group['rows']:
                if row.get('quality_met'):
                    selected.append(('Naja TPB512 thin1000', dict(row, model='iJO1366', variant='naja512_thin1000',
                        measured_process_wall_seconds=row['wall_seconds'], warmup=1000, chains=16, thinning=1000)))
    if not selected:
        return None
    order = ('e_coli_core', 'iAB_RBC_283', 'iJR904', 'iJO1366', 'iMM904', 'iAF1260', 'iML1515', 'RECON1', 'Recon3D')
    models = [m for m in order if any(r['model'] == m for _, r in selected)]
    labels = [('CPU quality-search variants', '#D55E00', 'o'), ('Naja TPB 128', '#0072B2', 's'),
              ('Naja TPB 512', '#009E73', '^'), ('CPU 16 physical cores', '#CC79A7', 'D'), ('Naja TPB512 thin1000', '#56B4E9', 'v')]
    labels = [entry for entry in labels if any(method == entry[0] for method, _ in selected)]
    fig, ax = plt.subplots(figsize=(max(9.2, len(models) * 1.9), 5.8))
    extracted, groups = [], []
    for method_index, (method, color, marker) in enumerate(labels):
        offset = (method_index - (len(labels) - 1) / 2) * .2
        for index, model in enumerate(models):
            rows = [r for name, r in selected if name == method and r['model'] == model]
            if not rows:
                continue
            times = [r['measured_process_wall_seconds'] for r in rows]
            middle = statistics.median(times)
            n = len(set(r['seed'] for r in rows))
            x = index + offset
            ax.errorbar(x, middle, yerr=[[middle - min(times)], [max(times) - middle]],
                        marker=marker, color=color, markerfacecolor=color if n == 3 else 'white',
                        markeredgewidth=1.4, markersize=7, capsize=4, linewidth=1.2,
                        label=method if index == next(i for i, m in enumerate(models)
                              if any(name == method and r['model'] == m for name, r in selected)) else None)
            ax.annotate(f'{n}/3', (x, max(times)), xytext=(0, 7), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, color=color)
            groups.append({'model': model, 'method': method, 'passing_seed_count': n,
                           'median_seconds': middle, 'minimum_seconds': min(times), 'maximum_seconds': max(times)})
            for row in rows:
                extracted.append(dict(model=model, method=method, actual_variant=row['variant'], seed=row['seed'],
                    draws_per_chain=row['draws_per_chain'], warmup=row['warmup'], chains=row['chains'],
                    measured_process_wall_seconds=row['measured_process_wall_seconds'],
                    split_rhat_median=row['split_rhat_median'], bulk_ess_median=row['bulk_ess_median'],
                    thinning=row['thinning'], run_dir=row['run_dir']))
    ax.set_xticks(range(len(models)), models, rotation=15, ha='right')
    ax.set_ylabel('Measured passing-budget wall time (seconds)')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_title('Measured time at the selected passing sample budget')
    ax.grid(axis='y', which='both', color='#DDDDDD', linewidth=.6)
    ax.margins(y=.3)
    ax.legend(frameon=False, loc='upper left', ncol=2, fontsize=8)
    fig.subplots_adjust(bottom=.3, top=.88)
    fig.text(.12, .02, 'Dots: median; whiskers: observed seed range; labels: passing seeds / 3. Hollow = partial group.\n'
             'Target: median split-Rhat ≤ 1.01 and median bulk ESS ≥ 100. No point = no passing measured run yet.\n'
             'CPU selection: quality-search variants only; 16-worker runs may pack SMT siblings. Physical-core follow-up shown separately.\n'
             'Includes startup, 1,000 retained-draw warmup, writing and Naja adapter. Bounds are reported separately.',
             fontsize=8, color='#444444')
    write_csv('measured_quality_source.csv', extracted)
    write_csv('measured_quality_groups.csv', groups)
    save(fig, 'measured_quality_time')
    return {'figure': 'measured_quality_time', 'source': provenance, 'physical_cpu_source': physical_provenance, 'gpu_thinning_source': thinning_provenance, 'groups': groups,
            'scope': value.get('scope'), 'selection_caveat': value.get('selection_caveat'),
            'criterion': value['criterion'], 'timing': value['timing'],
            'partial_groups_are_explicit': True, 'interpolated_times_used': False}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = [r for r in (scaling_figure(), quality_figure()) if r is not None]
    (OUT / 'provenance.json').write_text(json.dumps({'generated_utc': datetime.now(timezone.utc).isoformat(),
        'matplotlib_version': matplotlib.__version__, 'figures': records}, indent=2) + '\n')
    print(json.dumps({'figures': len(records), 'output': str(OUT)}))

if __name__ == '__main__':
    main()
