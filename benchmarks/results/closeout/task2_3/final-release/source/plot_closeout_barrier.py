#!/usr/bin/env python3
"""Recorded barrier-phase scaling, with historical CPU stage explicitly separated."""
from pathlib import Path
import json,argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=B/'results/closeout/task2_3');a=p.parse_args();R=a.out;x=json.loads((R/'task2_barrier_summary.json').read_text());rows=sorted(x['rows'],key=lambda r:r['d']);d=[r['d'] for r in rows];gpu=[r['median_barrier_rotation_s'] for r in rows];cpu=[r['original_CPU_round_component_s'] for r in rows]
fig,ax=plt.subplots(figsize=(6.5,3.8));ax.errorbar(d,gpu,yerr=[[r['median_barrier_rotation_s']-r['minimum_barrier_rotation_s'] for r in rows],[r['maximum_barrier_rotation_s']-r['median_barrier_rotation_s'] for r in rows]],color='#2463a0',marker='o',capsize=2,lw=1.3,label='Additional barrier phase (H100)');ax.plot(d,cpu,color='#777777',marker='s',ls='--',lw=1.2,label='Original PolyRound round stage (CPU)');ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Reduced dimension');ax.set_ylabel('Recorded phase wall time (s)');ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8);fig.tight_layout()
for ext in ('svg','pdf','png'):fig.savefig(R/('task2_barrier_dimension_scaling.'+ext),dpi=180)
