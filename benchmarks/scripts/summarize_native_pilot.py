#!/usr/bin/env python3
"""Collect the three matched one-seed iJO1366 build-comparison pilots."""
import csv
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
rows=[]
for variant,label in [('wheel','2000_w16_wheelcontrol'),
                      ('source_generic','2000_w16_sourcegeneric'),('native','2000_w16')]:
    path=root/'results/native_cpu/iJO1366'/label/'seed101'
    run=json.loads((path/'run.json').read_text())
    metrics=json.loads((path/'metrics.json').read_text())
    rows.append(dict(variant=variant,build=run['hopsy_build'],
        flags=run['hopsy_cxxflags'],model='iJO1366',seed=101,chains=16,draws=10000,
        warmup=1000,thinning=2000,workers=16,
        process_seconds=run['process_wall_seconds'],sample_seconds=run['timings']['sample_seconds'],
        split_rhat_median=metrics['split_rhat_median'],split_rhat_p95=metrics['split_rhat_p95'],
        bulk_ess_median=metrics['bulk_ess_median'],
        bound_violation_fraction=metrics['sample_bound_violation_fraction'],
        quality_met=metrics['split_rhat_median']<=1.01 and metrics['bulk_ess_median']>=100,
        environment_signature=run['environment_signature']))
generic,native=rows[1:]
result=dict(pilot_only=True,seeds=[101],rows=rows,
            matched_source_compiler_runtime_for=['source_generic','native'],
            native_process_speedup_vs_source_generic=generic['process_seconds']/native['process_seconds'],
            native_sampling_speedup_vs_source_generic=generic['sample_seconds']/native['sample_seconds'],
            note='One seed; build optimization comparison. Quality estimates near the threshold vary between floating-point trajectories; three-seed retiming is required for a fastest-CPU claim.')
out=root/'results/native_cpu'
(out/'pilot_summary.json').write_text(json.dumps(result,indent=2)+'\n')
with (out/'pilot_table.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(result,indent=2))
