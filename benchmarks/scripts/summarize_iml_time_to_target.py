"""Count actually executed iML fixed-budget screens; never count missing seeds."""
import json
from pathlib import Path

B=Path(__file__).resolve().parents[1]

def generate():
    rows=[]
    root=B/'results/larger_quality/iML1515/cpu2000/seed101'
    if (root/'pilot/run.json').exists() and (root/'pilot/rhat-precheck.json').exists():
        run=json.loads((root/'pilot/run.json').read_text())
        checks=json.loads((root/'pilot/rhat-precheck.json').read_text())['prefixes']
        check=next((c for c in checks if c['shape'][1]==10000),None)
        if run['status']=='complete' and check:
            rows.append(dict(seed=101,tested=True,quality_met=False,status='Rhat_fail_at_fixed_cap',draws=10000,warmup=1000,thinning=2000,
                split_rhat_median=check['split_rhat_median'],bulk_ess_median=None,bulk_ess_computed=False,
                sample_bound_violation_fraction=check['sample_bound_violation_fraction'],bounds_tolerance=check['bounds_tolerance'],
                sampler_phase_seconds=run['timings']['sample_seconds'],run_dir=str(root/'pilot'),source=str(root/'quality.json')))
    for seed in (202,303):
        p=B/f'results/iml_time_to_target/cpu_native_thin2000/seed{seed}/quality.json'
        if not p.exists():continue
        q=json.loads(p.read_text());path=Path(q['run_dir']);run=json.loads((path/'run.json').read_text())
        if run['status']!='complete':continue
        check=q['precheck'];m=q.get('metrics',{})
        rows.append(dict(seed=seed,tested=True,quality_met=q['quality_met'],status=q['status'],draws=10000,warmup=1000,thinning=2000,
            split_rhat_median=m.get('split_rhat_median',check['split_rhat_median']),bulk_ess_median=m.get('bulk_ess_median'),bulk_ess_computed=bool(m),
            sample_bound_violation_fraction=check['sample_bound_violation_fraction'],bounds_tolerance=check['bounds_tolerance'],
            sampler_phase_seconds=run['timings']['sample_seconds'],run_dir=str(path),source=str(p)))
    tested={r['seed'] for r in rows};passed=[r['seed'] for r in rows if r['quality_met']]
    guideline=[]
    for seed in (101,202,303):
        p=B/f'results/larger_quality/iML1515/cpujade/seed{seed}/quality.json'
        if p.exists():
            q=json.loads(p.read_text())
            if q.get('status')=='complete':guideline.append(dict(seed=seed,source=str(p),**q['measured']))
    f64=B/'results/precision_quality/iML1515/summary.json'
    result=dict(schema_version=1,model='iML1515',cpu_thin2000_fixed_cap=dict(draws=10000,warmup=1000,thinning=2000,
        tested_count=len(rows),passing_count=len(passed),passing_seeds=passed,failed_seeds=[r['seed'] for r in rows if not r['quality_met']],
        missing_seeds=[s for s in (101,202,303) if s not in tested],rows=rows,
        note='Only actually completed fixed10k pilots count. A cap failure does not imply failure to converge at larger budgets. ESS is omitted when Rhat already fails the AND target.'),
        cpu_guideline_passing_rows=guideline,verified_f64_gpu_summary=str(f64) if f64.exists() else None)
    out=B/'results/iml_time_to_target';out.mkdir(parents=True,exist_ok=True)
    temp=out/'summary.json.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(out/'summary.json')
    print('iMLthin2000',len(passed),'passed /',len(rows),'actually tested; missing',result['cpu_thin2000_fixed_cap']['missing_seeds'])
    return result

if __name__=='__main__':generate()
