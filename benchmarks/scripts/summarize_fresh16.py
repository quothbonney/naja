"""Never turn a fixed-budget failure into a passing validation group."""
import json
from pathlib import Path
import statistics

B = Path(__file__).resolve().parents[1]

def generate():
    base = B / "results/fresh16_validation/iJO1366"
    groups = {}
    for method,label in [("cpu","cpu_native_physical16_thin2000"),("gpu","naja_frozen_tpb512_thin1000")]:
        rows = []
        for seed in (101001,202001,303001):
            path = base / label / f"seed{seed}/validation.json"
            if path.exists():rows.append(json.loads(path.read_text()))
        fixed = [r["fixed"] for r in rows if "fixed" in r]
        passed = [r["seed"] for r in fixed if r["status"] == "PASS"]
        failed = [r["seed"] for r in fixed if r["status"] == "FAIL"]
        complete = len(fixed) == 3
        groups[method] = dict(status="PASS_3_OF_3" if len(passed) == 3 else "FAIL_FIXED_BUDGET" if complete else "PARTIAL",
            tested_fixed_seeds=[r["seed"] for r in fixed],passing_fixed_seeds=passed,failed_fixed_seeds=failed,
            required_seeds=[101001,202001,303001],rows=rows)
    result = dict(schema_version=1,model="iJO1366",chains=16,fixed_groups=groups,
        note="Held-out fresh processes validate the original fixed configurations. Legacy2.73x is wholeprocess timing, not math. Extensions remain separate and cannot rescue a fixed-budget generalization. CPUworker draw includes binding/state-copy; CPUinclusivephase includes IPC; GPUsamplerphase includes allocation/slackGEMM/RNG init, not isolated kernel-only time.")
    paired=[]
    cpu_rows={r['seed']:r for r in groups['cpu']['rows']};gpu_rows={r['seed']:r for r in groups['gpu']['rows']}
    def compare(cpu,gpu):
        return dict(cpu_draws=cpu['draws_per_chain'],gpu_draws=gpu['draws_per_chain'],
            process_ratio_secondary=cpu['process_seconds_secondary']/gpu['process_seconds_secondary'],
            inclusive_sampler_phase_ratio=cpu['sampler_phase_seconds']/gpu['sampler_phase_seconds'],
            cpu_worker_draw_over_gpu_sampler_phase_ratio=cpu['worker_draw_critical_path_seconds']/gpu['sampler_phase_seconds'],
            gpu_over_cpu_worker_ess_rate=gpu['pooled_bulk_ess_per_sampler_phase_second']/cpu['pooled_bulk_ess_per_worker_draw_second'])
    for seed in sorted(set(cpu_rows)&set(gpu_rows)):
        c,g=cpu_rows[seed],gpu_rows[seed]
        if 'fixed' not in c or 'fixed' not in g:continue
        fixed_pass=c['fixed']['quality_met'] and g['fixed']['quality_met']
        entry=dict(seed=seed,fixed_matched_quality=bool(fixed_pass),fixed_comparison=compare(c['fixed'],g['fixed']) if fixed_pass else None)
        ca=c['fixed'] if c['fixed']['quality_met'] else c.get('extension')
        ga=g['fixed'] if g['fixed']['quality_met'] else g.get('extension')
        if not fixed_pass and ca and ga and ca['quality_met'] and ga['quality_met']:
            entry['bounded_extension_comparison']=compare(ca,ga)
            entry['bounded_extension_note']='Separate extension-assisted comparison, not validation of the original fixed budget or its legacy2.73x estimate.'
        paired.append(entry)
    result['paired_seed_comparisons']=paired
    if all(g["status"] == "PASS_3_OF_3" for g in groups.values()):
        cpu = {r["fixed"]["seed"]:r["fixed"] for r in groups["cpu"]["rows"]}
        gpu = {r["fixed"]["seed"]:r["fixed"] for r in groups["gpu"]["rows"]}
        result["fixed_all_three_pass_comparison"] = dict(
            ratio_of_process_medians_cpu_over_gpu=statistics.median(r["process_seconds_secondary"] for r in cpu.values())/statistics.median(r["process_seconds_secondary"] for r in gpu.values()),
            ratio_of_sampler_phase_medians_cpu_over_gpu=statistics.median(r["sampler_phase_seconds"] for r in cpu.values())/statistics.median(r["sampler_phase_seconds"] for r in gpu.values()),
            ratio_of_cpu_worker_draw_over_gpu_sampler_phase_medians=statistics.median(r["worker_draw_critical_path_seconds"] for r in cpu.values())/statistics.median(r["sampler_phase_seconds"] for r in gpu.values()),
            median_paired_gpu_over_cpu_worker_ess_rate=statistics.median(gpu[s]["pooled_bulk_ess_per_sampler_phase_second"]/cpu[s]["pooled_bulk_ess_per_worker_draw_second"] for s in cpu))
    base.mkdir(parents=True,exist_ok=True)
    temp = base / "summary.json.tmp";temp.write_text(json.dumps(result,indent=2)+"\n");temp.replace(base / "summary.json")
    print({k:(v["status"],v["passing_fixed_seeds"],v["failed_fixed_seeds"]) for k,v in groups.items()})
    return result

if __name__ == "__main__":generate()
