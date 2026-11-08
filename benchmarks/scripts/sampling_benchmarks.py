"""Refresh output-excluded16-chain sampling phases from completed quality runs."""
import csv
import fcntl
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics

B = Path(__file__).resolve().parents[1]
SEEDS = (101,202,303)

def read(path):
    return json.loads(path.read_text())

def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else B.parent/path

def allocation(path,run,q):
    for directory in (path,path.parent):
        proof = directory/"physical_affinity.json"
        if proof.exists():
            a = read(proof)
            chosen = set(a.get("selected_logical_cpus",a.get("actual_python_affinity",[])))
            topology = a.get("topology",[])
            cores = len({(r["socket"],r["core"]) for r in topology if r["cpu"] in chosen}) if topology else a.get("selected_physical_core_count")
            return cores,a.get("reserved_logical_cpus"),str(proof)
    cpus = run.get("allocated_cpus") or q.get("slurm",{}).get("SLURM_CPUS_PER_TASK")
    return None,int(cpus) if cpus else None,None

def row(model,config,seed,path,q):
    run,m = read(path/"run.json"),read(path/"metrics.json")
    if m.get("schema_version") == 2 and (m.get("sample_bound_violation_fraction") or 0) > 0:
        raise ValueError(f"bounds_gate_failed: samplefraction={m['sample_bound_violation_fraction']}, maxabsolute={m.get('max_absolute_bound_violation')}, tolerance={m.get('bounds_tolerance')}, run={path}")
    if not (run["status"] == "complete" and run["chains"] == 16 and m.get("schema_version") == 2
            and m.get("diagnostic_mask_status") == "model_lp_verified" and m.get("split_rhat_median") is not None
            and m["split_rhat_median"] <= 1.01 and m.get("bulk_ess_median") is not None and m["bulk_ess_median"] >= 100
            and m.get("bounds_tolerance") == 1e-7 and m.get("sample_bound_violation_fraction") == 0):
        return None
    method = "CPU" if run["method"] == "hopsy_chrr" else "GPU"
    phase = run["timings"]["sample_seconds"] if method == "CPU" else run["profile"]["timing"]["gpu_sampling_s"]
    thin = run["thinning"] if method == "CPU" else run["profile"]["config"]["thinning"]
    total = run["chains"]*(run["draws"]+run["warmup"])*thin
    cores,logical,allocation_proof = allocation(path,run,q) if method == "CPU" else (None,None,None)
    return dict(model=model,method=method,config=config,seed=seed,chains=16,draws_per_chain=run["draws"],
        warmup_per_chain=run["warmup"],thinning=thin,sampler_phase_seconds=phase,
        pooled_bulk_ess_median=m["bulk_ess_median"],pooled_bulk_ess_per_sampler_phase_second=m["bulk_ess_median"]/phase,
        transitions_including_warmup=total,pooled_bulk_ess_per_total_transition=m["bulk_ess_median"]/total,
        retained_transitions=run["chains"]*run["draws"]*thin,
        split_rhat_median=m["split_rhat_median"],sample_bound_violation_fraction=m["sample_bound_violation_fraction"],
        bounds_tolerance=m["bounds_tolerance"],max_absolute_bound_violation=m.get("max_absolute_bound_violation"),
        diagnostic_mask_status=m["diagnostic_mask_status"],geometry_sha256=m.get("diagnostic_mask_geometry_sha256"),
        workers=run.get("workers"),verified_physical_cores=cores,reserved_logical_cpus=logical,
        allocation_proof=allocation_proof,build_variant=run.get("build_variant"),binary_or_module_sha256=run.get("hopsy_module_sha256",run.get("exe_sha256")),
        sample_dtype=run.get("sample_dtype",run.get("dtype")),
        process_seconds_secondary=m.get("wall_seconds"),run_dir=str(path),
        phase_scope="Inclusive hopsy.sample: pool startup, pickling, IPC and array assembly included; not pure arithmetic." if method == "CPU" else "GPU sampler phase includes allocation, initial slack GEMM/RNG setup and CHRR sampling; not isolated CHRR kernel time. Download/output/normalization excluded.")

def generate():
    lock=(B/'results/.sampling_benchmarks.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    candidates = {}
    rejected = []
    def add(model,config,seed,path,q):
        try:
            r = row(model,config,seed,path,q)
            if r is not None:
                candidates.setdefault((model,r["method"],config),{})[seed] = r
        except (OSError,KeyError,ValueError,TypeError) as error:
            rejected.append(dict(model=model,config=config,seed=seed,reason=str(error)))
    for family in ("quality","larger_quality"):
        for quality in (B/"results"/family).glob("*/*/seed*/quality.json"):
            try:
                q = read(quality)
                if q.get("status") != "complete" or not q.get("measured",{}).get("quality_met"):
                    continue
                seed = int(quality.parent.name.removeprefix("seed"))
                if seed in SEEDS:
                    add(quality.parents[2].name,f"{family}:{quality.parents[1].name}",seed,resolve(q["measured"]["run_dir"]),q)
            except (OSError,KeyError,ValueError,TypeError) as error:
                rejected.append(dict(source=str(quality),reason=str(error)))
    for summary in (B/"results/native_cpu").glob("*/physical16*/summary.json"):
        try:
            s = read(summary)
            if s.get("status") == "complete":
                for r in s["rows"]:
                    if r.get("quality_met") and r["seed"] in SEEDS:
                        add(summary.parent.parent.name,"native_cpu:"+summary.parent.name,r["seed"],resolve(r["run_dir"]),r)
        except (OSError,KeyError,ValueError,TypeError) as error:
            rejected.append(dict(source=str(summary),reason=str(error)))
    for quality in (B/"results/gpu_thinning").glob("*/thin*/seed*/quality.json"):
        try:
            q = read(quality)
            if q.get("status") == "complete" and q.get("measured",{}).get("quality_met"):
                add(quality.parents[2].name,"gpu_thinning:"+quality.parents[1].name,q["seed"],resolve(q["measured"]["run_dir"]),q)
        except (OSError,KeyError,ValueError,TypeError) as error:
            rejected.append(dict(source=str(quality),reason=str(error)))
    for quality in (B/"results/precision_quality").glob("*/seed*/quality.json"):
        try:
            q=read(quality)
            if q.get('status')=='complete' and q.get('quality_met') and q.get('strict_bounds_met'):
                add(q['model'],'precision_quality:'+q['variant'],q['seed'],resolve(q['run_dir']),q)
        except (OSError,KeyError,ValueError,TypeError) as error:
            rejected.append(dict(source=str(quality),reason=str(error)))
    groups,incomplete = [],[]
    for (model,method,config),by_seed in sorted(candidates.items()):
        if set(by_seed) != set(SEEDS):
            incomplete.append(dict(model=model,method=method,config=config,passing_seeds=sorted(by_seed)))
            continue
        rows = [by_seed[seed] for seed in SEEDS]
        if len({r["geometry_sha256"] for r in rows}) != 1:
            rejected.append(dict(model=model,config=config,reason="Geometry differs acrossseeds"))
            continue
        group = dict(model=model,method=method,config=config,seeds=list(SEEDS),rows=rows,
                     sampler_phase_seconds_median=statistics.median(r["sampler_phase_seconds"] for r in rows),
                     pooled_bulk_ess_per_sampler_phase_second_median=statistics.median(r["pooled_bulk_ess_per_sampler_phase_second"] for r in rows),
                     transitions_including_warmup_median=statistics.median(r["transitions_including_warmup"] for r in rows),
                     pooled_bulk_ess_per_total_transition_median=statistics.median(r["pooled_bulk_ess_per_total_transition"] for r in rows),
                     draws_per_chain_by_seed={str(r["seed"]):r["draws_per_chain"] for r in rows},
                     workers_by_seed={str(r["seed"]):r["workers"] for r in rows},
                     verified_physical_cores_by_seed={str(r["seed"]):r["verified_physical_cores"] for r in rows},
                     reserved_logical_cpus_by_seed={str(r["seed"]):r["reserved_logical_cpus"] for r in rows})
        groups.append(group)
    fastest = []
    for key in sorted({(g["model"],g["method"]) for g in groups}):
        fastest.append(min((g for g in groups if (g["model"],g["method"]) == key),key=lambda g:g["sampler_phase_seconds_median"]))
    comparisons = []
    for model in sorted({g["model"] for g in fastest}):
        selected = {g["method"]:g for g in fastest if g["model"] == model}
        comparison = dict(model=model,selected_configs={method:g["config"] for method,g in selected.items()})
        if set(selected) == {"CPU","GPU"}:
            cpu,gpu = selected["CPU"],selected["GPU"]
            assert cpu["rows"][0]["geometry_sha256"] == gpu["rows"][0]["geometry_sha256"]
            comparison.update(cpu_sampler_phase_seconds_median=cpu["sampler_phase_seconds_median"],
                gpu_sampler_phase_seconds_median=gpu["sampler_phase_seconds_median"],
                ratio_of_phase_medians_cpu_over_gpu=cpu["sampler_phase_seconds_median"]/gpu["sampler_phase_seconds_median"],
                median_paired_gpu_over_cpu_ess_per_phase_second=statistics.median(gr["pooled_bulk_ess_per_sampler_phase_second"]/cr["pooled_bulk_ess_per_sampler_phase_second"] for cr,gr in zip(cpu["rows"],gpu["rows"])),
                median_paired_cpu_over_gpu_total_transitions=statistics.median(cr["transitions_including_warmup"]/gr["transitions_including_warmup"] for cr,gr in zip(cpu["rows"],gpu["rows"])),
                median_paired_gpu_over_cpu_ess_per_total_transition=statistics.median(gr["pooled_bulk_ess_per_total_transition"]/cr["pooled_bulk_ess_per_total_transition"] for cr,gr in zip(cpu["rows"],gpu["rows"])))
        comparisons.append(comparison)
    result = dict(schema_version=1,generated_utc=datetime.now(timezone.utc).isoformat(),scope="Completed16-chain selected-budget groups; three disjoint replicate seeds101/202/303.",
        primary_metric="Sampling-phase time at observed passing budgets and median reaction pooled bulk ESS per sampling-phase second. Fastest complete configuration per method/model selected by phase median.",
        cpu_scope="hopsy.sample phase includes pool startup, task/result serialization, IPC and returned-array assembly. Inclusive sampler proxy, not pure arithmetic/worker compute.",
        gpu_scope="profile.gpu_sampling_s includes sampler allocation, initial slack GEMM/RNG init and CHRR sampling. Excludes output/download/normalization, but not isolated CHRR kernel-only time.",
        transitions_scope="Actual chains*(retained_draws+retained_warmup)*thinning, including warmup. ESS per total transition separates observed mixing efficiency from hardware timing. Warmup counts fixed in retained draws, so warmup moves vary by thinning.",
        quality_scope="Verified common original-reaction mask; median classical split Rhat<=1.01 and pooled rank-normalized bulk ESS median>=100; zero original-bound violations at1e-7. Mask geometry must match. Quality/search/diagnostic and output costs are excluded from phase timing.",
        complete_groups=groups,fastest_complete_groups=fastest,model_comparisons=comparisons,incomplete_groups=incomplete,read_rejections=rejected)
    cap_path=B/'results/iml_time_to_target/summary.json'
    if cap_path.exists():result['budget_cap_studies']=[read(cap_path)]
    dest = B/"results"
    def atomic(path,text):
        temp = path.with_suffix(path.suffix+".tmp");temp.write_text(text);temp.replace(path)
    atomic(dest/"SAMPLING_BENCHMARKS.json",json.dumps(result,indent=2)+"\n")
    fields = ["model","method","config","quality_status","passing_seed_count","tested_seed_count","sampler_phase_seconds_median","pooled_bulk_ess_per_sampler_phase_second_median","transitions_including_warmup_median","pooled_bulk_ess_per_total_transition_median","draws_per_chain_by_seed","workers_by_seed","verified_physical_cores_by_seed","reserved_logical_cpus_by_seed"]
    temp = dest/"SAMPLING_BENCHMARKS.csv.tmp"
    with temp.open("w") as handle:
        writer = csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore");writer.writeheader()
        for group in fastest:
            record=dict(group,quality_status='PASS_3_OF_3',passing_seed_count=3,tested_seed_count=3)
            writer.writerow({key:(json.dumps(record[key],sort_keys=True) if isinstance(record.get(key),dict) else record.get(key)) for key in fields})
        for study in result.get('budget_cap_studies',[]):
            cap=study['cpu_thin2000_fixed_cap']
            writer.writerow(dict(model=study['model'],method='CPU',config='budget_cap:native_thin2000_fixed10000',
                quality_status='FIXED_CAP_CONTROL_NOT_A_PASSING_TIMING_GROUP',passing_seed_count=cap['passing_count'],tested_seed_count=cap['tested_count'],
                draws_per_chain_by_seed=json.dumps({str(r['seed']):10000 for r in cap['rows']})))
    temp.replace(dest/"SAMPLING_BENCHMARKS.csv")
    lines = ["# Sampling-phase benchmarks","",result["primary_metric"],"",result["cpu_scope"],"",result["gpu_scope"],"",result["transitions_scope"],"",result["quality_scope"],"",
        "| Model | CPU phase(s) | GPU phase(s) | CPU/GPU phase | GPU/CPU ESS/s | CPU/GPU total transitions | GPU/CPU ESS/transition |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for c in comparisons:
        if "ratio_of_phase_medians_cpu_over_gpu" in c:
            lines.append(f"| {c['model']} | {c['cpu_sampler_phase_seconds_median']:.6g} | {c['gpu_sampler_phase_seconds_median']:.6g} | {c['ratio_of_phase_medians_cpu_over_gpu']:.3f} | {c['median_paired_gpu_over_cpu_ess_per_phase_second']:.3f} | {c['median_paired_cpu_over_gpu_total_transitions']:.3f} | {c['median_paired_gpu_over_cpu_ess_per_total_transition']:.3f} |")
        else:
            cpu = next((g for g in fastest if g['model'] == c['model'] and g['method'] == 'CPU'),None)
            gpu = next((g for g in fastest if g['model'] == c['model'] and g['method'] == 'GPU'),None)
            cpu_text = f"{cpu['sampler_phase_seconds_median']:.6g}" if cpu else "—"
            gpu_text = f"{gpu['sampler_phase_seconds_median']:.6g}" if gpu else "—"
            lines.append(f"| {c['model']} | {cpu_text} | {gpu_text} | — | — | — | — |")
    lines += ["","Configuration and CPU allocation details:","","| Model | Method | Selected configuration | Draws/chain by seed | Workers by seed | Verified physical cores by seed | Reserved logical CPUs by seed |","|---|---|---|---|---|---|---|"]
    for group in fastest:
        def values(key):
            return ", ".join(f"{seed}:{value if value is not None else 'unrecorded'}" for seed,value in group[key].items())
        lines.append(f"| {group['model']} | {group['method']} | {group['config']} | {values('draws_per_chain_by_seed')} | {values('workers_by_seed')} | {values('verified_physical_cores_by_seed')} | {values('reserved_logical_cpus_by_seed')} |")
    lines += ["","Ratios are empirical for the finite tested configurations; configuration selection uses sampling phase rather than whole process time. Secondary process/output timings and per-seed mask, bounds, ESS, transition and source records remain in SAMPLING_BENCHMARKS.json. Missing allocation evidence is labeled unrecorded.512-chain best-use-case experiments and worker-timed variants are separate.",""]
    bounds = [r for r in rejected if "bounds_gate_failed" in r.get("reason","")]
    if bounds:
        lines += ["Observed numerical exclusions from the strict comparison:",""]
        lines += [f"- {r['model']} / {r['config']} / seed{r['seed']}: {r['reason']}" for r in bounds]
        lines.append("")
    if result.get('budget_cap_studies'):
        lines += ['Fixed-budget controls, separate from passing timing comparisons:','','| Model | CPU configuration | Retained cap/chain | Actually passed/tested | Not yet run |','|---|---|---:|---:|---|']
        for study in result['budget_cap_studies']:
            cap=study['cpu_thin2000_fixed_cap']
            lines.append(f"| {study['model']} | native thin2000 | 10000 | {cap['passing_count']}/{cap['tested_count']} | {cap['missing_seeds'] or 'none'} |")
        lines += ['', 'A fixed-cap failure is not a claim about eventual convergence at larger budgets. Failed controls have no matched-quality speedup or ESS-rate headline.','']
    atomic(dest/"SAMPLING_BENCHMARKS.md","\n".join(lines))
    return result

if __name__ == "__main__":
    result = generate()
    print(json.dumps(dict(complete_groups=len(result["complete_groups"]),selected_groups=len(result["fastest_complete_groups"]),models=result["model_comparisons"]),indent=2))
