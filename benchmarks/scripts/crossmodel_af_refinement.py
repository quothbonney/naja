"""Small fixed-grid AF refinement evaluated under both aggregation policies."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
from crossmodel_budget_search import actual
from crossmodel_dual_assay import original_assay
from crossmodel_gpu128 import B,R,read,write,sha

def run_budget(method,draws,protocol):
    raw_passes=[]
    choice=dict(model='iAF1260',method=method,selected_draws=draws,tpb=512 if method=='gpu' else None)
    folder=f'cpu_selected_cached_n{draws}' if method=='cpu' else f'gpu_f64_tpb512_selected_n{draws}'
    name=f'cpu_fixed_refinement_n{draws}' if method=='cpu' else f'gpu_f64_tpb512_fixed_refinement_n{draws}'
    for seed in (101,202,303):
        actual(choice,seed)
        out=R/'selected_budget_runs/iAF1260'/folder/f'seed{seed}'
        row=read(out/'quality.json')
        row.update(method=name,selection=str(protocol),
            selection_policy='Predeclared fixed refinement; both original-model and cached-target aggregation policies are reported. No inferred prefix timing.')
        write(out/'quality.json',row)
        raw=original_assay(out,B/'data/models/iAF1260',out/'original_model_assay')
        raw_passes.append(raw['quality_and_bounds_met'])
    return all(raw_passes)

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=('cpu','gpu'),required=True);a=p.parse_args()
    protocol=R/f'iAF1260_{a.method}_refinement_protocol.json'
    write(protocol,dict(model='iAF1260',method=a.method,utc=datetime.now(timezone.utc).isoformat(),
        cpu_fixed_grid=[8000,9000],gpu_fixed_first=3000,gpu_conditional_second=4000,
        gpu_second_condition='Only if first3k group fails original-model-mask Rhat/ESS/bounds on at least one seed.',
        seeds=[101,202,303],source_sha256=sha(Path(__file__)),
        quality='Median split-Rhat<=1.01, bulk ESS>=100, zero original-bound violations at1e-7; both assays explicit.',
        job=os.getenv('SLURM_JOB_ID')))
    if a.method=='cpu':
        for budget in (8000,9000):run_budget('cpu',budget,protocol)
    elif not run_budget('gpu',3000,protocol):
        run_budget('gpu',4000,protocol)

if __name__=='__main__':main()
