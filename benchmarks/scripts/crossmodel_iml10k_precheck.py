"""Check the preserved10k CPU cap under the new common cached-target assay."""
import json
from pathlib import Path
from quality_diagnostics import rhat_precheck

B=Path(__file__).resolve().parents[1]
R=B/'results/crossmodel_extension'
model=R/'cached_models/iML1515'
rows=json.loads((B/'results/iml_time_to_target/summary.json').read_text())['cpu_thin2000_fixed_cap']['rows']
results=[]
for row in rows:
    source=Path(row['run_dir'])
    result=rhat_precheck(source/'samples.npy',model)
    results.append(dict(seed=row['seed'],source=str(source),precheck=result))
    (R/'iML1515_cached10k_precheck.json').write_text(json.dumps(dict(rows=results,
        note='Rhat/bounds screening only; no ESS or time-to-quality inference. Original10k raw-mask failures preserved.'),indent=2)+'\n')
print(json.dumps(results,indent=2))
