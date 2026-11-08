"""Independent fresh-array validation of full-budget CSV diagnostic reaggregation."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from reaggregate_cached_metrics import reaggregate

B=Path(__file__).resolve().parents[1]
R=B/'results/crossmodel_extension'
cases=[('iML1515',B/'results/precision_quality/iML1515/seed101','gpu_seed101'),
       ('iML1515',B/'results/native_cpu/iML1515/extension_20261108_worker_thin2000_n12000/seed101','cpu12k_seed101')]
proofs=[]
for model,source,label in cases:
    target=R/'reaggregate_validation'/f'{model}_{label}'
    modeldir=R/'cached_models'/model
    a=reaggregate(source,modeldir,target/'csv')
    draws=json.loads((source/'run.json').read_text())['draws']
    command=[str(B/'env-cpu/bin/python'),str(B/'metrics.py'),'--samples',str(source/'samples.npy'),
        '--model-dir',str(modeldir),'--output',str(target/'fresh'),'--prefixes',str(draws)]
    if not (target/'fresh/metrics.json').exists():
        with (target/'fresh.log').open('w') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
    fresh=json.loads((target/'fresh/metrics.json').read_text())
    keys=['split_rhat_median','split_rhat_p95','bulk_ess_median','diagnosed_reactions',
        'constant_reactions_excluded','solver_detected_nearconstant_reactions_excluded',
        'sample_bound_violation_fraction','coordinate_bound_violation_fraction',
        'max_absolute_bound_violation','exactly_constant_sampled_reactions']
    for key in keys:
        assert a[key]==fresh[key],(model,label,key,a[key],fresh[key])
    with (target/'csv/reaction_metrics.csv').open() as f:aggregate_rows=list(csv.DictReader(f))
    with (target/'fresh/reaction_metrics.csv').open() as f:fresh_rows=list(csv.DictReader(f))
    assert len(aggregate_rows)==len(fresh_rows)
    maximum=0.
    for x,y in zip(aggregate_rows,fresh_rows):
        assert x['reaction']==y['reaction'] and x['constant_excluded']==y['constant_excluded']
        assert x['solver_nearconstant_excluded']==y['solver_nearconstant_excluded']
        for key in ('empirical_range','split_rhat','bulk_ess','bound_violation_fraction',
                    'zero_tolerance_bound_violation_fraction','max_absolute_bound_violation'):
            xx,yy=float(x[key]),float(y[key])
            assert np.isclose(xx,yy,rtol=0,atol=0,equal_nan=True),(key,xx,yy)
            if np.isfinite([xx,yy]).all():maximum=max(maximum,abs(xx-yy))
    proofs.append(dict(model=model,label=label,status='passed',compared_summary_keys=keys,
        per_reaction_statistics_exactly_equal=True,max_absolute_statistic_difference=maximum,
        command=command,source=str(source),fresh_metrics=str(target/'fresh/metrics.json')))
    (R/'reaggregate_validation/verification.json').write_text(json.dumps(dict(status='passed',cases=proofs,
        reaggregation_code_sha256=hashlib.sha256((B/'scripts/reaggregate_cached_metrics.py').read_bytes()).hexdigest()),indent=2)+'\n')
print(json.dumps(proofs,indent=2))
