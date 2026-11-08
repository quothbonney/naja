"""Stationary CHRR and representative preparation/backtransform kernel counters."""
from pathlib import Path
import ctypes
import json
import os
import shlex
import subprocess
import time

B=Path(__file__).resolve().parents[1]
O=B/'results/nsight_timeline/ncu'
O.mkdir(parents=True,exist_ok=True)
NCU=B/'tools/nsight/extracted/opt/nvidia/nsight-compute/2025.2.1/ncu'
commands=[]
metrics=['lts__t_bytes.sum','lts__t_bytes.sum.per_second','lts__t_bytes.sum.pct_of_peak_sustained_elapsed',
    'lts__t_sector_hit_rate.pct','dram__bytes.sum','dram__bytes.sum.per_second',
    'dram__bytes.sum.pct_of_peak_sustained_elapsed','sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
    'smsp__sass_thread_inst_executed_op_dadd_pred_on.sum',
    'smsp__sass_thread_inst_executed_op_dmul_pred_on.sum',
    'smsp__sass_thread_inst_executed_op_dfma_pred_on.sum',
    'sm__ops_path_tensor_src_fp64.sum','gpu__time_duration.sum']
with (O/'queried-metrics.txt').open('w') as log:
    subprocess.run([str(NCU),'--query-metrics-mode','suffix','--metrics',
        'lts__t_bytes,lts__t_sector_hit_rate,dram__bytes,sm__throughput,sm__warps_active'],stdout=log,stderr=subprocess.STDOUT,check=True)
tasks=[(f'sample_chains{n}',n,'regex:.*chrKernel.*',None) for n in [16,128,512]]
tasks += [('prepare_syrk',16,'regex:.*syrk.*',None),
          ('prepare_sytrd4_gpu',16,'regex:.*sytrd4_gpu.*',None),
          ('prepare_steqr',16,'regex:.*steqr_ker.*',None),
          ('prepare_laed4',16,'regex:.*laed4_par.*',None)]
# The rotation-undo DGEMM is the last GEMM in a short 16-chain CUDA trace.
# Its invocation within its name is derived from the actual timeline trace.
import sqlite3
c=sqlite3.connect(O.parent/'chains16/timeline.sqlite')
kernels=c.execute('select k.start,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName order by k.start').fetchall()
chr_end=max(start for start,name in kernels if 'chrKernel' in name)
backmap=[(start,name) for start,name in kernels if start>chr_end and 'gemm' in name.lower()]
assert backmap, 'No rotation-backtransform GEMM in the real trace'
_,name=backmap[-1]
invocation=sum(n==name for _,n in kernels)
tasks.append(('rotation_backtransform_gemm',16,None,f'::{name}:{invocation}'))
(O/'backtransform-selection.json').write_text(json.dumps(dict(kernel=name,invocation=invocation,
    source=str(O.parent/'chains16/timeline.sqlite'),scope='Reduced-space rotation undo; not full-reaction affine backmap'),indent=2)+'\n')
for label,chains,pattern,kernel_id in tasks:
    out=O/label;out.mkdir(exist_ok=True)
    sample_count=11000 if label=='rotation_backtransform_gemm' else 1200
    cmd=[str(NCU),'--set','full','--metrics',','.join(metrics),'--clock-control','none',
        '--launch-count','1','--export',str(out/'report')]
    cmd+=['--kernel-id',kernel_id] if kernel_id else ['-k',pattern]
    cmd += [str(B/'build-h100-safe/naja'),'sample','run','--model-dir',str(B/'data/models/iJO1366'),
        '--out-root',str(out/'native'),'--gpu','0','--n-chains',str(chains),'--n-samples',str(sample_count),
        '--seed','101','--tpb','128','--thinning','0','--constraint-eps','0','--extra-constraint-eps','0',
        '--pair-prob','0','--bounds-policy','ignore','--write-npy','--barrier-rotate']
    commands.append(shlex.join(cmd))
    (O/'commands.sh').write_text('\n'.join(commands)+'\n')
    start=time.perf_counter()
    with (out/'stdout.log').open('w') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    record=dict(label=label,chains=chains,warmup=1000,retained_draws=sample_count-1000,
        command=cmd,wall_seconds=time.perf_counter()-start,returncode=result.returncode)
    (out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(label,result.returncode,record['wall_seconds'],flush=True)
    if result.returncode:continue
    export=[str(NCU),'--import',str(out/'report.ncu-rep'),'--page','raw','--csv']
    commands.append(shlex.join(export))
    with (out/'metrics.csv').open('w') as f, (out/'export.stderr').open('w') as e:
        subprocess.run(export,stdout=f,stderr=e,check=True,timeout=120)
    (O/'commands.sh').write_text('\n'.join(commands)+'\n')
