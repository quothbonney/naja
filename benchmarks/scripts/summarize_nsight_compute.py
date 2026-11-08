"""Summarize full-set Nsight Compute exports, retaining raw reports and units."""
from pathlib import Path
import csv
import json
import re

B=Path(__file__).resolve().parents[1]
O=B/'results/nsight_timeline/ncu'
def load(path):
    rows=list(csv.reader(path.open()))
    assert len(rows)==3 and len(rows[0])==len(rows[2]),path
    return {n:dict(unit=u,value=v) for n,u,v in zip(*rows)}
def value(m,name):
    try:return float(m[name]['value'].replace(',',''))
    except (KeyError,ValueError):return None
def base(m,name):
    v=value(m,name)
    if v is None:return None
    unit=m[name]['unit']
    factor={'Gbyte':1e9,'Mbyte':1e6,'Kbyte':1e3,'byte':1,
        's':1,'ms':1e-3,'us':1e-6,'ns':1e-9,
        'Gbyte/s':1e9,'Mbyte/s':1e6,'Kbyte/s':1e3,'byte/s':1}.get(unit,1)
    for prefix,scale in [('Gbyte',1e9),('Mbyte',1e6),('Kbyte',1e3)]:
        if unit.startswith(prefix):factor=scale
    return v*factor
rows=[];stalls=[]
for path in sorted(O.glob('*/metrics.csv')):
    m=load(path);label=path.parent.name
    seconds=base(m,'gpu__time_duration.sum')
    fadd=base(m,'smsp__sass_thread_inst_executed_op_dadd_pred_on.sum')
    fmul=base(m,'smsp__sass_thread_inst_executed_op_dmul_pred_on.sum')
    fma=base(m,'smsp__sass_thread_inst_executed_op_dfma_pred_on.sum')
    tensor=base(m,'sm__ops_path_tensor_src_fp64.sum')
    flops=(fadd+fmul+2*fma+tensor) if None not in (fadd,fmul,fma,tensor) else None
    row=dict(case=label,kernel=m['Kernel Name']['value'],kernel_seconds=seconds,
        l2_bytes_peak_pct=value(m,'lts__t_bytes.sum.pct_of_peak_sustained_elapsed'),
        l2_hit_rate_pct=value(m,'lts__t_sector_hit_rate.pct'),
        dram_bytes_peak_pct=value(m,'dram__bytes.sum.pct_of_peak_sustained_elapsed'),
        sm_throughput_peak_pct=value(m,'sm__throughput.avg.pct_of_peak_sustained_elapsed'),
        achieved_occupancy_pct=value(m,'sm__warps_active.avg.pct_of_peak_sustained_active'),
        theoretical_occupancy_pct=value(m,'sm__maximum_warps_per_active_cycle_pct'),
        registers_per_thread=value(m,'launch__registers_per_thread'),
        shared_memory_bytes_per_block=base(m,'launch__shared_mem_per_block'),
        shared_memory_allocated_bytes_per_block=base(m,'launch__shared_mem_per_block_allocated'),
        blocks_limit_registers=value(m,'launch__occupancy_limit_registers'),
        blocks_limit_shared_memory=value(m,'launch__occupancy_limit_shared_mem'),
        blocks_limit_warps=value(m,'launch__occupancy_limit_warps'),
        blocks_limit_hardware=value(m,'launch__occupancy_limit_blocks'),
        blocks_limit_barriers=value(m,'launch__occupancy_limit_barriers'),
        waves_per_sm=value(m,'launch__waves_per_multiprocessor'),
        scalar_fp64_flops=(fadd+fmul+2*fma) if None not in (fadd,fmul,fma) else None,
        tensor_fp64_flops=tensor,total_executed_fp64_flops=flops,
        dram_bytes=base(m,'dram__bytes.sum'),l2_bytes=base(m,'lts__t_bytes.sum'))
    row['l2_hit_rate_in_physical_range']=0<=row['l2_hit_rate_pct']<=100 if row['l2_hit_rate_pct'] is not None else False
    if flops is not None and row['dram_bytes']:
        row.update(fp64_flops_per_dram_byte=flops/row['dram_bytes'],fp64_flops_per_second=flops/seconds)
    rows.append(row)
    ratios={n: value(m,n) for n in m if n.startswith('smsp__average_warps_issue_stalled_') and n.endswith('_per_issue_active.ratio') and '_selected_' not in n}
    for rank,(name,v) in enumerate(sorted(ratios.items(),key=lambda kv:kv[1] or 0,reverse=True),1):
        stalls.append(dict(case=label,rank=rank,stall_reason=name.removeprefix('smsp__average_warps_issue_stalled_').removesuffix('_per_issue_active.ratio'),
            stalled_warps_per_issue_active=v,raw_metric=name))
    (path.parent/'parsed_metrics.json').write_text(json.dumps(m,indent=2)+'\n')
if rows:
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (O/'kernel_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    with (O/'warp_stalls.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(stalls[0]));w.writeheader();w.writerows(stalls)
    roofline_fields=['case','kernel','kernel_seconds','scalar_fp64_flops','tensor_fp64_flops',
        'total_executed_fp64_flops','dram_bytes','l2_bytes','fp64_flops_per_dram_byte','fp64_flops_per_second']
    with (O/'roofline_source.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=roofline_fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    (O/'summary.json').write_text(json.dumps(dict(kernels=rows,top_stalls=[r for r in stalls if r['rank']<=5]),indent=2)+'\n')
    print(json.dumps(rows,indent=2))
