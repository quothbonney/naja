"""Extract real Nsight counters/trace phases and render a shared-seconds figure."""
from pathlib import Path
import csv
import json
import sqlite3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

B=Path(__file__).resolve().parents[1]
O=B/'results/nsight_timeline'
labels=['chains16','chains512','chains512_prefault8']
titles=['16 chains','512 chains','512 chains · host prefault']
metric_names=['SMs Active [Throughput %]','DRAM Read Bandwidth [Throughput %]',
    'DRAM Write Bandwidth [Throughput %]','PCIe RX Throughput [Throughput %]',
    'PCIe TX Throughput [Throughput %]']
colors=['#0072B2','#D55E00','#009E73','#CC79A7']
series_names=['SM active','DRAM read + write','PCIe RX','PCIe TX']
summary=[];sources=[];plots=[];phases=[];phase_rows=[];overheads=[]
for label in labels:
    out=O/label
    manifest=json.loads((out/'manifest.json').read_text())
    byphase={r['phase']:r for r in manifest['runs']}
    record=byphase['profiled'];wall=record['target_wall_seconds']
    timing=record['naja_profile']['timing']
    c=sqlite3.connect(out/'timeline.sqlite')
    session_epoch=c.execute('select utcEpochNs from TARGET_INFO_SESSION_START_TIME').fetchone()[0]
    target=json.loads((out/'profiled/target-wall.json').read_text())
    offset=(target['start_epoch_ns']-session_epoch)/1e9
    gpu=c.execute('select id,name,busLocation,uuid,smCount,memoryBandwidth from TARGET_INFO_GPU').fetchall()
    assert len(gpu)==1,gpu
    metric_ids=dict(c.execute('select metricName,metricId from TARGET_INFO_GPU_METRICS'))
    vectors=[]
    for name in metric_names:
        rows=np.asarray(c.execute('select timestamp,value from GPU_METRICS where metricId=? order by timestamp',(metric_ids[name],)).fetchall(),dtype=np.float64)
        t=rows[:,0]/1e9-offset
        keep=(t>=0)&(t<=wall)
        if not vectors:times=t[keep]
        else:assert np.array_equal(times,t[keep])
        vectors.append(rows[keep,1])
    raw=np.column_stack([times,*vectors])
    counters=np.column_stack([vectors[0],vectors[1]+vectors[2],vectors[3],vectors[4]])
    # Centered 5ms moving average at the measured100us cadence; no padded zeros.
    smooth=np.column_stack([np.convolve(x,np.ones(50),mode='same')/
        np.convolve(np.ones(len(x)),np.ones(50),mode='same') for x in counters.T])
    plotted=np.column_stack([times[::10],smooth[::10]])
    sources.extend([[label,*row] for row in raw])
    plots.extend([[label,*row] for row in plotted])
    kernels=c.execute('select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id order by k.start').fetchall()
    chr_k=[(a/1e9-offset,b/1e9-offset,name) for a,b,name in kernels if name=='chrKernel']
    assert len(chr_k)==1,chr_k
    ss,se,_=chr_k[0]
    d2h=c.execute('select start,end,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=2 order by bytes desc limit 1').fetchone()
    ds,de=d2h[0]/1e9-offset,d2h[1]/1e9-offset
    assert 0<ss<se<ds<de<wall,(ss,se,ds,de,wall)
    p=[('Prepare',0,ss),('Sample',ss,se),('Map + D2H',se,de),('Write / exit',de,wall)]
    phases.append(p)
    phase_rows.extend([dict(run=label,phase=n,start_s=a,end_s=b,duration_s=b-a,
        source='CUDA kernel/memcpy trace; process lifetime from target timer and Nsight session epoch') for n,a,b in p])
    base=np.median([byphase[p]['target_wall_seconds'] for p in ['unprofiled_before','unprofiled_after']])
    overhead=dict(run=label,unprofiled_before_s=byphase['unprofiled_before']['target_wall_seconds'],
        unprofiled_after_s=byphase['unprofiled_after']['target_wall_seconds'],unprofiled_midpoint_s=float(base),
        profiled_target_wall_s=wall,profiled_target_overhead_pct=100*(wall/base-1),
        profiler_command_wall_including_postprocessing_s=record['outer_wall_seconds'])
    overheads.append(overhead)
    summary.append(dict(gpu=gpu,metric_cadence_median_s=float(np.median(np.diff(times))),
        gpu_metric_set='gh100',sm_active_sample_mean_pct=float(counters[(times>=ss)&(times<=se),0].mean()),
        sample_kernel_s=se-ss,naja_sampling_s=timing['gpu_sampling_s'],
        d2h_device_copy_s=de-ds,naja_download_including_host_allocation_s=timing['download_from_gpu_s'],
        backtransform_download_trace_s=de-se,naja_overlapping_backtransform_s=timing['backtransform_s'],
        write_exit_idle_s=wall-de,naja_write_s=timing['write_output_s'],
        process_epoch_offset_from_trace_s=offset,profile=timing,**overhead))
    with (out/'kernel_trace.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['start_s','end_s','kernel']);w.writerows([(a/1e9-offset,b/1e9-offset,n) for a,b,n in kernels])
    manifest['trace_gpu']=gpu
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

def write_csv(name,header,rows):
    with (O/name).open('w') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)
write_csv('timeline_source.csv',['run','time_s','sm_active_pct','dram_read_pct','dram_write_pct','pcie_rx_pct','pcie_tx_pct'],sources)
write_csv('timeline_plotted.csv',['run','time_s','sm_active_pct','dram_read_write_pct','pcie_rx_pct','pcie_tx_pct'],plots)
write_csv('phases.csv',list(phase_rows[0]),[list(r.values()) for r in phase_rows])
write_csv('profiler_overhead.csv',list(overheads[0]),[list(r.values()) for r in overheads])
(O/'timeline_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
    'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(3,1,figsize=(13,7.5),sharex=True,sharey=True)
for i,(ax,label,title,p) in enumerate(zip(axes,labels,titles,phases)):
    data=np.array([r[1:] for r in plots if r[0]==label])
    for j,(name,color) in enumerate(zip(series_names,colors)):
        ax.plot(data[:,0],data[:,j+1],lw=1.1,color=color,label=name,alpha=.95)
    ax.set_ylim(0,102);ax.set_yticks([0,50,100]);ax.grid(axis='y',color='#d9dde2',lw=.5)
    ax.text(0,1.36,title,transform=ax.transAxes,fontweight='bold',fontsize=11,ha='left',va='bottom')
    trans=blended_transform_factory(ax.transData,ax.transAxes)
    for k,(name,start,end) in enumerate(p):
        y=1.08
        texty=1.10
        if i==0 and k==2:y=1.23;texty=1.25
        ax.plot([start,start,end,end],[y-.035,y,y,y-.035],transform=trans,color='#333333',lw=.75,clip_on=False)
        ax.text((start+end)/2,texty,name,transform=trans,ha='center',va='bottom',fontsize=8,clip_on=False)
    ax.axvline(p[-1][2],color='#888888',lw=.6,ls=':')
axes[0].legend(loc='upper right',ncol=4,frameon=False,fontsize=9,bbox_to_anchor=(1.005,1.42),handlelength=2)
axes[-1].set_xlabel('Time (s)')
axes[-1].set_xlim(0,max(p[-1][2] for p in phases)*1.02)
fig.supylabel('Utilization (% of peak)',x=.027,fontsize=11)
fig.subplots_adjust(left=.075,right=.99,top=.89,bottom=.085,hspace=.9)
for ext in ['svg','pdf','png']:
    fig.savefig(O/f'naja_h100_utilization_timeline.{ext}',dpi=180,bbox_inches='tight')
print(json.dumps(summary,indent=2))
