"""Same-Slurm-GPU order-balanced TPB control; exact arrays validate quality reuse."""
import json
from pathlib import Path
import subprocess
import statistics
import os
import numpy as np
from crossmodel_gpu128 import read,write,sha,B,R

def main():
    outroot=R/'tpb_paired_control';outroot.mkdir(exist_ok=True)
    model=R/'cached_models/iML1515';binary=B/'build-h100-f64/naja'
    reference=B/'results/precision_quality/iML1515/seed101'
    actual_reference=np.load(reference/'samples.npy',mmap_mode='r')
    rows=[]
    for i,tpb in enumerate((512,128,128,512)):
        out=outroot/f'order{i}_tpb{tpb}'
        command=[str(B/'env-cpu/bin/python'),str(B/'naja_runner.py'),'--exe',str(binary),
            '--model',str(model),'--out',str(out),'--seed','101','--chains','16','--draws','2000',
            '--warmup','1000','--tpb',str(tpb),'--thinning','1000','--barrier','rotate',
            '--pair-prob','0','--sample-dtype','float64','--timeout','600']
        if not (out/'run.json').exists():
            with (outroot/f'order{i}.sampling.log').open('w') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=660)
        run=read(out/'run.json');assert run['status']=='complete'
        x=np.load(out/'samples.npy',mmap_mode='r')
        assert x.shape==actual_reference.shape and x.dtype==actual_reference.dtype
        assert np.array_equal(x.view(np.uint64),actual_reference.view(np.uint64)), 'Reference quality reuse requires exact trajectory bit equality'
        rows.append(dict(order=i,tpb=tpb,sampler_seconds=run['profile']['timing']['gpu_sampling_s'],
            saved_array_bitwise_equal_to_passing_reference=True,reference=str(reference),
            source_samples_sha256=sha(out/'samples.npy'),reference_samples_sha256=sha(reference/'samples.npy'),
            command=command,run_directory=str(out),slurm_job_id=os.getenv('SLURM_JOB_ID'),
            cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),slurm_job_gpus=os.getenv('SLURM_JOB_GPUS')))
        write(outroot/'summary.json',dict(rows=rows,note='Four actual runs in one one-GPU Slurm allocation, order512/128/128/512. Fixed seed/budget/binary/target; all arrays exactly equal to quality-passing F64 reference. Performance control, not independent convergence seeds.'))
    med={tpb:statistics.median(row['sampler_seconds'] for row in rows if row['tpb']==tpb) for tpb in (128,512)}
    write(outroot/'summary.json',dict(rows=rows,median_sampler_seconds_by_tpb=med,
        ratio_tpb128_over_tpb512=med[128]/med[512],
        note='Same physical GPU allocation; order-balanced duplicate timings. Exact saved-array quality parity; these are throughput controls, not independent convergence seeds.'))

if __name__=='__main__':main()
