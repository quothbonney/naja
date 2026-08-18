# naja Slurm layer — detailed guide

How to run large **bulk sampling** and **bulk prepare** jobs for dataset
generation (e.g. RS_10K) on a Slurm + Pyxis/enroot cluster, using the optional
scheduling layer in `naja/slurm/`.

> **This layer is optional.** naja itself has no Slurm dependency: the binary
> runs standalone with `naja sample bulk ...`. This layer is *cluster glue* that
> shards a model list, submits Slurm job arrays that call `sample bulk`/`sample
> prepare`, stages I/O through node-local NVMe, and reports progress. On a box
> without Slurm you skip all of this and run the binary directly.

---

## 1. Mental model

The dataset is generated in four phases (a DAG):

```
A. reround   base polytope        (once, CPU, hours)        -> oneshot.sbatch
B. prepare   per-condition setup  (CPU array, Gurobi LP)    -> prepare.sbatch     [BULK]
C. sample    per-condition draws  (GPU array, the heavy one)-> sample.sbatch      [BULK]
D. migrate   assemble RS layout   (once, CPU, single proc)  -> oneshot.sbatch
```

Phases B and C are the "bulk" phases — thousands of **independent, per-condition
tasks**. The layer runs each bulk phase as a **Slurm job array**, and inside each
array task it invokes naja's built-in **work-stealing** scheduler (`sample bulk`)
so every GPU on the node stays busy.

### Why per-node array tasks (not per-GPU)

Each **array task = one whole node**. A B300 node has 8 GPUs; the task runs one
`naja sample bulk --gpus 0-7`, and naja's work-stealing (one worker thread per
GPU pulling the next model off an atomic counter) load-balances across those 8
GPUs. This keeps all GPUs busy despite the ~10 s/model host-prep variance —
better than one Slurm task per GPU (which would idle a GPU whenever its shard has
a slow model, and pay a container start per GPU).

Slurm handles the cross-node fan-out, queueing, requeue, and accounting; naja
handles intra-node load balancing.

### Data flow for one sample task (the efficiency story)

```
   shard_i.txt  (list of ~N model names for this task)
        |
   sample bulk --gpus 0-7 --stage-dir $STAGE --flat-output $OUT --skip-existing
        |                         |
   8 GPUs work-steal models   writes each ~117MB .npy to  $STAGE = node-local NVMe (/tmp)
        |                         |
        |                    background rsync (every ~45s) drains  $STAGE --> $OUT (NFS)
        |                         |                                   (atomic temp+rename)
   bulk_summary.csv ----------> $OUT/_logs_shard_i.csv
```

The ~117 MB `.npy` files are written to **node-local NVMe first**, so sampling
never contends on NFS (the old pipeline's bottleneck: 40 GPUs writing 117 MB
blobs to one NFS mount at once). A single background rsync per node drains them
to NFS at a controlled rate.

---

## 2. Prerequisites

1. **Built binary** at `$NAJA_BIN` (default `$NAJA_REPO/build/naja`). Build it
   with `scripts/build_container.sh` (see `../docs/building.md`).
2. **Container image** with the CUDA runtime (`$NAJA_IMAGE`, an enroot `.sqsh`).
   Pyxis/`srun --container-image` must be available (it is on SUNK).
3. **Gurobi license** for phases B and C at **runtime** (the feasible-start LP):
   `export GRB_LICENSE_FILE=/path/to/gurobi.lic`. Not needed to build; not needed
   for models that ship a precomputed start.
4. **Prepared model dirs** for sampling: one dir per condition under
   `--models-root`, each containing its rounding + `extra_A/extra_b` + bounds +
   start (produced by phase B). See `../docs/model-format.md`.

---

## 3. Configuration

Defaults live in `slurm/config.sh` and are read (with the same names) by the
`naja-slurm` CLI. Override any by exporting the env var before running.

| Env var | Default | Meaning |
|---|---|---|
| `NAJA_GPU_PARTITION` | `hpc-mid` | partition for sampling (`-low/-mid/-high/-prod` = politeness tiers) |
| `NAJA_CPU_PARTITION` | `cd-gp-i64-erapids` | partition for prepare/reround/migrate (no GPUs) |
| `NAJA_GRES` | `gpu:b300:8` | GPUs per array task (one full node) |
| `NAJA_GPUS_CSV` | `0,1,2,3,4,5,6,7` | passed to `sample bulk --gpus`; must match the gres count |
| `NAJA_IMAGE` | dist-lab pytorch sqsh | enroot container image |
| `NAJA_REPO` | `/mnt/home/jack/naja` | repo mount (binary + vendored gurobi rpath) |
| `NAJA_BIN` | `$NAJA_REPO/build/naja` | the built binary |
| `NAJA_STAGE_ROOT` | `/tmp` | node-local NVMe root for staging |
| `NAJA_SYNC_INTERVAL` | `45` | seconds between background NVMe→NFS rsyncs |
| `NAJA_MAX_ARRAY` | `1000` | MaxArraySize-1 guard (this cluster: MaxArraySize=1001) |
| `GRB_LICENSE_FILE` | *(unset)* | Gurobi license (required to run B/C) |

---

## 4. Bulk SAMPLE (Phase C) — the main event

### Command

```bash
export GRB_LICENSE_FILE=/path/to/gurobi.lic

slurm/naja-slurm sample \
    --models-root  /mnt/vast/.../models \      # dir with one subdir per condition
    --model-list   lists/all_conditions.txt \  # one condition name per line
    --out          /mnt/vast/.../RS/samples \  # where the final <model>.npy land (NFS)
    --tasks        5 \                          # array size = how many nodes at once
    -- \                                        # everything after -- goes to `sample bulk`
       --n-chains 4 --n-samples 12500 \
       --start-policy cube_center --barrier-whiten \
       --constraint-eps 0.01 --bounds-policy ignore
```

`--tasks` is the parallelism knob: with per-node tasks, set it to the number of
nodes you want to occupy (≤ 5 on this cluster's B300 pool; ≤ `NAJA_MAX_ARRAY`).
The condition list is split into that many contiguous shards.

### What happens, step by step

1. **Shard.** The CLI writes `--tasks` shard files `shard_0.txt … shard_{T-1}.txt`
   under `$OUT/_slurm/shards/`, splitting the model list contiguously. (Contiguous
   is fine — correctness comes from `--skip-existing`, not shard boundaries.)
2. **Submit.** It submits `sample.sbatch` as `--array=0-{T-1}` on
   `$NAJA_GPU_PARTITION`, each task `--exclusive --gres=gpu:b300:8 --requeue`, and
   records `$OUT/_slurm/run.json` (job id, shard/model counts, flags).
3. **Per task** (`_task_sample.sh`, inside the container): picks
   `shard_${SLURM_ARRAY_TASK_ID}.txt`, makes `$STAGE=/tmp/naja_stage_<jobid>_<task>`,
   starts the background NVMe→NFS rsync, then runs:
   ```
   naja sample bulk --models-root M --model-list shard_i.txt \
        --gpus 0,1,...,7 --skip-existing \
        --flat-output $OUT --stage-dir $STAGE \
        --out-root $OUT/_slurm/meta --name shard_i  <your sample flags>
   ```
   `sample bulk` work-steals models across the 8 GPUs; each finished `.npy` is
   written to `$STAGE` (local), drained to `$OUT` (NFS) by the background rsync.
4. **Finish.** On exit it does a final drain and copies that shard's
   `bulk_summary.csv` to `$OUT/_logs_shard_i.csv`.

### Outputs

```
$OUT/<condition>.npy              # the samples (117MB each), one per condition
$OUT/_logs_shard_<i>.csv          # per-shard bulk_summary: model,status,elapsed_s,device_id,...
$OUT/_slurm/shards/shard_<i>.txt  # the shards
$OUT/_slurm/logs/sample_<A>_<a>.out  # per-array-task stdout/stderr
$OUT/_slurm/meta/shard_<i>_*/     # per-model metadata (profile.json, run_manifest.json)
$OUT/_slurm/run.json              # run record
```

### Monitor

```bash
slurm/naja-slurm status --out /mnt/vast/.../RS/samples
# -> phase/job/tasks, "<done>/<total> .npy (NN%)", squeue state, OK/FAIL/SKIP tallies
squeue -j <jobid>                              # raw queue
tail -f $OUT/_slurm/logs/sample_*_0.out        # a task's live log
```

### Resume / requeue (idempotent)

Every condition is idempotent at completed-model granularity:
- `--skip-existing` skips any condition whose `.npy` is already in `$OUT` (synced)
  or `$STAGE` (this run). So **re-running the exact same command resumes** — done
  conditions are skipped, only missing ones sample.
- `--requeue` means a node failure re-runs that task from the top; skip-existing
  makes that safe (it picks up where it left off).
- Add more/failed conditions: just re-submit with the same `--out`; only the
  missing `.npy` get produced.

---

## 5. Bulk PREPARE (Phase B)

Phase B conditions each model (KO bounds → extra half-plane rows in the reduced
space → feasible-start LP) and inherits the base rounding. It's CPU/Gurobi work,
so it runs as an array on a **CPU partition** — keeping the GPU nodes free for
sampling. No NVMe staging (prepare outputs are tiny).

```bash
export GRB_LICENSE_FILE=/path/to/gurobi.lic

slurm/naja-slurm prepare \
    --models-root /mnt/vast/.../models \
    --model-list  lists/all_conditions.txt \
    --tasks       20 \                       # CPU array, more tasks OK (<= MaxArraySize)
    -- \                                     # flags passed to `naja sample prepare`
       --base-model-dir /mnt/vast/.../models/base_model \
       --mode symlink
```

Per task it runs `naja sample prepare --models-root M --model-list shard_i.txt
<your prepare flags>` in the container on `$NAJA_CPU_PARTITION`. `sample prepare`
itself parallelizes the feasible-start LPs across CPU threads within the task.
Prepare is idempotent (skips models already conditioned), so re-running resumes.

> The upstream gene-KO bounds generation (`gen_ko_models.py` etc.) that *creates*
> the per-condition dirs is RS_10K-specific and lives in
> `/mnt/home/jack/rs10k_build/`; run it (or wrap it via `oneshot`) before prepare,
> or fold it into your `--prepare` flags. This layer stays generic.

---

## 6. One-shot phases (reround, migrate)

`oneshot.sbatch` runs a single command in the container. Use it for the
non-array phases:

```bash
# Phase A — reround the base model (CPU, hours)
slurm/naja-slurm reround \
    --cmd    "python /mnt/home/jack/rs10k_build/reround_ijo1366.py" \
    --mounts /mnt/home/jack/rs10k_build

# Phase D — assemble the RS dataset (CPU, single process)
slurm/naja-slurm migrate \
    --cmd    "python /mnt/home/jack/rs10k_build/migrate_to_RS_V3.py --src $OUT --dst $DATASET/RS" \
    --mounts "$DATASET,/mnt/home/jack/rs10k_build"
```

Add `--gpu` to a oneshot if the command needs a GPU (reround/migrate don't).

---

## 7. Tuning

| Knob | Effect |
|---|---|
| `--tasks` | Parallelism. Per-node sampling: = nodes used at once (≤5 B300 nodes here). Higher = more nodes but smaller shards (finer requeue granularity). |
| `NAJA_GPU_PARTITION` | Use `hpc-low` to be polite (others preempt in queue order — no eviction), `hpc-high/-prod` to jump the queue. |
| `NAJA_SYNC_INTERVAL` | Lower = fresher NFS progress + less loss on requeue, but more rsync churn. 45 s is a good default. |
| `NAJA_STAGE_ROOT` | Node-local NVMe (`/tmp`, 28 TB on B300). Set empty to disable staging (write straight to NFS — not recommended at scale). |
| sample flags after `--` | `--n-chains`, `--n-samples`, `--thinning`, `--start-policy`, `--barrier-whiten`, `--constraint-eps`, `--bounds-policy`, etc. (see `naja sample bulk --help`). |

---

## 8. Troubleshooting

- **Job sampled for minutes then failed at `atomic rename ... ENOENT`** — the model
  sampled but no `.npy` was written (`WRITE_DATA` off). `--flat-output`/`--stage-dir`
  now **imply `WRITE_DATA=true`**, so current binaries handle this automatically; on
  an older binary, pass `--write-npy` in the sample flags after `--`.
- **`tasks=N exceeds MaxArraySize cap`** — lower `--tasks` (fewer, bigger shards)
  or split the model list; this cluster caps array indices at 1000.
- **All tasks `PENDING`** — only 5 B300 nodes (40 GPUs); tasks beyond that queue
  (no preemption, they just wait). Check `squeue`.
- **`FAIL` in a shard's `_logs_shard_i.csv`** — read that model's log under
  `$OUT/_slurm/logs/`; common cause is a missing/expired `GRB_LICENSE_FILE` (the
  feasible-start LP) or a malformed model dir. Fix and re-submit (skip-existing
  leaves the good ones alone).
- **Partial `.npy` in `$OUT`?** — shouldn't happen: writes are atomic (temp +
  rename), and only complete `*.npy` are drained (in-flight files are
  `*.npy.tmp.<pid>.<dev>`).
- **NVMe fills up** — the background rsync uses `--remove-source-files`, so drained
  files free space; if a task dies its `$STAGE` is node-local and reclaimed by the
  next job. Lower `NAJA_SYNC_INTERVAL` if a single node produces faster than it drains.

---

## 9. Full RS_10K sequence (reference)

```bash
export GRB_LICENSE_FILE=/path/to/gurobi.lic
L=lists/all_conditions.txt ; M=/mnt/vast/.../models ; OUT=/mnt/vast/.../RS/samples

slurm/naja-slurm reround --cmd "python .../reround_ijo1366.py" --mounts .../rs10k_build
slurm/naja-slurm prepare --models-root $M --model-list $L --tasks 20 -- --mode symlink
slurm/naja-slurm sample  --models-root $M --model-list $L --out $OUT --tasks 5 \
    -- --n-chains 4 --n-samples 12500 --start-policy cube_center --barrier-whiten \
       --constraint-eps 0.01 --bounds-policy ignore
slurm/naja-slurm status  --out $OUT           # watch until done
slurm/naja-slurm migrate --cmd "python .../migrate_to_RS_V3.py --src $OUT --dst .../RS" --mounts ...
```
