# naja Slurm layer (optional)

A thin, **optional** scheduling layer for running large bulk sampling/prepare
jobs on a Slurm + Pyxis/enroot cluster. **naja does not depend on this.** The
binary runs standalone anywhere — `naja sample bulk --gpus 0,1,... --model-list
L --models-root M --out-root O` — and on a cluster without Slurm (e.g. rosetta4)
you just run that directly. This directory only adds cluster glue; nothing here
is compiled into naja.

## Why

The old production path was a bash static round-robin (`awk (NR-1)%m==k`,
`nohup &` per GPU, `NGPU=8` hardcoded): no load balancing, no requeue, no
monitoring, and it hammered NFS. This layer replaces it with:
- **Slurm job arrays** (fan-out, requeue, backfill, priority tiers, accounting);
- **per-node work-stealing** — each array task is one whole node running `naja
  sample bulk --gpus 0-7`, reusing naja's built-in work-stealing so all 8 GPUs
  stay busy despite ~10 s/model host-prep variance;
- **node-local NVMe staging** — `sample bulk --stage-dir` writes the ~117 MB
  `.npy` to local NVMe; a background rsync drains them to NFS, so sampling never
  contends on NFS (the old path's biggest bottleneck).

## Pipeline (DAG) and commands

`config.sh` holds cluster defaults (partitions, container image, gres, paths);
override any via env. All commands are idempotent (`--skip-existing`), so
re-running resumes.

```bash
export GRB_LICENSE_FILE=/path/to/gurobi.lic     # needed to RUN prepare/sample

# A. reround the base model (once; CPU; hours)  — Phase A
slurm/naja-slurm reround --cmd "python /mnt/home/jack/rs10k_build/reround_ijo1366.py" \
    --mounts /mnt/home/jack/rs10k_build

# B. condition + prepare per model (CPU array)  — Phase B
slurm/naja-slurm prepare --models-root $MODELS --model-list lists/all.txt --tasks 20

# C. sample (GPU array, per-node work-stealing + NVMe staging)  — Phase C  <-- the win
slurm/naja-slurm sample --models-root $MODELS --model-list lists/all.txt \
    --out $DATASET/samples --tasks 5 \
    -- --n-chains 4 --n-samples 12500 --start-policy cube_center --barrier-whiten \
       --constraint-eps 0.01 --bounds-policy ignore

slurm/naja-slurm status --out $DATASET/samples     # progress across the array

# D. assemble into the RS layout (single process)  — Phase D
slurm/naja-slurm migrate --cmd "python /mnt/home/jack/rs10k_build/migrate_to_RS_V3.py --src $DATASET/samples --dst $DATASET/RS" \
    --mounts $DATASET
```

`--tasks` is a parallelism knob (per-node sampling: set it to how many nodes you
want to use, e.g. 5). Contiguous sharding + `--skip-existing` make requeued or
re-submitted arrays safe. Everything for a run lands under `$OUT/_slurm/`
(`shards/`, `logs/`, `meta/`, `run.json`).

## Cluster assumptions & caveats (CoreWeave SUNK)

- Slurm 25.05.3 with Pyxis (`srun --container-image`); GPU gres type is `b300`
  (`gpu:b300:8` per node). Default GPU partition `hpc-mid` (+ `-low/-high/-prod`
  priority tiers, no preemption).
- **`MaxArraySize = 1001`** → `--tasks` must be ≤ 1000 (guarded; for per-node
  sampling it's tiny anyway). `MaxJobCount = 10000`.
- 28 TB `/tmp` NVMe per node is the stage area (`NAJA_STAGE_ROOT`); it is **not**
  a schedulable TRES, so it's best-effort scratch, not reserved.
- Prepare/reround/migrate run on CPU partitions (`cd-gp-i64-erapids` /
  `turin-gp`) to keep GPU nodes for sampling.

## Files

- `config.sh` — cluster defaults (env-overridable).
- `sample.sbatch` + `_task_sample.sh` — per-node GPU array + NVMe staging.
- `prepare.sbatch` — CPU array (Phase B).
- `oneshot.sbatch` — generic single job (Phase A reround, Phase D migrate).
- `naja-slurm` — Python submit/status CLI (sharding, MaxArraySize guard, submit).

The RS_10K-specific pipeline (which lists, which conditioning, the reround/migrate
scripts) lives in `/mnt/home/jack/rs10k_build/` and *calls* this layer — this
stays a generic naja capability.
