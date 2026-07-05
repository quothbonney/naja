# Naja Model Format Reference

A **model** is a directory on disk that fully specifies one polytope to sample.
Naja expects a fixed directory layout and a fixed set of files inside it.
This document describes exactly what each file is, why it exists, and where
the format is getting unwieldy.

---

## Directory layout

```
<models_root>/
└── <model_name>/              # e.g. ko_b0002_b0573
    ├── gem/                   # GEM-level files (reaction space)
    │   ├── reaction_ids.txt   # one reaction ID per line, defines column order
    │   ├── l_bounds.csv       # lower flux bounds (n_rxn × 1)
    │   ├── u_bounds.csv       # upper flux bounds (n_rxn × 1)
    │   ├── A_eq.csv           # stoichiometric equality rows (optional, legacy)
    │   ├── b_eq.csv           # RHS for A_eq (optional, legacy)
    │   └── conditioning.signature   # hash for provenance checking (prepare writes this)
    │
    └── rounding/              # pre-rounded, reduced polytope (sampling space)
        ├── <model_name>_rounding_A.csv      # constraint matrix  (m × d)
        ├── <model_name>_rounding_b.csv      # RHS vector         (m × 1)
        ├── <model_name>_rounding_start.csv  # feasible start point (d × 1)
        ├── <model_name>_rounding_T.csv      # backmap matrix T   (n_rxn × d)
        ├── <model_name>_rounding_shift.csv  # backmap shift      (n_rxn × 1)
        ├── <model_name>_rounding_extra_A.csv  # KO tightening rows (optional)
        ├── <model_name>_rounding_extra_b.csv  # KO tightening RHS  (optional)
        └── INHERITED_FROM.txt               # provenance if rounding was symlinked
```

---

## What each rounding file means

The polytope naja actually samples is `{z ∈ ℝᵈ : A·z ≤ b}` in a
pre-rounded (reduced) coordinate system. The full-space flux vector is
recovered by `x = T·z + shift`.

| File | Shape | Description |
|---|---|---|
| `_rounding_A.csv` | `m × d` | Constraint matrix in reduced space. Rows are tight-constraint halfplanes. `d` is the rounding dimension (≤ n_rxn after equality reduction). |
| `_rounding_b.csv` | `m × 1` | RHS vector. Feasibility requires `A·z ≤ b`. |
| `_rounding_start.csv` | `d × 1` | A known strictly feasible interior point in reduced space. Required; used as MCMC start and for Gurobi cube-center LP. |
| `_rounding_T.csv` | `n_rxn × d` | Linear backmap. Converts reduced-space sample `z` to full reaction-space flux `x`. |
| `_rounding_shift.csv` | `n_rxn × 1` | Affine shift for backmap: `x = T·z + shift`. |
| `_rounding_extra_A.csv` | `k × d` | (Optional) additional halfplane rows encoding gene-KO-specific bound tightenings, in reduced space. Appended to A at sampling time. |
| `_rounding_extra_b.csv` | `k × 1` | (Optional) RHS for extra constraints. |

### Typical dimensions (iJO1366 RS_V3)
- `n_rxn` = 2583 (full reaction space)
- `d` = 582 (rounding space after equality reduction)
- `m` ≈ 1000–1800 (constraint rows; varies by KO tightening)
- `k` ≈ 0–200 (extra constraints from KO bound changes)

---

## GEM files

`gem/` holds the reaction-space representation before rounding.
It is used by:
- `naja sample prepare` — to build `extra_A/extra_b` from bound differences
- `naja sample bulk --bounds-policy filter` — to validate samples against original bounds
- External conditioners (e.g. E-Flux2) — to read/write flux bounds

| File | Description |
|---|---|
| `reaction_ids.txt` | One reaction ID per line. Defines column ordering for all n_rxn-dimensional vectors. Must match the base model's list exactly. |
| `l_bounds.csv` | Lower bounds for each reaction (n_rxn × 1). |
| `u_bounds.csv` | Upper bounds for each reaction (n_rxn × 1). |
| `conditioning.signature` | Written by `prepare` after conditioning. Contains a hash of the conditioner command + args + base reaction IDs + model name, used to detect stale conditioning. |

---

## Inheritance and the base-model pattern

For bulk KO runs, only one model (the WT or a shared base) has a
fully computed rounding (T, shift, A, b, start). All KO variants inherit
the rounding by symlink or copy:

```
naja sample prepare \
  --models-root /path/to/models \
  --model-list ko_list.txt \
  --base-model-dir /path/to/models/wt \
  --mode symlink
```

This writes symlinks like:
```
ko_b0002/rounding/ko_b0002_rounding_T.csv -> ../../wt/rounding/wt_rounding_T.csv
```

And writes `INHERITED_FROM.txt` for provenance:
```
base_model_dir=/path/to/models/wt
target_model_dir=/path/to/models/ko_b0002
mode=symlink
created_at=2026-07-05T14:22:01
```

KO-specific bound tightenings go into `extra_A/extra_b` in the reduced
space. `prepare` builds these by:
1. Loading `l_bounds/u_bounds` for the KO model and the base
2. For each reaction where the KO tightens the bound, projecting the
   tightening into reduced space via the backmap matrix T:
   - upper tightening: row `T[i, :]`, RHS `ub_new[i] - shift[i]`
   - lower tightening: row `-T[i, :]`, RHS `-lb_new[i] + shift[i]`

---

## What's painful about the current format

### 1. Redundant name prefix on every rounding file

Every file in `<model>/rounding/` is prefixed with `<model_name>_rounding_`.
Since the files are already scoped inside that directory, this prefix adds
nothing. For a model named `ko_b0002_b0573_b1234`, every filename is 30+
characters of redundant noise.

**Current:** `ko_b0002_b0573_b1234/rounding/ko_b0002_b0573_b1234_rounding_A.csv`
**Could be:** `ko_b0002_b0573_b1234/rounding/A.csv`

The prefix exists because `RuntimeConfig::derive_paths()` constructs paths as
`ROUND_PREFIX + "_A.csv"` where `ROUND_PREFIX = ROUNDING_DIR + "/" + MODEL_NAME + "_rounding"`.
Fixing this is a one-line change in `derive_paths()` and `inherit.cpp`, but
would require migrating all existing data on disk.

### 2. CSV for dense matrices

`T.csv` is `n_rxn × d = 2583 × 582 ≈ 1.5M doubles` stored as ASCII decimal.
At 12 significant figures per number plus commas and newlines, this is
roughly **25 MB per model** for T alone. For 10K models that's 250 GB just
for T matrices — most of which are identical symlinks, so it's fine in
practice, but loading a 25 MB CSV on every job start adds ~0.3s of I/O even
on NFS cache hits.

A float64 `.npy` for the same matrix would be **12 MB** and load in ~5ms.

### 3. Split bounds files

`l_bounds.csv` and `u_bounds.csv` are always read and written together.
They're never used independently. They could be one two-column file or a
two-array `.npz`.

### 4. No schema / version field

There's no `manifest.json` or equivalent in the model directory. If a new
optional file is added (e.g. a precomputed pair schedule, or dikin
directions), there's no way to tell whether it's absent because it was never
computed vs. because it's from an older version of the format.

### 5. `gem/` and `rounding/` serve different audiences but are tightly coupled

`gem/` is the reaction-space contract (for humans and conditioners).
`rounding/` is the sampling contract (for naja). They're linked via T/shift,
but there's no file that records *which* rounding corresponds to *which* gem
version. If you re-round a model and update `rounding/` but forget to update
`gem/`, the backmap silently produces wrong flux vectors.

---

## Simplification proposals

### Option A: Drop the model_name prefix (low risk, high value)

Change `derive_paths()` to use fixed names inside `rounding/`:

```
rounding/A.csv
rounding/b.csv
rounding/start.csv
rounding/T.csv
rounding/shift.csv
rounding/extra_A.csv   (optional)
rounding/extra_b.csv   (optional)
```

Pros: eliminates the main cosmetic annoyance, trivial to implement.
Cons: requires a migration script for existing data (one `mv` per file per model).

### Option B: Single NPZ archive per model (medium risk, big win)

Replace all rounding CSVs with one `rounding/polytope.npz` (numpy zip):

```python
np.savez(
    "rounding/polytope.npz",
    A=...,         # float64 (m, d)
    b=...,         # float64 (m,)
    start=...,     # float64 (d,)
    T=...,         # float64 (n_rxn, d)  -- only if backmap
    shift=...,     # float64 (n_rxn,)
    extra_A=...,   # float64 (k, d)      -- only if present
    extra_b=...,   # float64 (k,)
)
```

Pros: 2–3× smaller than CSV, ~10× faster to load, single file to copy/check,
schema embedded in array names, trivially extensible (add new arrays without
breaking old readers).
Cons: naja is C++ — would need a minimal NPZ reader (the existing `npy.h`
handles `.npy`; `.npz` is just a ZIP of `.npy` files, so `libzip` or manual
ZIP parsing). Also breaks the symlink-inheritance trick since you can't
symlink individual arrays within a NPZ.

For the symlink inheritance pattern, a compromise: keep T and shift as
separate `.npy` files (since those are the shared rounding basis and benefit
most from symlinking), bundle everything else into a per-model `polytope.npz`.

### Option C: Manifest + lazy validation (low risk, good hygiene)

Add a `rounding/manifest.json` written by `prepare` and `inherit`:

```json
{
  "format_version": 2,
  "model_name": "ko_b0002",
  "rounding_dim": 582,
  "n_reactions": 2583,
  "n_constraints": 1247,
  "n_extra_constraints": 14,
  "base_model": "wt",
  "inherit_mode": "symlink",
  "created_at": "2026-07-05T14:22:01"
}
```

Lets naja validate shapes before loading large files, enables `naja sample list`
to show a useful summary without parsing CSVs, and gives a migration path
for format versioning.

### Recommended path

1. **Now (zero migration):** Add `manifest.json` (Option C). Pure addition, no
   existing files change, immediately useful for tooling.
2. **Next batch run prep:** Drop the name prefix (Option A) + use `.npy`
   instead of `.csv` for T and shift (heaviest files). Write a one-shot
   migration script.
3. **Future:** If storage or load time becomes a bottleneck on non-symlinked
   models, bundle A/b/start/extra into per-model NPZ (Option B hybrid).
