# Common reaction diagnostic mask (metrics schema 2)

Primary reaction-space Rhat/ESS aggregation uses a **model-derived** mask, shared by Naja and every CPU baseline. Numerical nullspace/rounding residuals can give a truly blocked reaction a tiny but nonzero row of T; diagnosing this numerical noise as a freely varying reaction changes medians substantially.

`diagnostic_mask.py` selects candidate reactions with max(abs(T row)) <= 1e-6. It then solves two original-model GLPK LPs per candidate to minimize/maximize that reaction's flux, with feasibility tolerance 1e-9 and no added objective/growth constraint. Only successful finite LP flux ranges <= 1e-7 are excluded. Wider ranges and failed/unresolved LPs stay active. These are **solver-detected nearconstant reactions**, not an exact arithmetic proof of structural blockage. A source and geometry SHA256 pin each mask to the preprocessed model.

Cached artifacts per model:

- `active_reactions.npy`: boolean shared diagnostic mask.
- `diagnostic_mask.json`: original-model and geometry hashes, tolerances, policy, solver version/status, per-reaction min/max/range, and measured mask-building wall time.

Schema-2 `metrics.json` applies this mask plus exclusion of exactly constant sampled reactions. It retains the old method-dependent 1e-12 numerical-range aggregation separately as `legacy_empirical_numerically_variable_summary`. It also records the observed range of excluded reactions, including how many exceed 1e-7, so backmap numerical residue remains visible.

**All original bound checks still use every reaction and every sample.** Diagnostic exclusions never filter samples or erase violations. Both tolerance-1e-7 and exact zero-tolerance violation fractions remain in the metrics.

Initial mask costs and excluded counts: e_coli_core 8/95 (0.23 s); iAB_RBC_283 16/469 (0.31 s); iJR904 409/1075 (2.30 s); iMM904 693/1577 (5.50 s); iAF1260 851/2382 (8.58 s); iJO1366 878/2583 (10.0 s); iML1515 968/2712 (11.3 s). Several tiny-T-row candidates had LP ranges between 1e-5 and 6e-4 and were correctly retained, demonstrating why T-row threshold alone was insufficient.

Recompute stale completed-run metrics:

```bash
# From benchmarks/, after the model masks are ready:
env-cpu/bin/python scripts/recompute_metrics.py --submit --concurrency 4
```

This writes an auditable Slurm-array manifest, snapshots immutable old `metrics.schema1.json` / `reaction_metrics.schema1.csv` files, then atomically writes schema 2. It rechecks completion at execution and serializes recomputations per output directory. A final sweep catches records produced by older jobs. The report excludes old/unverified diagnostic schemas from primary quality cells and speedup ratios.

`test_metrics.py` verifies IID versus separated-chain diagnostics, Naja interleaved layout, affine backmapping, independent bound counts, and exclusion of a solver-masked reaction despite deliberately nonzero 1e-9 backmap noise. The saved verification log is `logs/metrics-verification.log`.
