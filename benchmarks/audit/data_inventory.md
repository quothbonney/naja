# Model and historical data inventory

Audit date: 2026-11-07 UTC. Repository: `naja-test-jack/naja`, branch `publication-prep`, SHA `23873f01d3268a79bbf84499a66adce5e88cc6cb` (one reachable commit).

## Locally available inputs

- The repository contains **no metabolic models**, rounded polytopes, raw benchmark results, paper source/PDF, or knockout corpus. `data/` contains only `petitprince.jpg`.
- `rs10k_build` is not present. The documented path `/mnt/home/jack/rs10k_build/` does not exist on this server. `/storage/jdcarson/models/iJO1366`, used by environment-dependent tests, does not exist either.
- A bounded directory-name search under the current user's home (depth 3) found no other Naja/model/rs10k/Reconstruction directories beyond the current checkout. `/mnt/vast` is empty. Other users' private home directories were not traversed.
- No original list of nine benchmark models is present. No alternative Naja revisions are locally available in Git history.

All nine proposed public models have now been downloaded consistently from the official legacy archive using `scripts/fetch_models.py` (24 MB). SHA256/source/timestamp/count manifest: `data/raw/manifest.json`. Cached downloads are checksum-verified.

Consequently, public unconditioned models can be rebuilt, but blocks 2/5/6 cannot reproduce the original conditioned panel or 8,074-variant library without its source definitions. A newly generated knockout panel must be labeled as such. The original unique-signature count cannot be settled from an unrelated generated panel.

## Public model catalog

The live official BiGG catalog is reachable at <https://bigg.bio/api/v3/models>; its response is saved as `benchmarks/data/bigg_model_catalog.json`, with source URL alongside. This captures public names/counts, **not** reduced sampling dimension. The legacy BiGG archive also serves model pages at <https://biggdata.ucsd.edu/models/Recon3D> and documents downloads at <https://biggdata.ucsd.edu/data_access>.

Candidate nine-model panel (user's proposed panel, not recovered historical panel):

| Model ID | Reactions n | Metabolites | Status |
|---|---:|---:|---|
| e_coli_core | 95 | 72 | Live catalog |
| iAB_RBC_283 | 469 | 342 | Live catalog |
| iJR904 | 1075 | 761 | Live catalog |
| iAF1260 | 2382 | 1668 | Live catalog |
| iJO1366 | 2583 | 1805 | Live catalog |
| iML1515 | 2712 | 1877 | Live catalog |
| iMM904 | 1577 | 1228 | Live catalog |
| RECON1 | 3742 | 2766 | Exact capitalization is RECON1; live catalog |
| Recon3D | 10600 | 5835 | Legacy official model page; archive JSON download verified |

Do not sort by reaction count and call it dimension order. Compute d only after identical equality elimination/blocked-reaction handling. Raw reaction counts include fixed/blocked reactions.

The catalog reports RECON1 n=3742 whereas the frozen legacy JSON contains n=3741; the downloaded manifest is authoritative for this campaign.

Use a single explicitly frozen BiGG model vintage when possible; the live service has 2026 updates whereas the legacy Recon3D archive page reports 2019. Save source URL, UTC retrieval time, SHA256, raw bounds, objective and medium for every downloaded model. An endpoint substitution must be recorded. Recon2.2 was not chosen: RECON1 is a public catalog entry and the user permits either.

## Existing reusable preprocessing components

- `tools/pack_model.py`: `write_bundle(...)` writes float64 `polytope.npz` with A/b/start/T/shift plus `manifest.json`; `extra.npz` holds per-condition delta rows. This is suitable for new PolyRound export without generating large CSVs.
- No SBML/COBRA-to-PolyRound preprocessing script is included. `sample prepare` only inherits an already rounded base and projects tightened reaction bounds through T/shift; it does not replace equality elimination and full PolyRound rounding.
- `docs/model-format.md` defines full backmap `x = T z + shift` and requires GEM reaction IDs/lower/upper bounds for reaction-space checks.
- `slurm/README.md` points to external `reround_ijo1366.py` and `migrate_to_RS_V3.py`; neither script is included.

Preserve all PolyRound affine transformations when exporting T/shift. Record raw n, reduced d, and final inequality m separately. Save independent preprocessing times; reuse the exact same rounded polytope for Naja and hopsy comparisons.

## Historical claims require verification

`docs/technical-summary.md` is the only local source of production numbers. It describes a 'RECON-family' model with **d=582**, but `docs/model-format.md` identifies **iJO1366 RS_V3** with d=582. It reports 8,074 conditions, eight A100s, 7.7 h wall, 34.2 GPU-h, 403.7 million samples and 940 GB output, without raw records. Treat these as unverified documentation claims, not recovered benchmark evidence. The 7,836 count and single-row-among-10,335 example do not appear in the checkout.

For new results, distinguish allocated GPU-hours (wall × allocated GPUs) from measured active GPU time. The documented 7.7 h × 8 GPUs equals 61.6 allocated GPU-h, so its 34.2 GPU-h is a different measure and cannot be compared without definition.

## Missing user inputs

1. Original nine-model identities and exact model medium/objective convention, if historical reproduction is required.
2. `rs10k_build`, original knockout definitions/model bounds, and a model-list manifest for the 8,074 entries.
3. Raw historical runs/paper tables if exact claims are to be checked.
4. A valid Gurobi license if LP preprocessing requires Gurobi; alternatively validate a supported non-Gurobi solver path.

## Exact knockout geometry warning

Exact knockout bounds (lb=ub=0) generally add affine equalities and reduce dimension. An inherited base coordinate system then need not have a strictly feasible interior. Barrier correction alone cannot repair a rank-deficient feasible set: eliminate the new equalities, or explicitly label a positive-width relaxed target. `extra-constraint-eps .01` widens each extra RHS; filtering afterward against exact fixed bounds need not preserve many samples. Report pre-filter validity, acceptance fraction, and numerical tolerance independently. Do not infer exact uniform KO sampling from a post-filter zero-violation count.
