# Same-GPU conditional barrier ablation

Updated2026-11-08T20:22:09.550315+00:00

Two highest saved-startSVD-Hessian-kappa targets among9 previous rotate50k failures (b0133,b0103), plus finite-width PGIcapacity, frozenbeforemixtureresults. Exploratory6 configurations; compare separately with currentwhitenpair0study.

16chains,1000warmup,thinning1000,TPB512,float64; sameallocatedGPU within each condition. Full50,000-draw pilot plus separately executed selectedbudget; no inferred timing. Median splitRhat<=1.05, bulkESS>=100, allbounds/massbalance1e-7.

| Condition | Mode | Status | Selected draws | Sampler s | Correction s | Rhat | ESS |
|---|---|---|---:|---:|---:|---:|---:|
| b0133 | whiten_fixed_pairs025 | no_quality_within_cap | — | — | — | — | — |
| b0133 | whiten_jacobi025_dikin015 | no_quality_within_cap | — | — | — | — | — |
| b0103 | whiten_fixed_pairs025 | not_run_closeout | — | — | — | — | — |
| b0103 | whiten_jacobi025_dikin015 | not_run_closeout | — | — | — | — | — |
| PGI_capacity | whiten_fixed_pairs025 | not_run_closeout | — | — | — | — | — |
| PGI_capacity | whiten_jacobi025_dikin015 | not_run_closeout | — | — | — | — | — |

[Rhat SVG](rhat_vs_chain_length.svg) · [PDF](rhat_vs_chain_length.pdf) · [ESS SVG](ess_vs_chain_length.svg) · [Curve CSV](curves.csv) · [Full result states](summary.json) · [Exact manifest and selection](manifest.json).

This is an outcome/geometry-selected hard-case study, not the original50-target denominator. Unfinished task states are last-recorded; verifySlurm accounting. Existing original results untouched. User closed out old studies; queued cases were canceled and no new selected GPU retimes will be launched. Completed full-budget observations remain distinct from fresh selected-budget validation.
