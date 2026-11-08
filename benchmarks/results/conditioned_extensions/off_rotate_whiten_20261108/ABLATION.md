# Same-GPU conditional barrier ablation

Updated2026-11-08T20:22:09.575095+00:00

All9 previous rotate50k failures, highest SVD-kappa remaining noncontrol target, plus finite-width capacity reference; this is exploratory outcome/geometry-selected, not an unbiased panel.

16chains,1000warmup,thinning1000,TPB512,float64; sameallocatedGPU within each condition. Full50,000-draw pilot plus separately executed selectedbudget; no inferred timing. Median splitRhat<=1.05, bulkESS>=100, allbounds/massbalance1e-7.

| Condition | Mode | Status | Selected draws | Sampler s | Correction s | Rhat | ESS |
|---|---|---|---:|---:|---:|---:|---:|
| b0025 | off | no_quality_within_cap | — | — | — | — | — |
| b0025 | rotate | no_quality_within_cap | — | — | — | — | — |
| b0025 | whiten | no_quality_within_cap | — | — | — | — | — |
| b0031 | off | no_quality_within_cap | — | — | — | — | — |
| b0031 | rotate | no_quality_within_cap | — | — | — | — | — |
| b0031 | whiten | no_quality_within_cap | — | — | — | — | — |
| b0052 | off | no_quality_within_cap | — | — | — | — | — |
| b0052 | rotate | no_quality_within_cap | — | — | — | — | — |
| b0052 | whiten | no_quality_within_cap | — | — | — | — | — |
| b0103 | off | no_quality_within_cap | — | — | — | — | — |
| b0103 | rotate | no_quality_within_cap | — | — | — | — | — |
| b0103 | whiten | no_quality_within_cap | — | — | — | — | — |
| b0109 | off | not_run_closeout | — | — | — | — | — |
| b0109 | rotate | not_run_closeout | — | — | — | — | — |
| b0109 | whiten | not_run_closeout | — | — | — | — | — |
| b0120 | off | not_run_closeout | — | — | — | — | — |
| b0120 | rotate | not_run_closeout | — | — | — | — | — |
| b0120 | whiten | not_run_closeout | — | — | — | — | — |
| b0131 | off | not_run_closeout | — | — | — | — | — |
| b0131 | rotate | not_run_closeout | — | — | — | — | — |
| b0131 | whiten | not_run_closeout | — | — | — | — | — |
| b0133 | off | not_run_closeout | — | — | — | — | — |
| b0133 | rotate | not_run_closeout | — | — | — | — | — |
| b0133 | whiten | not_run_closeout | — | — | — | — | — |
| b0134 | off | not_run_closeout | — | — | — | — | — |
| b0134 | rotate | not_run_closeout | — | — | — | — | — |
| b0134 | whiten | not_run_closeout | — | — | — | — | — |
| b0003 | off | not_run_closeout | — | — | — | — | — |
| b0003 | rotate | not_run_closeout | — | — | — | — | — |
| b0003 | whiten | not_run_closeout | — | — | — | — | — |
| PGI_capacity | off | not_run_closeout | — | — | — | — | — |
| PGI_capacity | rotate | not_run_closeout | — | — | — | — | — |
| PGI_capacity | whiten | not_run_closeout | — | — | — | — | — |

With only coordinate moves and fixed thinning, exact CHRR is invariant under positive diagonal scaling after a fixed rotation: x=QSw and a coordinate step along S_j e_j spans the same physical chord as e_j after rescaling its uniform step. Thus whitening alone is a numerical/control ablation here; changes in finite-precision trajectories do not establish a different exact mixing kernel. Pair/dense directions can change under whitening.

[Rhat SVG](rhat_vs_chain_length.svg) · [PDF](rhat_vs_chain_length.pdf) · [ESS SVG](ess_vs_chain_length.svg) · [Curve CSV](curves.csv) · [Full result states](summary.json) · [Exact manifest and selection](manifest.json).

This is an outcome/geometry-selected hard-case study, not the original50-target denominator. Unfinished task states are last-recorded; verifySlurm accounting. Existing original results untouched. User closed out old studies; queued cases were canceled and no new selected GPU retimes will be launched. Completed full-budget observations remain distinct from fresh selected-budget validation.
