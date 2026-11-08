# Same-GPU conditional barrier ablation

Updated2026-11-08T20:22:09.559908+00:00

All9 nontrivial exactKO cases that failed original rotate50k cap; everyone extended to200k, with original50k failures retained. Outcome-selected extension, not an unbiased panel.

16chains,1000warmup,thinning1000,TPB512,float64; GPU captures and CPU diagnostics use separate jobs; physical H100 may differ between pilot and fresh retime. Full200,000-draw pilot plus separately executed selectedbudget; no inferred timing. Median splitRhat<=1.05, bulkESS>=100, allbounds/massbalance1e-7.

| Condition | Mode | Status | Selected draws | Sampler s | Correction s | Rhat | ESS |
|---|---|---|---:|---:|---:|---:|---:|
| b0025 | rotate | observed_budget_pass_no_selected_retime_closeout | — | — | — | — | — |
| b0031 | rotate | no_quality_within_cap | — | 543.73 | 0.1548 | 1.0646 | 156.07 |
| b0052 | rotate | observed_budget_pass_no_selected_retime_closeout | — | — | — | — | — |
| b0103 | rotate | no_quality_within_cap | — | 537.1 | 0.1454 | 1.0549 | 174.68 |
| b0109 | rotate | interrupted_capture_closeout | — | — | — | — | — |
| b0120 | rotate | interrupted_capture_closeout | — | — | — | — | — |
| b0131 | rotate | not_run_closeout | — | — | — | — | — |
| b0133 | rotate | not_run_closeout | — | — | — | — | — |
| b0134 | rotate | not_run_closeout | — | — | — | — | — |

## Actual full-budget observations

These are directly measured full-budget runs, not selected-prefix timing estimates. Fresh selected-budget validation remains separately labeled above.

| Condition | Full retained budget | Status | Sampler s | Rhat | ESS |
|---|---:|---|---:|---:|---:|
| b0025 | 200000 | observed_budget_pass | 540.793 | 1.04371 | 225.0 |
| b0031 | 200000 | observed_budget_failed | 543.733 | 1.06462 | 156.1 |
| b0052 | 200000 | observed_budget_pass | 543.597 | 1.04838 | 194.4 |
| b0103 | 200000 | observed_budget_failed | 537.096 | 1.05490 | 174.7 |

[Rhat SVG](rhat_vs_chain_length.svg) · [PDF](rhat_vs_chain_length.pdf) · [ESS SVG](ess_vs_chain_length.svg) · [Curve CSV](curves.csv) · [Full result states](summary.json) · [Exact manifest and selection](manifest.json).

This is an outcome/geometry-selected hard-case study, not the original50-target denominator. Unfinished task states are last-recorded; verifySlurm accounting. Existing original results untouched. User closed out old studies; queued cases were canceled and no new selected GPU retimes will be launched. Completed full-budget observations remain distinct from fresh selected-budget validation.
