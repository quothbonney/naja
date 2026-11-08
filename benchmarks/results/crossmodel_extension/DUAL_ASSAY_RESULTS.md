# Cross-model results under both diagnostic aggregation policies

Both use the same saved trajectories and unmasked original-bound checks. Conservative LP exclusions on cached A/b do not make a median-Rhat criterion uniformly harder: adding unresolved tiny reactions can lower the median. Budgets and comparisons remain assay-specific.

| Model | Method | Original-model mask pass/tested | Cached-target mask pass/tested |
|---|---|---:|---:|
| iML1515 | cpu_guideline | 3/3 | 3/3 |
| iML1515 | cpu_thin2000_n12000 | 2/3 | 3/3 |
| iML1515 | cpu_thin2000_n15000 | 3/3 | 3/3 |
| iML1515 | gpu_f64 | 3/3 | 3/3 |
| iML1515 | gpu_f64_tpb128 | 3/3 | 3/3 |
| iAF1260 | cpu_thin2000_n10000 | 3/3 | 3/3 |
| iAF1260 | gpu | 3/3 | 3/3 |
| iAF1260 | gpu_f64_tpb128 | 3/3 | 3/3 |

| Assay | Model | Tested CPU | Tested GPU | CPU/GPU time | GPU/CPU ESS/s |
|---|---|---|---|---:|---:|
| Original-model LP | iML1515 | cpu_thin2000_n15000 | gpu_f64 | 3.4912 | 2.9119 |
| Original-model LP | iAF1260 | cpu_thin2000_n10000 | gpu | 0.9671 | 1.8009 |
| Cached-target LP | iML1515 | cpu_thin2000_n12000 | gpu_f64 | 2.5648 | 3.2255 |
| Cached-target LP | iAF1260 | cpu_thin2000_n10000 | gpu | 0.9671 | 3.6356 |

These are finite tested configurations, with fresh selected-budget timings preferred over an earlier timing at an identical configuration. Neither assay changes sample arrays, feasibility tolerance, or sampler math. Original-model LP exclusions and cached-target LP exclusions have different mathematical domains; neither table is silently substituted for the other.
