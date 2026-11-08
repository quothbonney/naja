# Cross-model conservative cached-target reassay

Exact cached A/b LP range certificates; unresolved remain active; samples and sampler timings unchanged. Every original bound independently checked at1e-7.

CPU worker draw critical path includes bindings/statecopy/descheduling; GPU sampler includes allocation/slack/RNG initialization. Both exclude download/write; not identical pure-arithmetic counters.

| Model | Method | Passing / tested seeds | Median sampler compute proxy s |
|---|---|---:|---:|
| iML1515 | cpu_guideline | 3/3 | 69.0553 |
| iML1515 | cpu_thin2000_n12000 | 3/3 | 20.8185 |
| iML1515 | cpu_thin2000_n15000 | 3/3 | 28.3382 |
| iML1515 | gpu_f64 | 3/3 | 8.1171 |
| iML1515 | gpu_f64_tpb128 | 3/3 | 13.0376 |
| iAF1260 | cpu_thin2000_n10000 | 3/3 | 14.9873 |
| iAF1260 | gpu | 3/3 | 15.4979 |
| iAF1260 | gpu_f64_tpb128 | 3/3 | 23.1973 |

| Model | Selected tested CPU | CPU/GPU time | GPU/CPU ESS/s |
|---|---|---:|---:|
| iML1515 | cpu_thin2000_n12000 | 2.5648 | 3.2255 |
| iAF1260 | cpu_thin2000_n10000 | 0.9671 | 3.6356 |

Finite tested CPU/GPU configurations only; preserved failed groups remain visible. Original raw-model-mask results are separate.
