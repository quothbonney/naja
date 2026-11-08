# Cross-model CPU budget extensions

Fixed predeclared fresh budget extensions; original10k failures retained. CPU maximum accumulated worker C++draw time includes bindings/statecopy/descheduling, excludes IPC/output. GPU sampler includes allocation/slack/RNG. All original bounds checked at1e-7.

| Model | Retained draws | Passing / tested seeds | CPU worker s | GPU sampler s | CPU/GPU time | GPU/CPU ESS/s |
|---|---:|---:|---:|---:|---:|---:|
| iML1515 | 12000 | 2/3 | — | — | — | — |
| iML1515 | 15000 | 3/3 | 28.3382 | 8.1171 | 3.4912 | 2.9119 |
| iAF1260 | 10000 | 3/3 | 14.9873 | 15.4979 | 0.9671 | 1.8009 |

Every shown time is an actual execution at the stated budget. Failed and partial groups have no speedup estimate. These are finite tested configurations, not globally optimal CPU/GPU settings. See summary.json and per-seed JSON for commands, parity and geometry hashes.
