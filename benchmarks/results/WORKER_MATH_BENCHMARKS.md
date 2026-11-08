# Worker draw timing comparisons

Separate instrumented variant. Full retained-output SHA parity required for every case. Three-seed groups only; one-seed results remain partial. GPU sampler phases include GPUinit; genuine CHRRkernel-only comparisons require matching trace evidence.

| Model | CPU worker draw critical path(s) | GPU sampler phase(s) | Time ratioCPU/GPU | Paired ESS/s ratioGPU/CPU |
|---|---:|---:|---:|---:|
| iAB_RBC_283 | 0.660333 | 0.606 | 1.090 | 2.474 |
| iJO1366 | 15.9995 | 7.8132 | 2.048 | 1.923 |
| iML1515 | 69.0553 | 8.1171 | 8.507 | 6.657 |

CPU draw timing includes Python binding/state-copy and OS descheduling. GPU sampler phase includes allocation/slack GEMM/RNG initialization. Neither is labeled pure arithmetic-only; no IPC/output cost is hidden in the CPU draw counter.

iML1515 native thin2000 fixed10k control: 0/3 actually tested seeds pass; missing none. Cap failure does not mean eventual nonconvergence. This control is separate from timing comparisons.
