# Task2: production barrier phase across eight models

Production commit`fcc66fe63d5a6700786aebcf0b925374340e4c8b`, binarySHA256`5c01c66c90f46565eaafe0927265d135bcfa89af50a017df6c51de70044afe42`.

16 chains, 8 retained draws, warmup 0, thinning 1, TPB 128, barrier rotate, float64 export, host prefault 0; phase timing only, not convergence

| Model | d | m | Median barrier phase s | Min–max s | Historical CPU MVE-round s |
|---|---:|---:|---:|---:|---:|
| e_coli_core | 24 | 36 | 0.067900 | 0.066900–0.067900 | 0.102 |
| iAB_RBC_283 | 130 | 188 | 0.074900 | 0.073200–0.075000 | 27.816 |
| iJR904 | 232 | 291 | 0.075600 | 0.070400–0.079400 | 30.396 |
| iMM904 | 288 | 413 | 0.074500 | 0.074200–0.082400 | 47.113 |
| iAF1260 | 524 | 671 | 0.094300 | 0.092400–0.096200 | 61.085 |
| iJO1366 | 581 | 750 | 0.097700 | 0.095400–0.099000 | 81.811 |
| iML1515 | 635 | 801 | 0.095500 | 0.094900–0.097400 | 79.262 |
| RECON1 | 931 | 2120 | 0.113200 | 0.111800–0.154200 | 209.929 |

The measured phase includes GPU math, transfers and host work inside barrier_rotation_s. This is additional correction on already-rounded cached geometry. The historical CPU column is original PolyRound rounding on its original input; this barrier phase does not replace full base preparation. CPU references are single observations; e_coli_core used the login CPU and the other seven used the H100 node CPU.

[Per-seed phases and commands](task2_per_seed.csv) · [Full summary/provenance](task2_barrier_summary.json).
