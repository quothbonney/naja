# Task2: production barrier phase across eight models

Production commit`c9e5db8b8ed02c692b36e74e14b71eedcd871476`, binarySHA256`2fc4654b47d9a2f8a3fb46df6fd5a70dc1fe1bbbb48f758d816a2b9a565b10b1`.

16 chains, 8 retained draws, warmup 0, thinning 1, TPB 128, barrier rotate with explicit backend, float64 export, host prefault 0; phase timing only, not convergence

| Model | d | m | Median barrier phase s | Min–max s | Historical CPU MVE-round s |
|---|---:|---:|---:|---:|---:|
| e_coli_core | 24 | 36 | 0.064300 | 0.064100–0.064700 | 0.102 |
| iAB_RBC_283 | 130 | 188 | 0.080100 | 0.070800–0.153100 | 27.816 |
| iJR904 | 232 | 291 | 0.070200 | 0.069800–0.083000 | 30.396 |
| iMM904 | 288 | 413 | 0.071100 | 0.070400–0.071400 | 47.113 |
| iAF1260 | 524 | 671 | 0.087700 | 0.086200–0.087800 | 61.085 |
| iJO1366 | 581 | 750 | 0.088800 | 0.088100–0.089000 | 81.811 |
| iML1515 | 635 | 801 | 0.089600 | 0.089500–0.089700 | 79.262 |
| RECON1 | 931 | 2120 | 0.245900 | 0.105800–0.275700 | 209.929 |

The measured phase includes GPU math, transfers and host work inside barrier_rotation_s. This is additional correction on already-rounded cached geometry. The historical CPU column is original PolyRound rounding on its original input; this barrier phase does not replace full base preparation. CPU references are single observations; e_coli_core used the login CPU and the other seven used the H100 node CPU.

[Per-seed phases and commands](task2_per_seed.csv) · [Full summary/provenance](task2_barrier_summary.json).
