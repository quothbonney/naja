# Conservative 16-chain held-out validation

Updated 2026-11-08T19:39:39.514062+00:00. Complete: **True**; all six pass: **True**.

Budgets were fixed before these new seeds: CPU 15,000 retained draws/chain at thinning 2,000; Naja 3,000 retained at thinning 1,000, rotation, TPB 512. Both use 16 chains + 1,000 warmup. CPU 16 physical workers reserve 32 logical CPUs. New seeds: 444001, 555001, 666001. The same exact cached 581-D target and conservative 478-exclusion assay apply to both. No extensions or seed-dependent retuning are allowed.

| Method | Seed | Status | Compute s | Median split-Rhat | Median bulk ESS | ESS/compute s |
|---|---:|---|---:|---:|---:|---:|
| cpu | 444001 | PASS | 21.2532 | 1.005542 | 3614.4 | 170.06 |
| gpu | 444001 | PASS | 10.3878 | 1.004367 | 4286.9 | 412.69 |
| cpu | 555001 | PASS | 21.0998 | 1.005907 | 3398.8 | 161.08 |
| gpu | 555001 | PASS | 10.3876 | 1.004975 | 4191.1 | 403.47 |
| cpu | 666001 | PASS | 21.0778 | 1.005689 | 3620.5 | 171.77 |
| gpu | 666001 | PASS | 10.4658 | 1.004782 | 4165.4 | 398.00 |

Median paired CPU/GPU compute-time ratio: **2.031×**; GPU/CPU ESS-rate ratio: **2.427×**.

Quality criterion:median classical split-Rhat≤1.01,median pooled rank-normalized bulkESS≥100, zero unmasked bound violations at1e-7. CPU timing is max accumulated draw-call critical path (binding/state-copy/descheduling included;IPC/output excluded). GPU uses native sampler phase (allocation/slack/RNG setup included;download/write excluded).

These larger fixed budgets were chosen from earlier bounded-extension evidence. Original10kCPU/2kGPUfixed failures remain visible separately; this study does not rescue their fixed-budget generalization or retroactively validate the legacy2.73× claim.

[Frozen protocol](protocol.json) · [Source runs and comparisons](summary.json) · [CSV](table.csv).
