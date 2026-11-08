# Targeted same-GPU off/rotate controls

Updated 2026-11-08T20:00:08.791373+00:00; completed pairs **1/13**.

All 13 original conditions where the Naja+rotation pipeline passed and inherited CPU failed the 50k cap were declared before these controls. These are targeted posthoc category controls, not a representative panel or new independent validation seeds. Each pair uses the original selected Naja retained budget, seed101, 16 chains, 1,000 warmup, thinning1,000, TPB512 and the same verified F64 binary and conditional model/mask. Only the barrier flag changes. Pair order alternates; CUDA physical UUID is checked within each allocation. All cases, including new off passes and rotation failures, remain in the denominator.

Completed-pair contingency: {'rotate_only': 1}. Unfinished: 12.

| Gene KO | Retained/chain | Off Rhat | Rotate Rhat | Off ESS | Rotate ESS | Outcome |
|---|---:|---:|---:|---:|---:|---|
| b0003 | 1,000 | 1.23726 | 1.03306 | 66.5 | 588.1 | rotate_only |
| b0009 | 10,000 | — | — | — | — | cancelled closeout |
| b0029 | 500 | — | — | — | — | cancelled closeout |
| b0073 | 50,000 | — | — | — | — | cancelled closeout |
| b0085 | 1,000 | — | — | — | — | cancelled closeout |
| b0088 | 1,000 | — | — | — | — | cancelled closeout |
| b0091 | 1,000 | — | — | — | — | cancelled closeout |
| b0096 | 1,000 | — | — | — | — | cancelled closeout |
| b0154 | 1,000 | — | — | — | — | cancelled closeout |
| b0181 | 1,000 | — | — | — | — | cancelled closeout |
| b0918 | 1,000 | — | — | — | — | cancelled closeout |
| b2530 | 5,000 | — | — | — | — | cancelled closeout |
| b3198 | 1,000 | — | — | — | — | cancelled closeout |

The user closed this extension after its first completed pilot. The remaining 12 planned pairs were cancelled, preserving partial files. They are not sampler failures and no full13-case conclusion is supported. No further controls will launch from these arrays.


Pass gate: median classical split-Rhat≤1.05, pooled median bulkESS≥100, zero original/conditional-bound violations at1e-7 and maximum mass-balance residual≤1e-7. Compute timings use native GPU sampler phases; correction and secondary process times are separate. Only full-budget endpoints are diagnosed; these controls do not estimate time-to-quality.

[Frozen selection/protocol](protocol.json) · [Complete source/status JSON](summary.json) · [Per-mode CSV](source.csv) · [Paired CSV](paired.csv) · [Exact Slurm commands](jobs.json).
