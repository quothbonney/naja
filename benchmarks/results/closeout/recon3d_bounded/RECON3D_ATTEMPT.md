# Bounded Recon3D sampling attempt

Updated 2026-11-08T21:20:29.998173+00:00. Status: **DID_NOT_MEET_TARGET**.

One H100 allocation, limited to59minutes, containing one pilot and one main run; no retries. CPU normalization, convergence diagnostics and all-point feasibility checks run after GPU release. The cached target has n=10,600 reactions,d=4,861 dimensions,m=18,477 inequalities. It uses PolyRound-library equality elimination but **no iterative MVE rounding**. No matched-quality CPU sampling speedup or global time-to-quality optimum is claimed.

| Configuration | Value |
|---|---|
| Sampler |512 chains,F64,barrier whitening,GPU A×Q,TPB128,seed101|
| Pilot |64warmup+64retained draws/chain,thinning810|
| Main |1,000warmup+800retained draws/chain|
| Main thinning |5409|
| Quality gate |Median classical split-Rhat≤1.01,median pooled bulkESS≥100,zero bound violations at1e-7,mass residual≤1e-7|

All10,600 reactions are active in the assay; no nearconstant/LP exclusions are used. Only exactly constant sampled reactions are omitted by the frozen estimator. Original-target FVA has minimum reaction width0.2. The2,723 pruned objects were strictly inactive inequality rows, not reactions. Cached equality elimination took128.8417s; MVE0s; total original cached preprocessing287.7203s includes FVA134.2945s.

Source commit: `fcc66fe63d5a6700786aebcf0b925374340e4c8b`; binary SHA256: `5c01c66c90f46565eaafe0927265d135bcfa89af50a017df6c51de70044afe42`. This FCC executable is immutable. It explicitly selects the optimized GPU rotation backend; the finalC9release changes the compatibility default, not this recorded run. Production wide-index/FP64 tests were passed before launch; no legacy experimental bit-parity flag was fabricated.

Pilot GPU sampler: 28.7482s; coordinate lag1 median 0.969697. The frozen rule requested thinning60,612 and capped it at5,409 to fit the GPU budget. This is a pilot-trained attempt, not an independent confirmatory replicate.

| Measured main endpoint | Value |
|---|---:|
| GPU sampler s |2702.853300|
| Barrier phase s |1.400100|
| Median split-Rhat |2.0143671405386683|
|95th percentile split-Rhat |3.1721502800915204|
| Median pooled bulkESS |831.6440378953325|
| Sample bound-violation fraction |0.0|
| Maximum original-bound violation |1.8038903704109543e-12|
| Maximum mass-balance residual |2.693951728360844e-10|
| Reactions diagnosed |10,600/10,600|
| Exactly constant sampled reactions |0|
| Retained points checked |409,600|

All values above come from this actual main execution. The sampler includes discarded warmup. Barrier, output, normalization and CPU diagnostics are separate; no prefix wall-time interpolation is used.

[Frozen protocol](protocol.json) · [Main result](summary.json) · [Capture state](capture.json) · [Pilot decision](pilot_decision.json) · [Exact jobs/commands](jobs.json) · [Resource accounting](slurm_accounting.psv).
