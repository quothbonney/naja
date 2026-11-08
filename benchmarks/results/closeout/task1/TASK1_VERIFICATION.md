# Task1: merged-build verification

Audited binary commit `fcc66fe63d5a6700786aebcf0b925374340e4c8b`, SHA256 `5c01c66c90f46565eaafe0927265d135bcfa89af50a017df6c51de70044afe42`. This is the GPU-rotation-default FCC build, retained immutably at `build-closeout/naja`. Later compatibility-default builds require their own final gate.

## Correctness and error propagation

All 7 CTests passed: five GPU correctness tests, GPU wide-index/error checks, and host overflow/FP64/profile checks. The backmap test includes 600,000 columns, beyond the previous 2-D launch grid limit. All 5 real negative probes returned nonzero without writing samples: invalid GPU, invalid backend, invalid TPB, unsupported sample-column axis, and actual CUDA allocation failure.

## Legacy trajectory comparison

Public commit 23873f0 has no explicit seed option. The reference therefore has exactly the saved one-line constant-seed 101 shim; it is not claimed to be an unmodified public binary. Both runs have 16 chains,1,000 warmup+10,000 retained draws/chain, thinning 96, TPB 128.

| Rotation backend | All saved F32 values bit-identical to reference | Different scalar values | Maximum coordinate difference |
|---|---|---:|---:|
| GPU A×Q (FCC default) | False | 101,983,508 | 9.16336966 |
| CPU A×Q (compatibility) | True | 0 | 0 |

The compatibility path preserves every F32 bit. GPU A×Q changes floating-point rounding and the long trajectory diverges; no default-backend byte-parity claim is made.

## RECON1 unmasked original-bound audit

Four actual executions, same 16-chain/seed 101/1,000 warmup/10k retained/thin 96/TPB 128 protocol and GPU A×Q backend. Every original reaction at every saved point was checked, with no sample filtering or reaction mask.

| Export | Warmup-inclusive samples | Fraction violating 1e-7 | Retained fraction violating 1e-7 | Maximum violation |
|---|---:|---:|---:|---:|
| RECON1_reduced_f32 | 176,000 | 0.60146023 | 0.59701875 | 2.357616692e-07 |
| RECON1_reaction_f32 | 176,000 | 0.60146023 | 0.59701875 | 2.357616751e-07 |
| RECON1_reduced_f64 | 176,000 | 0.60146023 | 0.59701875 | 2.357616734e-07 |
| RECON1_reaction_f64 | 176,000 | 0.60146023 | 0.59701875 | 2.357616734e-07 |

Both F64 runs cast back to their actual F32 counterparts exactly. Counterfactual F32 casts leave the bound-violation counts unchanged. The RECON1 floor here is approximately 2.36e-7 in the cached geometry/backmap, not repaired by F64 export. This is below the separately declared 1e-6 geometry tolerance but fails a zero-at1e-7 claim.

## Automatic host prefault

The 64-chain/four-draw short configuration smoke resolved 8 prefault threads within the 8-CPU affinity set. This checks the effective configuration, not convergence or speedup.

[Full summary](summary.json) · [CTest log](gpu-correctness-tests.log) · [Negative probes](negative_probes.json) · [Reference patch](reference-seed-only.patch) · [Reference source hashes](reference-source.json).
