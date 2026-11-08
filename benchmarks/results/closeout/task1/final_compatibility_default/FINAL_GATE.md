# Final compatibility-default gate

Commit `c9e5db8b8ed02c692b36e74e14b71eedcd871476`, binary SHA256 `2fc4654b47d9a2f8a3fb46df6fd5a70dc1fe1bbbb48f758d816a2b9a565b10b1`.

- All seven correctness/index/FP64 tests passed.
- All five real failure probes returned nonzero without writing samples.
- Default CPU rotation preserves every iJO1366 F32 bit against the public reference plus seed-only shim.
- Explicit GPU rotation preserves every bit of the prior FCC GPU trajectory; it remains a distinct numerical path from CPU rotation.
- Automatic prefault resolves 8 threads within the 8-CPU allocation on the short64-chain F64 smoke.

The separately saved FCC RECON1 four-export study remains applicable to the explicit GPU path. At1e-7 it records nonzero original-bound violations from the cached geometry; F64 export alone does not repair them.

[Full summary](summary.json) · [CTest output](gpu-correctness-tests.log) · [Failure probes](negative_probes.json).

A separate positive large-array verification also passed: 2,147,484,160 doubles
(17,179,873,280 bytes), sentinel indices above 2^31, full D2H copy, full F64
streaming write to /dev/null, and cuBLAS backmap from the wide input with all
8,388,610 output values checked. This links the final build's production GPU
objects and writes no large payload to disk.

[Wide allocation result](../wide_allocation_result.json) ·
[Wide backmap result](../wide_backmap_result.json) ·
[Linked source/object hashes](../wide-backmap-all-inputs.sha256).

The mirror large-output case also passed (job962, exit0): a tiny
2×4,194,305 input and512×2 transform produced a17,179,873,280-byte output.
Every one of its2,147,484,160 values was checked after cuBLAS GEMM, shift,
and full D2H copy. The verification is deliberately not a timing benchmark.
[Large-output result](../wide_output_result.json) ·
[Production object/source hashes](../wide-output-all-inputs.sha256) ·
[Slurm outcomes](../slurm-accounting.psv).
