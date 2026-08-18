# Contributing to Naja

Thanks for helping improve Naja. Small, focused pull requests are easiest to
review and safest to validate on GPU hardware.

## Development expectations

- Keep pull requests scoped to one behavioral change and include a clear
  description of the model layout and command line used to exercise it.
- Add or update a regression test for bug fixes and numerical changes. CPU-only
  tests belong in the default CTest suite; GPU-kernel tests must be guarded by
  `NAJA_ENABLE_GPU_TESTS`.
- Do not commit model data, sample output, Gurobi installations or licenses.
  These can be large, proprietary, or sensitive. Use small synthetic fixtures
  in tests instead.
- Preserve the documented model contract. If a format change is necessary,
  document backward compatibility and update `docs/model-format.md`.

## Validation before opening a pull request

On a CUDA-capable Linux host with Eigen, Gurobi and zlib installed:

```bash
./scripts/build.sh clean
ctest --test-dir build --output-on-failure

# Run only where a CUDA device is available.
cmake -S . -B build-gpu -DNAJA_ENABLE_GPU_TESTS=ON
cmake --build build-gpu -j"$(nproc)"
ctest --test-dir build-gpu -R gpu --output-on-failure
```

Record the CUDA toolkit version, GPU model, and `NAJA_CUDA_ARCH` for any
performance or numerical result. Naja writes a run manifest and generated
configuration to each output directory; include those artifacts when reporting
sampling or reproducibility problems.

## Reporting issues

Include the exact command, Naja commit, CUDA driver/toolkit versions, GPU model,
and a minimal non-sensitive model or model-shape description. For numerical
results, include the validation output (`naja validate`) and whether fast math
was enabled (the release build enables it).
