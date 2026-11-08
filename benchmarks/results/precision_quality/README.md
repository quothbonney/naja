# Precision-corrected quality runs

These are separate fresh runs using the verified optional F64 export binary
`build-h100-f64/naja`. Canonical F32 runs, metrics and tables are unchanged.
`protocol.json` records the binary and frozen estimator hashes. Run with:

```bash
sbatch --chdir="$PWD/benchmarks" benchmarks/scripts/precision_quality.sbatch
```

The iJR904 protocol matches the prior selected 20,000 retained + 1,000 warmup
samples per chain, 16 chains, TPB128, automatic thinning (50), rotate, seed101
first. If quality and all-reaction bounds pass, seeds202/303 follow. The iML1515
protocol starts with seed303 at 2,000 retained + 1,000 warmup, 16 chains,
TPB512, thinning1,000, rotate; seeds101/202 follow only after the first passes.

Every F64 sample set receives fresh external frozen-estimator diagnostics:
masked median split-Rhat≤1.01, median pooled bulk ESS≥100. The bounds test
includes every reaction and every retained sample, without masking, filtering
or clipping, at tolerance1e-7. Passing this tolerance is distinct from exact
zero-tolerance bound compliance; paired reports preserve several tolerances.

Each paired precision control backmaps the actual F64 coordinates and the
same points quantized to F32, both with a float64 affine map. The F64→F32
coordinates are also compared bitwise against the historical frozen F32
normalized output. This isolates export quantization where historical output
is reproduced. `precision_comparison.json` retains both bounds distributions,
maxima, reaction identities, shapes, hashes and bit mismatch counts.

Per-model `summary.json`/`table.csv` contain actual phase/wall/quality/bounds
results. Timings include the separately reported normalization cost and exclude
external diagnostics/precision analysis. Different-build, different-time F64
versus historical F32 wall times are not an export-only speed experiment.
The sampler phase includes allocation/init and is not a pure CHRR kernel timer.
The worker can resume completed F64 targets after interruption, retaining
incomplete prior targets under timestamped names.
