# hopsy binary optimization audit

The generic CPU reference uses the official hopsy 1.7.0 CPython 3.12 manylinux
wheel, reporting build commit `dadab7b`. Its module SHA256 is saved in
`hopsy_generic_module.sha256`. Disassembly of the installed 2.5 MB module
contained no `ymm` or `zmm` register operands, consistent with a generic SSE
build rather than AVX/AVX512 specialization. This is static evidence about the
binary, not a measured performance gain.

The exact official PyPI source archive is cached under `source/`; URL and
verified SHA256 are in `hopsy_source.json`. Source CMake explicitly forces
`HOPS_BUILD_NATIVE=OFF`, leaves Eigen vectorization enabled, and disables
OpenMP. The HOPS Release flags use `-O3`. The build-info string does not expose
compiler flags, so commit identity alone does not establish SIMD tuning.

The optional native experiment keeps all runtime packages and source version
matched, but builds on the actual compute-node Xeon Platinum 8462Y+ using
`-O3 -march=native -mtune=native`; no `-ffast-math` is added. Global CXX flags
cover both the Python binding and linked HOPS library. A separate
`env-cpu-native` and native wheel preserve the generic environment. Build tools
and runtime versions are locked separately in `cpu-native-requirements.lock.txt`.
The build script records compiler target flags, wheel/module SHA256, and an
analytic cube mean/variance/bounds smoke check before any comparison.

Native exploratory output is kept under `results/native_cpu`, with
`run.json` field `build_variant=native`, and is excluded from canonical results.
Only matched geometry, chains, retained draws, warmup and thinning comparisons
can quantify the gain. Native code may produce different floating-point paths;
bitwise identity is not required, but common quality and feasibility checks are.

The first native build reports commit string `9b787d9`, from the official
sdist's `.commit`, while the official wheel reports `dadab7b`. All eight Python
source files match byte for byte; compiled-source identity is not established
by that check. Therefore a third isolated environment, `env-cpu-sdist`, builds
the same verified source archive with generic `-O3` flags. Its paired pilot
controls source, compiler and runtime while changing native CPU flags. The
wheel-to-native comparison is an implementation comparison; attributing the
gain specifically to SIMD requires this same-source generic comparison.

The completed matched iJO1366 pilot uses one seed, 16 chains, 10,000 retained
draws after 1,000 warmup draws, thinning 2,000, and 16 workers. Source-generic
sampling took 44.132 seconds and native sampling 31.390 seconds: a 1.406-fold
gain with source/compiler/runtime controlled. Process times were 51.012 and
36.958 seconds (1.380-fold). Both have zero bound violations at 1e-7. Native
median split-Rhat is 1.009759 versus 1.010030 for generic, with similar median
bulk ESS (2,151 versus 2,192); that small threshold crossing is not evidence of
improved statistical efficiency. Three-seed passing-budget retiming remains
required for a fastest-CPU quality claim. Machine-readable rows are saved in
`results/native_cpu/pilot_table.csv` and `pilot_summary.json`.
