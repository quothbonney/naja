**Incompatible for the main same-target comparison:** independent raw-model
PolyRound has582 dimensions, whereas the declared cached conditional target has581.
These costs remain historical measurements; no workflow speedup may be inferred
from this comparator. A separate global rerounding of the exact cached A/b is
being prepared, with invertible-map and constraint-equivalence proof.

# Historical independent raw-model preparation — incompatible comparator

PGI capacity is exactly [-0.1, 0.1], a knockdown rather than a zero knockout.
Both full and inherited geometries pass the 1e-7 source/bounds/affine-map/pilot
validation. Conditional LP masks were independently recomputed and share a
conservative common exclusion of 878 reactions. No original sampling target
files were changed.

| Measured geometry setup | Algorithm phase s | Secondary standalone process s |
|---|---:|---:|
| Full fresh PolyRound | 134.899617 | 142.122677 |
| Inherited NPZ load and exact row assembly | 0.018819 | |
| Strict interior LP | 1.555838 | |
| Inherited complete preparation | 1.574657 (assembly + LP) | 3.045237 |

The inherited LP uses positive row normalization only inside the numerical
solver, preserving every saved halfspace. Its actual minimum original raw
slack is 4.87408e-10 and inscribed radius 0.0195258903450. A prior unnormalized
LP point failed the raw strict-interior gate before sampling; that failure
remains preserved.

GPU local correction is additional and remains unmeasured until its sampling
record completes. These geometry setup measurements therefore establish no
sampler speedup or equal-quality end-to-end win. One-time base preprocessing
156.830614 s is reported separately and must be amortized transparently across
conditions. Phase scopes differ from complete process scopes; no file-output
cost is converted into a mathematical kernel claim.

[Full preparation provenance](iJO1366/preparation.json) ·
[First failure log](iJO1366/shared-preprocess.failed477.log) ·
[Saved sampling progress](TABLE.md).

Both preparation paths ran in the same four-logical-CPU Slurm allocation with
BLAS/OMP thread counts fixed at one. This is the measured setup configuration,
not a globally optimized CPU preprocessing claim. Subsequent CPU sampling uses
a separate 16-physical-core allocation; hardware counts must not be conflated.
