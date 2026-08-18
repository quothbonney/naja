# GPU Coordinate Hit-and-Run in Naja

This note explains the mathematics behind Naja's sampler and how that
mathematics becomes a CUDA kernel. It assumes familiarity with convex polytopes,
Markov chains, and basic linear algebra, but not with this particular codebase.
It describes the implementation in `src/gpu/chr.cu` and the preparation work in
`src/engine/job_sampling.cu`.

## 1. The sampling problem

Naja samples a bounded polytope in a reduced coordinate space,

\[
  P = \{x \in \mathbb{R}^d : Ax \le b\},
\]

where `A` has \(m\) rows. In metabolic-flux work, this is normally a
pre-rounded representation of a flux polytope. If requested, Naja maps samples
back to reaction space afterward using an affine map \(v = Tx + q\).

The target distribution is uniform over \(P\). Directly generating independent
uniform points is generally impractical in the dimensions of interest, so Naja
runs several Markov chains and applies coordinate hit-and-run (CHR).

## 2. One hit-and-run step

At a current feasible point \(x\), choose a direction \(u\), then intersect the
line \(x + \alpha u\) with the polytope. The next point is drawn uniformly from
that chord.

For row \(a_i^\top x \le b_i\), define the current slack and directional
derivative

\[
  s_i = b_i - a_i^\top x, \qquad q_i = a_i^\top u.
\]

Feasibility after a move requires \(s_i - \alpha q_i \ge 0\). Therefore each
row supplies either an upper or lower bound:

\[
  \alpha \le \frac{s_i}{q_i} \quad (q_i > 0), \qquad
  \alpha \ge \frac{s_i}{q_i} \quad (q_i < 0).
\]

Taking the tightest bounds gives \([\alpha_{\min},\alpha_{\max}]\), and one
uniform variate \(r\) gives

\[
  \alpha = \alpha_{\min} + r(\alpha_{\max}-\alpha_{\min}),
  \qquad r \sim \operatorname{Uniform}(0,1).
\]

For any fixed direction, uniform sampling along this chord leaves the uniform
distribution on \(P\) invariant. Naja uses mixtures of directions that are
independent of the current chain state and symmetric in sign; these mixtures
retain the same invariant distribution. Direction choice changes mixing speed,
not the target distribution.

### The ratio form used by the kernel

The kernel stores \(q_i/s_i\), not \(s_i/q_i\). This makes each constraint's
work a multiply/divide on quantities it already has in hand. Let
\(\rho_i=q_i/s_i\). Then the extreme positive and negative \(\rho_i\) values
invert to the two chord endpoints:

\[
  \alpha_{\max} = \frac{1}{\max_i \rho_i}, \qquad
  \alpha_{\min} = \frac{1}{\min_i \rho_i}.
\]

Rows with \(q_i=0\) do not constrain the chord. The implementation handles
zero slack explicitly so a boundary case produces the appropriate infinite
ratio rather than a NaN.

## 3. Why coordinate directions

In plain CHR, \(u=e_j\) for a coordinate \(j\) chosen uniformly. A step then
needs only column \(j\) of \(A\): \(q_i=A_{ij}\). This has two useful
consequences for a GPU:

1. It avoids forming a dense matrix-vector product for every Markov step.
2. With Naja's column-major matrix layout, a block can repeatedly read one
   column while its threads distribute the \(m\) constraint rows.

Coordinate directions can mix poorly when the polytope is tilted relative to
the chosen axes. Naja therefore supports a controlled mixture of additional,
still sparse directions:

| Direction | Vector \(u\) | Purpose |
|---|---|---|
| Coordinate | \(e_i\) | Lowest-cost baseline move. |
| Pair | \((e_i-e_j)/\sqrt{2}\), or a rotated two-coordinate axis | Cross-coordinate motion in coupled planes. |
| k-sparse | Up to 32 randomly chosen coordinates with signs \(\pm1/\sqrt{k}\) | Occasional escape from axis-aligned behavior. |
| Dikin | A precomputed dense eigenvector of a constraint-derived Hessian | Escape from directions pinched by newly added constraints. |

The pair and k-sparse moves still calculate \(q_i=a_i^\top u\) only from the
few nonzero components. Dikin moves precompute \(Av\) once during setup, so a
dense direction costs a read rather than a fresh dense dot product in the hot
loop. Pair, k-sparse, and Dikin direction signs are randomized to preserve
symmetry.

## 4. The CUDA decomposition: one block, one chain

Naja maps one independent Markov chain to one CUDA thread block. The chain
position \(x\in\mathbb{R}^d\) lives in shared memory because every constraint
calculation depends on it. The much longer slack vector lives in global memory,
with one column per chain.

For a proposed direction, thread `t` visits rows

\[
  t,\ t+T,\ t+2T,\ldots
\]

where \(T\) is the block size. It computes local extrema of \(q_i/s_i\).
`cub::BlockReduce` combines these partial extrema into the global chord bounds.
Thread 0 draws \(\alpha\) with its per-chain cuRAND state and updates the shared
position; the block then updates its assigned slack entries in parallel.

This split is important: the arithmetic cost per step is \(O(m)\), but the
constraint rows are processed concurrently. Chains are independent, so their
blocks provide a second level of parallelism. The usual default is 128 threads
per block, with compile-time specializations from 32 through 1024 threads.

## 5. Incremental slack updates

Recomputing \(b-Ax\) from scratch after every step would add a full matrix-vector
product to every iteration. Instead, the kernel initializes

\[
  s = b-Ax
\]

once and applies the exact recurrence

\[
  s \leftarrow s - \alpha Au.
\]

For coordinate and sparse moves, \(Au\) is assembled directly from the relevant
columns of \(A\); for Dikin moves it is precomputed. This is the central
throughput optimization: the same row-wise work used to find the chord also
updates feasibility afterward.

Floating-point roundoff accumulates in any recurrence. `--resync-interval N`
asks the kernel to recompute \(s=b-Ax\) every \(N\) steps. It is a numerical
diagnostic and safeguard, not a free optimization: resynchronization costs
\(O(md)\), so the default is off.

## 6. Rounding and the log-barrier geometry

CHR is sensitive to anisotropy: a long thin polytope can require many steps to
travel between its ends. Naja can form a local geometric diagnostic at a feasible
reference point \(x_c\), using the log-barrier Hessian

\[
  H(x_c) = A^\top\operatorname{diag}(s^{-2})A.
\]

Facets with small slack contribute more strongly, so \(H\) identifies locally
pinched and coupled directions. Naja computes this matrix as
\(B^\top B\), with rows \(B_{i:}=a_i^\top/s_i\), on the GPU. It clamps
exceptionally small slacks and adds a tiny ridge before eigendecomposition for
numerical stability.

With \(H=Q\Lambda Q^\top\), `--barrier-rotate` changes coordinates by the
orthogonal basis \(Q\). This preserves volume and chord lengths but aligns axes
with the local barrier geometry. `--barrier-whiten` additionally rescales by
\(\Lambda^{-1/2}\), with eigenvalue floors to prevent extreme scale factors.
The transformed constraints and start point are passed to the same CHR kernel;
the inverse transformation is composed into the backmap before output.

This is preconditioning, not a substitute for validation. The Hessian is local,
and conditioning can make geometry difficult far from the reference point.

## 7. Dikin escape directions for conditioned models

Conditioned models often append a few restrictive rows to a well-rounded base
polytope. Naja optionally builds a Hessian from the tight added rows, selects its
largest-eigenvalue directions, and occasionally uses those directions in the
CHR mixture. Intuitively, these are normal combinations associated with the
newly narrow directions. The vectors and their products with \(A\) are fixed
before sampling, so they remain valid symmetric proposal directions throughout
the run.

## 8. Chains, thinning, and output

Each block owns a separate PRNG state and chain. Naja advances a chain by the
requested thinning interval between retained samples. The stored `.npy` array is
float32 with shape `(n_samples, dimension)`; internally, sampling calculations
use double precision.

Thinning reduces serial correlation in stored output but does not create
independent samples. Run multiple chains and use `naja validate` to inspect
effective sample size, split-\(\hat R\), feasibility, and chord statistics.
For a new model family or GPU architecture, treat these diagnostics and the GPU
correctness tests as part of the experiment—not as an afterthought.

## 9. Reading the code beside this note

- `src/gpu/chr.cu` — direction choice, block reduction, step draw, and slack recurrence.
- `src/engine/job_sampling.cu` — feasibility gates, affine-hull reduction, transformations, and launch setup.
- `src/rounding/barrier_rotation.cu` — GPU formation and eigendecomposition of the barrier Hessian.
- `src/rounding/dikin_directions.cu` — construction of escape directions from added constraints.
- `src/validate/` — diagnostics applied to saved chains.

For implementation-wide detail and historical design context, see the
[technical summary](technical-summary.md). For the on-disk input contract, see
the [model format reference](model-format.md).
