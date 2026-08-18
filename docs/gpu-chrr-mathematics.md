# GPU Coordinate Hit-and-Run with Rounding in Naja

This note explains the mathematics behind Naja's sampler and how that
mathematics becomes a CUDA kernel. It assumes familiarity with convex polytopes,
Markov chains, and basic linear algebra, but not with this particular codebase.
It describes the implementation in `src/gpu/chr.cu` and the preparation work in
`src/engine/job_sampling.cu`.

## 1. The conditioned sampling problem

For a genome-scale model, Naja first eliminates equalities and rounds the
unperturbed base polytope once. This fixes a shared reduced coordinate system.
An intervention is translated through gene--protein--reaction rules into an
effect signature: the reaction bounds changed by that intervention. Each unique
effect signature defines a condition and adds condition-specific halfspaces
inside the shared coordinates.

The resulting conditioned polytope is

$$
  P_c = \{y \in \mathbb{R}^d : A_c y \le b_c\},
$$

where $A_c$ has $m$ rows. If requested, Naja maps samples from reduced space
back to reaction space with an affine map $v = Ty + q$.

The target is the uniform distribution over $P_c$. Directly generating
independent uniform points is generally impractical in the dimensions of
interest, so Naja runs several Markov chains and applies coordinate hit-and-run
with rounding (CHRR).

The GPU contribution is not simply “run several chains at once.” A naive GPU
port would still recompute $A_cy$ at every step and spend most of its time in
small, repeated matrix-vector products. Naja instead turns one CHR step into a
row-parallel reduction over the constraints, carries forward the information
needed by the next step, and keeps the chain's hot state on the device.

## 2. One hit-and-run step

At a current feasible point $y$, choose a direction $u$, then intersect the
line $y + \alpha u$ with the polytope. The next point is drawn uniformly from
that chord.

For row $a_{c,i}^\top y \le b_{c,i}$, define the current slack and directional
derivative

$$
  s_i = b_{c,i} - a_{c,i}^\top y, \qquad q_i = a_{c,i}^\top u.
$$

Feasibility after a move requires $s_i - \alpha q_i \ge 0$. Therefore each
row supplies either an upper or lower bound:

$$
  \alpha \le \frac{s_i}{q_i} \quad (q_i > 0), \qquad
  \alpha \ge \frac{s_i}{q_i} \quad (q_i < 0).
$$

Taking the tightest bounds gives $[\alpha_{\min},\alpha_{\max}]$, and one
uniform variate $r$ gives

$$
  \alpha = \alpha_{\min} + r(\alpha_{\max}-\alpha_{\min}),
  \qquad r \sim \mathrm{Uniform}(0,1).
$$

For any fixed direction, uniform sampling along this chord leaves the uniform
distribution on $P_c$ invariant. Naja uses mixtures of directions that are
independent of the current chain state and symmetric in sign; these mixtures
retain the same invariant distribution. Direction choice changes mixing speed,
not the target distribution.

### The ratio form used by the kernel

The kernel stores $q_i/s_i$, not $s_i/q_i$. This makes each constraint's
work a multiply/divide on quantities it already has in hand. Let
$\rho_i=q_i/s_i$. Then the extreme positive and negative $\rho_i$ values
invert to the two chord endpoints:

$$
  \alpha_{\max} = \frac{1}{\max_i \rho_i}, \qquad
  \alpha_{\min} = \frac{1}{\min_i \rho_i}.
$$

Rows with $q_i=0$ do not constrain the chord. The implementation handles
zero slack explicitly so a boundary case produces the appropriate infinite
ratio rather than a NaN.

## 3. The GPU trick: make the constraints the parallel work

For one chain, finding a chord is intrinsically a reduction over all $m$
halfspaces: every row can tighten one endpoint. That is the parallel axis Naja
uses. It does **not** assign one thread to one coordinate or one complete chain
step; it assigns a CUDA block to a chain and shares the constraint rows among
the threads in that block.

Before sampling, Naja computes the initial slack matrix for all chains with one
GPU matrix multiplication,

$$
  S = b_c\mathbf{1}^\top - A_cY,
$$

where each column of $Y$ is a chain state. The kernel then maintains $S$ rather
than repeatedly evaluating $A_cy$. For every MCMC step, a block performs this
fused sequence:

```text
thread 0: choose a direction and draw the chord-uniform variate
all threads: evaluate q_i / s_i for their rows
all threads: reduce local minima and maxima to the two chord endpoints
thread 0: update y in shared memory
all threads: apply s_i <- s_i - alpha q_i for their rows
```

The same projected direction $q_i=a_{c,i}^\top u$ is used twice: first to find
the chord and then to update slack. That reuse is the main algebraic saving.
Without it, every step would need a new $A_cy$ product after the position update.

### Why the reciprocal reduction matters

The direct expression for the chord needs a masked minimum over positive $q_i$
and a masked maximum over negative $q_i$. The kernel instead lets every thread
evaluate the same simple quantity, $\rho_i=q_i/s_i$, and runs an ordinary block
minimum and maximum. Their reciprocals are the lower and upper chord limits.

This removes sign-specific branches from the reduction itself and gives each
thread one streaming pass over its rows. The current implementation uses two
`cub::BlockReduce` operations, one minimum and one maximum, with locally
accumulated extrema. The reduction depth is logarithmic in the thread-block
size; the necessary $O(m)$ reads are distributed over the block.

### What stays where

| Quantity | Location during the hot loop | Reason |
|---|---|---|
| Current reduced-space point $y$ | Shared memory, one copy per block | Every row calculation reads it; coordinate moves update only one entry. |
| Slack column $s$ | Global memory, one column per chain | Usually much larger than shared memory; each thread owns disjoint rows. |
| Constraint matrix $A_c$ | Global memory, column-major | A coordinate move reads one column; pair moves read two. |
| Direction metadata and scalar step $\alpha$ | Shared memory | Thread 0 chooses once, then broadcasts to the block. |
| PRNG state | Global memory, one state per chain | Only thread 0 mutates it, avoiding synchronization of random draws. |

Naja's column-major matrix layout is deliberate. For a coordinate move, the
threads in a block access consecutive constraint rows of the selected column,
which is the natural contiguous access pattern for the matrix. A pair move is
two such reads. A dense Dikin move uses a precomputed column of $A_cV$ for the
same reason.

### Kernel pseudocode

For one block $k$ and one step, the implementation in `chr.cu` is equivalent to
the following. This is schematic: it omits barriers and the special move types,
but preserves the data dependencies that matter for performance.

```text
shared y[0:d], alpha, direction
global slack[0:m, chain=k]

thread 0: direction <- symmetric draw; u <- Uniform(0, 1)
synchronize block

for i assigned to this thread:
    q_i <- a_c,i^T direction
    local_min <- min(local_min, q_i / slack_i)
    local_max <- max(local_max, q_i / slack_i)

rho_min <- block_reduce_min(local_min)
rho_max <- block_reduce_max(local_max)

thread 0: alpha <- 1/rho_min + u * (1/rho_max - 1/rho_min)
thread 0: y <- y + alpha * direction
synchronize block

for i assigned to this thread:
    slack_i <- slack_i - alpha * q_i
```

One block owns a chain for the whole kernel launch, so the position does not
cross the host--device boundary between retained samples. A retained sample is
copied from shared memory to a device output matrix only after the requested
number of thinning steps.

## 4. Why coordinate directions

In plain CHR, $u=e_j$ for a coordinate $j$ chosen uniformly. A step then
needs only column $j$ of $A_c$: $q_i=A_{c,ij}$. This has two useful
consequences for a GPU:

1. It avoids forming a dense matrix-vector product for every Markov step.
2. With Naja's column-major matrix layout, a block can repeatedly read one
   column while its threads distribute the $m$ constraint rows.

Coordinate directions can mix poorly when the polytope is tilted relative to
the chosen axes. Naja therefore supports a controlled mixture of additional,
still sparse directions:

| Direction | Vector $u$ | Purpose |
|---|---|---|
| Coordinate | $e_i$ | Lowest-cost baseline move. |
| Pair | $(e_i-e_j)/\sqrt{2}$, or a rotated two-coordinate axis | Cross-coordinate motion in coupled planes. |
| k-sparse | Up to 32 randomly chosen coordinates with signs $\pm1/\sqrt{k}$ | Occasional escape from axis-aligned behavior. |
| Dikin | A precomputed dense eigenvector of a constraint-derived Hessian | Escape from directions pinched by newly added constraints. |

The pair and k-sparse moves still calculate $q_i=a_{c,i}^\top u$ only from the
few nonzero components. Dikin moves precompute $A_cv$ once during setup, so a
dense direction costs a read rather than a fresh dense dot product in the hot
loop. Pair, k-sparse, and Dikin direction signs are randomized to preserve
symmetry.

## 5. Block-per-chain execution and occupancy

Naja maps one independent Markov chain to one CUDA thread block. This gives a
chain a single sequential owner for its Markov state while exposing its $m$
constraint rows to parallel work. The chain position $y\in\mathbb{R}^d$ lives
in shared memory because every constraint calculation depends on it. The much
longer slack vector lives in global memory, with one column per chain.

For a proposed direction, thread `t` visits rows

$$
  t,\ t+T,\ t+2T,\ldots
$$

where $T$ is the block size. It computes local extrema of $q_i/s_i$.
`cub::BlockReduce` combines these partial extrema into the global chord bounds.
Thread 0 draws $\alpha$ with its per-chain cuRAND state and updates the shared
position; the block then updates its assigned slack entries in parallel.

This split is important: the arithmetic cost per step is $O(m)$, but the
constraint rows are processed concurrently. Chains are independent, so their
blocks provide a second level of parallelism. The usual default is 128 threads
per block, with compile-time specializations from 32 through 1024 threads. The
thread count is a practical balance: more threads shorten the row loop, while
the shared point, reduction workspace, register pressure, and available chains
limit occupancy. It is exposed as `--tpb` for hardware-specific tuning.

## 6. Incremental slack updates

Recomputing $b_c-A_cy$ from scratch after every step would add a full matrix-vector
product to every iteration. Instead, the kernel initializes

$$
  s = b_c-A_cy
$$

once and applies the exact recurrence

$$
  s \leftarrow s - \alpha A_cu.
$$

For coordinate and sparse moves, $A_cu$ is assembled directly from the relevant
columns of $A_c$; for Dikin moves it is precomputed. This is the central
throughput optimization: the same row-wise work used to find the chord also
updates feasibility afterward.

Floating-point roundoff accumulates in any recurrence. `--resync-interval N`
asks the kernel to recompute $s=b_c-A_cy$ every $N$ steps. It is a numerical
diagnostic and safeguard, not a free optimization: resynchronization costs
$O(md)$, so the default is off.

## 7. Amortized rounding and GPU barrier correction

CHR is sensitive to anisotropy: a long thin polytope can require many steps to
travel between its ends. Rather than reround every condition, Naja applies a
condition-specific correction in the shared reduced coordinate system. At an
interior point $y_c$ of the conditioned polytope, it forms the log-barrier
Hessian

$$
  H(y_c) = A_c^\top\mathrm{diag}(s^{-2})A_c.
$$

Tight conditioned constraints contribute more strongly, so $H$ identifies
locally pinched and coupled directions introduced by the effect signature.
Naja computes this matrix as $B^\top B$, with rows
$B_{i:}=a_{c,i}^\top/s_i$, on the GPU. It clamps
exceptionally small slacks and adds a tiny ridge before eigendecomposition for
numerical stability.

With $H=Q\Lambda Q^\top$, `--barrier-rotate` changes coordinates by the
orthogonal basis $Q$. This preserves volume and chord lengths but aligns axes
with the local barrier geometry. `--barrier-whiten` additionally rescales by
$\Lambda^{-1/2}$, with eigenvalue floors to prevent extreme scale factors.
The transformed constraints and start point are passed to the same CHR kernel;
the inverse transformation is composed into the backmap before output.

This is preconditioning, not a substitute for validation. The Hessian is local,
and conditioning can make geometry difficult far from the reference point.

### The second GPU-specific trick: do not reround every condition

The expensive global rounding is amortized across the knockout library: the base
model is rounded once, and every condition inherits that shared reduced-space
map. Reusing that map unchanged is not sufficient, because a small effect
signature can create a narrow, oblique conditioned polytope. Naja corrects the
local geometry of each condition with the barrier Hessian above rather than
solving a new global rounding problem.

This correction is particularly well matched to GPU dense linear algebra:

1. Compute the slacks at $y_c$ and clamp exceptionally small positive values.
2. Row-scale $A_c$ to form $B=\mathrm{diag}(s^{-1})A_c$.
3. Form $H=B^\top B$ with a symmetric rank-$k$ GPU operation.
4. Add a small ridge and compute the symmetric eigendecomposition on the GPU.
5. Rotate, and optionally whiten, the condition before launching CHR.

The first rounding is global and expensive; the per-condition operation is a
single matrix factorization that directly reflects the new tight halfspaces.
That is the mathematical and computational compromise that makes a large
conditioned corpus practical: shared global geometry plus fresh local
preconditioning.

## 8. Dikin escape directions for conditioned models

Conditioned models often append a few restrictive rows to a well-rounded base
polytope. Naja optionally builds a Hessian from the tight added rows, selects its
largest-eigenvalue directions, and occasionally uses those directions in the
CHR mixture. Intuitively, these are normal combinations associated with the
newly narrow directions. The vectors and their products with $A_c$ are fixed
before sampling, so they remain valid symmetric proposal directions throughout
the run.

## 9. Device-resident output, backmapping, and streams

Each block owns a separate PRNG state and chain. Naja advances a chain by the
requested thinning interval between retained samples. The stored `.npy` array is
float32 with shape `(n_samples, dimension)`; internally, sampling calculations
use double precision.

When reaction-space output is requested, Naja applies the composed affine map
$v=Ty+q$ on the GPU with a dense matrix-matrix multiply and a shift-add kernel.
This avoids downloading reduced-space samples only to perform a large CPU
backmap. For sample counts that would make the device output matrix too large,
the streamed path alternates two device buffers and two pinned host buffers on
separate compute and copy streams. Sampling of the next chunk can therefore
overlap transfer of the previous one.

Thinning reduces serial correlation in stored output but does not create
independent samples. Run multiple chains and use `naja validate` to inspect
effective sample size, split-$\hat R$, feasibility, and chord statistics.
For a new model family or GPU architecture, treat these diagnostics and the GPU
correctness tests as part of the experiment—not as an afterthought.

## 10. What is exact, and what should be measured

The chord update and slack recurrence are algebraically exact in real
arithmetic. Their device realization is floating-point, so Naja provides slack
resynchronization and feasibility diagnostics. The affine transformations are
invertible and account for their Jacobian through the transformed polytope; they
do not change the intended uniform distribution when samples are mapped back.

Mixing quality, however, is empirical. Barrier correction, pair moves, Dikin
directions, and thinning are designed to improve it, but none replaces multiple
chains and diagnostics. Use `naja validate` to inspect effective sample size,
split-$\hat R$, feasibility, and chord statistics for each model family and GPU
configuration.

## 11. Reading the code beside this note

- `src/gpu/chr.cu` — direction choice, block reduction, step draw, and slack recurrence.
- `src/engine/job_sampling.cu` — feasibility gates, affine-hull reduction, transformations, and launch setup.
- `src/rounding/barrier_rotation.cu` — GPU formation and eigendecomposition of the barrier Hessian.
- `src/rounding/dikin_directions.cu` — construction of escape directions from added constraints.
- `src/validate/` — diagnostics applied to saved chains.

For implementation-wide detail and historical design context, see the
[technical summary](technical-summary.md). For the on-disk input contract, see
the [model format reference](model-format.md).
