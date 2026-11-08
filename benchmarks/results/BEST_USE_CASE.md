# Where this prototype supports using Naja

Primary interpretation should separate output-excluded sampling work from
conditioning setup/correction; output handling and complete-process wall are
secondary operational measures. CPU worker-compute instrumentation and GPU
kernel profiles are being validated before an unqualified mathematical-speedup
claim. Existing sampler-phase proxies have explicit allocation/IPC limits.

The latest tuned iJO1366 **secondary operational** comparison uses explicit thinning
selection: Naja TPB512, thinning 1,000, 16 chains and 2,000 retained draws/chain
measures 9.46715 s median including normalization. Native hopsy on 16 physical
cores (32 logical CPUs reserved), thinning 2,000 and 10,000 retained draws/chain
measures 25.8015 s. Both pass the declared median Rhat ≤ 1.01 and ESS ≥ 100 criterion;
all three Naja ESS medians are around 2,000 and all bound fractions are zero at
1e-7. Ratio of medians is 2.725×; median same-seed CPU/GPU ratio is 2.747×.
This is an empirical selected-budget comparison, with different sample counts
and thinning, not equal-count throughput or proof of global optimality. Search,
pilot and diagnostic costs are excluded and preserved separately. The three
seeds were used for tuning, so fresh nonoverlapping seeds are needed for the
separate 512-chain quality study now underway.
[Actual runs and selection costs](gpu_thinning/iJO1366/summary.json).

The main CHR CUDA kernel assigns one chain to each block. Sixteen chains expose
only 16 blocks, limiting how much of an H100 can work concurrently; increasing
threads per block does not create more chain blocks. This explains why a small
16-chain benchmark need not show the device's best throughput. It is an
architectural limit on available block parallelism, not a measured occupancy
percentage or a guarantee that 512 chains improve convergence. CPU comparisons
currently use 16 physical sampling cores, not all 128 logical CPUs on the node.

The separate wide-index 512-chain iJO1366 probe at TPB128 measured 1,116,880
retained samples per GPU sampling second at automatic thinning 96. That figure
uses 512×10,000 retained samples and includes warmup time in the sampling-phase
denominator. Its whole-run inclusive throughput was 105,814 retained samples/s
and wall 48.3869 s, including the adapter. It used a separate verified wide-index
binary because the frozen binary overflowed; original failed runs remain invalid.
Only 1,024 deterministic points were feasibility-checked and no full-output
quality comparison was established by that probe. Kernel throughput therefore
cannot be converted into a CPU end-to-end speedup. TPB512 was slower than 128
at 512 chains, so small-chain tuning does not transfer automatically.
[Scaling evidence](scaling-extension-safe/iJO1366/summary.json).

Naja is a plausible fit when many concurrent chains or large repeated sampling
workloads amortize startup, preprocessing and output costs. Small core-model
workloads are dominated by short-process overhead, and larger models can need
substantial convergence budgets or expose numerical precision limitations.
Shared-base-rounding reuse across conditioned models may be a distinct benefit,
but the present evidence does not yet establish its equal-quality end-to-end
advantage. A fair reuse comparison must keep the conditional target exact,
include correction/re-rounding costs and verify original reaction-space bounds.
