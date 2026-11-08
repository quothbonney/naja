# Hardware and software paragraph

Benchmarks ran through Slurm on one node with eight NVIDIA H100 GPUs (80 GB HBM3 each)
and two Intel Xeon Platinum 8462Y+ processors (64 physical cores,128 logical CPUs,
approximately 2 TiB system memory). Each reported single-GPU run uses one H100;
the full-node CPU comparison uses 64 native hopsy workers with128 logical CPUs
reserved and one worker pinned per physical core. The NVIDIA driver is 580.126.20
and the CUDA compiler is 12.9.86; the13.0 shown by nvidia-smi is driver compatibility,
not the compiler toolkit version. The base Naja checkout is git
23873f01d3268a79bbf84499a66adce5e88cc6cb. Each executable used for experimental
variants is separately identified by SHA256 and an archived source/patch;
the git revision alone does not identify an experimental build.

The local CPU environment uses Python 3.12.15,hopsy 1.7.0,PolyRound 0.5.0,
COBRApy 0.32.1,NumPy 2.5.4,SciPy 1.18.1,ArviZ 1.3.0,HiGHS/highspy 1.15.1
and swiglpk 5.0.13. CPU hopsy also has a separate verified native build
(-O3 -march=native -mtune=native); worker timing instrumentation is validated
by complete trajectory parity. MATLAB is unavailable; the CPU baseline is
hopsy CHRR with PolyRound, not MATLAB COBRA Toolbox CHRR.

Raw hardware evidence: ../setup/h100-environment.txt. GPU correctness tests
(cube moments, simplex moments, feasibility, backmap and thin polytope) all
passed 5/5 on H100; output is ../setup/gpu-correctness.log. Source patches,
binary fingerprints and per-run manifests remain alongside the results.
All environments and generated artifacts are workspace-local.

# Reporting scopes

The primary CPU compute timer is the maximum accumulated C++MarkovChain.draw
interval across workers, including Python-binding/state-copy cost and any
scheduling delays, while excluding pool startup, IPC and output assembly.
The GPU timer is the native sampler phase, including allocation, initial
slack construction and RNG initialization, excluding backmap/download/output.
These scopes are explicitly reported proxies rather than identical arithmetic
counters. Isolated kernel timings belong to the separate Nsight profile.

Original-model LP reaction masks and conservative exact-cached-target LP masks
are separate diagnostic assays. Neither sampler samples through a reaction
mask: masks only choose reactions aggregated by convergence diagnostics.
All original reaction bounds are checked without the diagnostic mask.
Frozen metrics and samples are preserved; mask sensitivity is reaggregation
of all-reaction full-budget statistics and has independent fresh-run validation.

512-chain fixed-budget throughput and 16-chain latency studies are separate.
The H100 is poorly occupied by 16 chains: Nsight measured 11.7% mean SM active,
versus 97.2% at 512 chains in the throughput profile. SM active is not arithmetic
throughput. Large-chain comparisons are not globally minimal time-to-quality
across all chain counts. Preparation reuse against full rerounding is reported
separately from comparison against the fastest tested passing CPU pipeline.
