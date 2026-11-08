# Sampling-phase benchmarks

Sampling-phase time at observed passing budgets and median reaction pooled bulk ESS per sampling-phase second. Fastest complete configuration per method/model selected by phase median.

hopsy.sample phase includes pool startup, task/result serialization, IPC and returned-array assembly. Inclusive sampler proxy, not pure arithmetic/worker compute.

profile.gpu_sampling_s includes sampler allocation, initial slack GEMM/RNG init and CHRR sampling. Excludes output/download/normalization, but not isolated CHRR kernel-only time.

Actual chains*(retained_draws+retained_warmup)*thinning, including warmup. ESS per total transition separates observed mixing efficiency from hardware timing. Warmup counts fixed in retained draws, so warmup moves vary by thinning.

Verified common original-reaction mask; median classical split Rhat<=1.01 and pooled rank-normalized bulk ESS median>=100; zero original-bound violations at1e-7. Mask geometry must match. Quality/search/diagnostic and output costs are excluded from phase timing.

| Model | CPU phase(s) | GPU phase(s) | CPU/GPU phase | GPU/CPU ESS/s | CPU/GPU total transitions | GPU/CPU ESS/transition |
|---|---:|---:|---:|---:|---:|---:|
| e_coli_core | 0.149214 | 0.1508 | 0.989 | 1.202 | 1.600 | 1.865 |
| iAB_RBC_283 | 3.98506 | 0.606 | 6.576 | 14.930 | 11.738 | 26.648 |
| iAF1260 | 18.6619 | 15.4979 | 1.204 | 2.264 | 3.667 | 6.862 |
| iJO1366 | 20.5832 | 7.8132 | 2.634 | 2.403 | 7.333 | 6.829 |
| iJR904 | 6.99669 | 2.7324 | 2.561 | 2.696 | 12.816 | 13.561 |
| iML1515 | 78.8702 | 8.1171 | 9.717 | 7.556 | 33.602 | 26.305 |

Configuration and CPU allocation details:

| Model | Method | Selected configuration | Draws/chain by seed | Workers by seed | Verified physical cores by seed | Reserved logical CPUs by seed |
|---|---|---|---|---|---|---|
| e_coli_core | CPU | quality:cpu_nativejade1 | 101:250, 202:250, 303:250 | 101:1, 202:1, 303:1 | 101:unrecorded, 202:unrecorded, 303:unrecorded | 101:8, 202:8, 303:8 |
| e_coli_core | GPU | quality:naja128 | 101:500, 202:500, 303:500 | 101:unrecorded, 202:unrecorded, 303:unrecorded | 101:unrecorded, 202:unrecorded, 303:unrecorded | 101:unrecorded, 202:unrecorded, 303:unrecorded |
| iAB_RBC_283 | CPU | native_cpu:physical16_native_jade | 101:250, 202:250, 303:250 | 101:16, 202:16, 303:16 | 101:16, 202:16, 303:16 | 101:32, 202:32, 303:32 |
| iAB_RBC_283 | GPU | quality:naja128 | 101:5000, 202:5000, 303:5000 | 101:unrecorded, 202:unrecorded, 303:unrecorded | 101:unrecorded, 202:unrecorded, 303:unrecorded | 101:unrecorded, 202:unrecorded, 303:unrecorded |
| iAF1260 | CPU | larger_quality:cpu2000 | 101:10000, 202:10000, 303:10000 | 101:16, 202:16, 303:16 | 101:16, 202:16, 303:16 | 101:32, 202:32, 303:32 |
| iAF1260 | GPU | larger_quality:gpu | 101:5000, 202:5000, 303:5000 | 101:unrecorded, 202:unrecorded, 303:unrecorded | 101:unrecorded, 202:unrecorded, 303:unrecorded | 101:unrecorded, 202:unrecorded, 303:unrecorded |
| iJO1366 | CPU | native_cpu:physical16_thin2000 | 101:10000, 202:10000, 303:10000 | 101:16, 202:16, 303:16 | 101:16, 202:16, 303:16 | 101:32, 202:32, 303:32 |
| iJO1366 | GPU | gpu_thinning:thin1000 | 101:2000, 202:2000, 303:2000 | 101:unrecorded, 202:unrecorded, 303:unrecorded | 101:unrecorded, 202:unrecorded, 303:unrecorded | 101:unrecorded, 202:unrecorded, 303:unrecorded |
| iJR904 | CPU | native_cpu:physical16_native_jade | 101:500, 202:500, 303:500 | 101:16, 202:16, 303:16 | 101:16, 202:16, 303:16 | 101:32, 202:32, 303:32 |
| iJR904 | GPU | precision_quality:verified_f64_export | 101:20000, 202:20000, 303:20000 | 101:unrecorded, 202:unrecorded, 303:unrecorded | 101:unrecorded, 202:unrecorded, 303:unrecorded | 101:unrecorded, 202:unrecorded, 303:unrecorded |
| iML1515 | CPU | larger_quality:cpujade | 101:500, 202:500, 303:500 | 101:16, 202:16, 303:16 | 101:16, 202:16, 303:16 | 101:32, 202:32, 303:32 |
| iML1515 | GPU | precision_quality:verified_f64_export | 101:2000, 202:2000, 303:2000 | 101:unrecorded, 202:unrecorded, 303:unrecorded | 101:unrecorded, 202:unrecorded, 303:unrecorded | 101:unrecorded, 202:unrecorded, 303:unrecorded |

Ratios are empirical for the finite tested configurations; configuration selection uses sampling phase rather than whole process time. Secondary process/output timings and per-seed mask, bounds, ESS, transition and source records remain in SAMPLING_BENCHMARKS.json. Missing allocation evidence is labeled unrecorded.512-chain best-use-case experiments and worker-timed variants are separate.

Observed numerical exclusions from the strict comparison:

- iJO1366 / quality:naja_tune_pair025 / seed303: bounds_gate_failed: samplefraction=1.25e-06, maxabsolute=2.6665152930149816e-07, tolerance=1e-07, run=/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/quality/iJO1366/naja_tune_pair025/seed303/final-budget50000
- iJR904 / quality:naja128 / seed101: bounds_gate_failed: samplefraction=3.125e-06, maxabsolute=0.0035504193510860205, tolerance=1e-07, run=/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/quality/iJR904/naja128/seed101/final-budget20000
- iML1515 / larger_quality:gpu / seed303: bounds_gate_failed: samplefraction=3.125e-05, maxabsolute=3.0012861316208728e-06, tolerance=1e-07, run=/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/larger_quality/iML1515/gpu/seed303/final
- iMM904 / larger_quality:cpu2000 / seed101: bounds_gate_failed: samplefraction=0.0294625, maxabsolute=2.0060176644332893e-07, tolerance=1e-07, run=/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/larger_quality/iMM904/cpu2000/seed101/final
- iMM904 / larger_quality:cpu2000 / seed202: bounds_gate_failed: samplefraction=0.0295875, maxabsolute=1.9329087017160132e-07, tolerance=1e-07, run=/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/larger_quality/iMM904/cpu2000/seed202/final
- iMM904 / larger_quality:cpu2000 / seed303: bounds_gate_failed: samplefraction=0.0289625, maxabsolute=1.9950769012596553e-07, tolerance=1e-07, run=/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/larger_quality/iMM904/cpu2000/seed303/final
- iMM904 / larger_quality:cpujade / seed101: bounds_gate_failed: samplefraction=0.030125, maxabsolute=1.8169061561563338e-07, tolerance=1e-07, run=/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/larger_quality/iMM904/cpujade/seed101/final
- iMM904 / larger_quality:cpujade / seed202: bounds_gate_failed: samplefraction=0.02675, maxabsolute=1.8325737844442565e-07, tolerance=1e-07, run=/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/larger_quality/iMM904/cpujade/seed202/final
- iMM904 / larger_quality:cpujade / seed303: bounds_gate_failed: samplefraction=0.028875, maxabsolute=1.6421943864549763e-07, tolerance=1e-07, run=/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/larger_quality/iMM904/cpujade/seed303/final
- iMM904 / larger_quality:gpu / seed101: bounds_gate_failed: samplefraction=0.0281875, maxabsolute=1.9680662670438904e-07, tolerance=1e-07, run=/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/larger_quality/iMM904/gpu/seed101/final
- iMM904 / larger_quality:gpu / seed202: bounds_gate_failed: samplefraction=0.0275625, maxabsolute=1.8554893401420804e-07, tolerance=1e-07, run=/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/larger_quality/iMM904/gpu/seed202/final
- iMM904 / larger_quality:gpu / seed303: bounds_gate_failed: samplefraction=0.026625, maxabsolute=2.0494548938426178e-07, tolerance=1e-07, run=/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/results/larger_quality/iMM904/gpu/seed303/final

Fixed-budget controls, separate from passing timing comparisons:

| Model | CPU configuration | Retained cap/chain | Actually passed/tested | Not yet run |
|---|---|---:|---:|---|
| iML1515 | native thin2000 | 10000 | 0/3 | none |

A fixed-cap failure is not a claim about eventual convergence at larger budgets. Failed controls have no matched-quality speedup or ESS-rate headline.
