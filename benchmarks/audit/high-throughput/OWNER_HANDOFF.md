# CPU throughput owner handoff

Latest audit, **November8 19:31UTC**: the full64-CPU and512-chain GPU groups pass
all3seeds under both historical878 and conservative exact-cached478 assays.
Lead observed CPU/GPU compute ratio is **11.896x**; median paired GPU/CPUESS/s
is **14.473x** under478 (**11.463x** under878). This is fixed512-chain throughput,
not globally minimized latency. All6 retained-file SHA256 values were independently
rechecked. A fresh512-chain478 rerun matches all2,583 per-reaction rows exactly.
See **results/high_throughput/iJO1366/AUDITED_COMPARISON.md** and comparison.json/.csv.
The remainder below preserves earlier snapshots and the audit trail.


Snapshot: 2026-11-08T09:04:21.190943+00:00. Live state JSON files are authoritative after this snapshot.

Interactive stop05:22:37UTC; autonomous batch deadline11:00UTC. Root cleanup557 runs11:05UTC.

The full64 sequencer and independent deadline guard are detached processes. The guard restarts only an unexpectedly absent active sequencer. It never resurrects completed main/coordinate/prefault dispatchers.

Full sampling chain451→552→553 uses128logical/64physical cores,512chains,12kretained,1000warmup,thin2000.451 awaits originalRecon3D150_6 completion/timeout (cap08:23:25UTC). Firstwhole-array identity proves shared per-chain streams before afterok follow-ons run; if identity fails, actual first-case diagnostics gate rebuilt follow-ons.

Diagnostics554/555/556 run only after the finalsample,8CPU128GiB45min each. Original453 was cancelled/replaced. Mask202_8 and any new workspace Recon3D validators/campaign jobs remain held until sample capture completes; their original dependencies are recorded in full64-held-jobs.json. Deadline cleanup cancels owned unfinished cases and held pending jobs; no holds are left deliberately.

| Controller | PID at snapshot | Alive | State / PID file |
|---|---:|---|---|
| full64_sequence | 357882 | True | /mnt/home/jordan+123/history-native-20261015-r5/benchmarks/audit/high-throughput/full64-state.json |
| deadline_guard | 357889 | True | /mnt/home/jordan+123/history-native-20261015-r5/benchmarks/audit/high-throughput/deadline-guard-status.json |
| Recon3D_completion | 342818 | True | /mnt/home/jordan+123/history-native-20261015-r5/benchmarks/audit/recon3d-completion/watcher-state.json |

Restart only if the relevant controller is unexpectedly dead while its state is active, from the workspace root. Exclusive locks reject duplicates. The deadline guard normally performs this recovery itself.

full64_sequence:
```bash
setsid /usr/bin/python3 -u '/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/full64_sequence_controller.py' >> '/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/audit/high-throughput/full64_sequence_controller.restart.log' 2>&1 < /dev/null &
```

deadline_guard:
```bash
setsid /usr/bin/python3 -u '/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/full64_deadline_guard.py' >> '/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/audit/high-throughput/full64_deadline_guard.restart.log' 2>&1 < /dev/null &
```

Recon3D_completion:
```bash
setsid /usr/bin/python3 -u '/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/watch_recon3d_completion.py' >> '/mnt/home/jordan+123/history-native-20261015-r5/benchmarks/audit/high-throughput/watch_recon3d_completion.restart.log' 2>&1 < /dev/null &
```

Scientific interpretation:
- CPU48 is a one-seed preliminary, not the full64 comparison. Full64 rows stay pending until observed.
- Fresh16 fixed CPU passes2/3; fixed GPU passes1/3. Separate extensions cannot rescue a universal legacy2.73x claim.
- Worker draw timing excludes IPC but includes binding/state-copy; GPU sampler phase includes allocation/slackGEMM/RNG initialization. Neither is labeled pure arithmetic-only.
- iML guideline3/3passes, verifiedF64GPU3/3passes; CPUthin2000 fixed10k0/3passes (cap only, not eventual convergence).
- Coordinate5123/3valid, actualphase median13.0012s. Initialsinglepaired1.29x specialization gain is exploratory and did not replicate in the controlled512audit.

Refresh central report: python3 /mnt/home/jordan+123/history-native-20261015-r5/benchmarks/scripts/new_results_report.py
Status entry point: /mnt/home/jordan+123/history-native-20261015-r5/benchmarks/RESUME.md


## Reopened audit, November8 19:12UTC (three additional hours)

All full64CPU sample jobs451/552/553 and diagnostic jobs554/555/556 completed.
All three legacy878-mask groups pass, with zero unmasked1e-7 bound violations.
64physical/128reserved logical topology and512 worker-log timings are verified.
Legacy figures:CPU worker median154.658s/GPU sampler median13.0012s=11.896x;
median paired ESS/s advantage11.463x. These are not pure same-transition hardware
speedups or globally minimal latency across chain counts.

Audit exposed a certificate-scope limitation: historical878 exclusions were
certified on the original raw model, whereas exact cachedA/b currently has478
verified exclusions. Both methods used identical878 masks and cachedgeometry,
but strict cached-target conclusions require the new conservative reassay.
Six saved-array diagnostic jobs853–858 are running with8CPUs128GB each,
45-minute caps and22:00UTC deadline. Original metrics and arrays are unchanged.
The cloned assay model uses identical geometry/gem symlinks and the frozen478
cached-target mask; all unresolved reactions stay active. Only full-budget
endpoints are recomputed using unchanged metrics.py. Commands and sources:

- scripts/submit_full64_cached_target_audit.py
- scripts/full64_cached_target_metrics.sbatch
- results/high_throughput/iJO1366/cached_target_audit/jobs.json
- scripts/audit_full64_comparison.py

The latest standalone report is results/high_throughput/iJO1366/AUDITED_COMPARISON.md;
source comparison.json/.csv show pending assay records and provenance. Regenerate:
`benchmarks/env-cpu/bin/python benchmarks/scripts/audit_full64_comparison.py`.
Do not publish the legacy11.46x as strict cached-target quality until that JSON
status is complete and the conservative six-run pass gate is checked.


19:30UTC audit optimization: frozen metrics.py already computes all non-exact
reaction diagnostics at the full draw budget, including LP-masked reactions.
The new scripts/reaggregate_cached_metrics.py therefore remasks saved full-budget
reaction_metrics.csv exactly, with no new FFT/GEMM. Its independent freshGPU2k
andCPU12k validations match all per-reaction statistics and primary summaries.
Six reaggregated478 cases are saved under cached_target_aggregate/; all pass.
Keep job856 as a full512-chain exact-dataset cross-check. Other fresh reruns
853–855/857–858 were cancelled as redundant after root coordination; partial
logs/artifacts remain, with reason in cached_target_audit/redundant_diagnostic_cancellations.json.
Do not interpret those cancellations as sampler or scientific failures.
The standalone comparison remains pending until856's fresh data match.

New held-out conservative16-chain validation array883 (tasks0–2%1) uses entirely
new seeds444001/555001/666001, fixedCPU15kthin2000 andGPU3kthin1000TPB512,
sharedconservative478 assay, and no extensions. Both methods execute sequentially
inside each32-logical/16-physical CPU+oneGPU reservation. Frozen protocol and
input hashes are under results/fresh16_conservative/iJO1366/. Summary entry:
scripts/summarize_fresh16_conservative.py. Original10k/2k failures remain separate.


19:39UTC: fresh conservative16-chain validation is COMPLETE, all6cases pass.
CPU3/3 andGPU3/3, newseeds444001/555001/666001, fixedCPU15k/GPU3k.
CPUworker median21.0998s;GPU sampler median10.3878s; median paired time ratio
2.031x andESS/s ratio2.427x. Old10k/2k failures remain separate. See
results/fresh16_conservative/iJO1366/VALIDATION.md and summary.json/table.csv.
GPU preflight initially refused my nonempty output directory before sampling;
this was fixed by moving command/log files outside the GPU output folder.
Original source/failures and mechanical retries883/895 are preserved under
source_snapshots/,preflight_repair.json and per-seed archived GPU folders.
CPU completed records were reused, not resampled. No scientific retuning.

Optional SMT128 pilot is PREPARED, NOT SUBMITTED. Sources:
scripts/high_throughput_smt128.sbatch and high_throughput_smt128.py.
It reserves all128logical CPUs/64physical cores and768GB,15min cap,22:00 deadline,
same512chains/12k/thin2000. Whole retained-file SHA must equal the corresponding
64-worker source before conservative478 diagnostics are reused. It records
requested128/actual worker count and new worker critical-path time separately.
Do not queue this full-node pilot until main CPU/GPU waves drain; it would block
the current six-GPU conditioned campaign. Root will coordinate release around
20:30–21:15 if time permits. First seed101001 is a same-stream retime, not a
new independent RNG replicate. If faster, test202001/303001 before a3seedclaim;
if slower, preserve/report the one-seed pilot rather than suppressing it.
