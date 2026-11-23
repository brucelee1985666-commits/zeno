# Aether execution diagnostics

The optional Aether profiler now collects one report for each outer
`Graph::applyNodes()` execution. Nested graph applications on the same thread
share the outer report and do not reset its counters.

## Build switches

Enable the CPU/node and graph-execution scope:

```sh
-DZENO_AETHER_ENABLE_PROFILER=ON
```

For CUDA builds, also enable the explicit ZPC memory-copy counters:

```sh
-DZENO_WITH_CUDA=ON
-DZENO_AETHER_ENABLE_PROFILER=ON
-DZENO_AETHER_ENABLE_TRANSFER_PROFILER=ON
```

The transfer-profiler switch is consumed by `projects/CUDA`; it is not
available in a CPU-only build. All options remain OFF by default.

## Reading a report

Include `<zeno/core/NodeExecutionProfiler.h>` and call:

```cpp
const std::string json = zeno::getAetherExecutionReportJson();
```

Or consume the typed snapshot:

```cpp
const zeno::AetherExecutionReport report =
    zeno::getAetherExecutionReport();
```

The report contains:

- `nodes`: calls, failures, total/average/maximum node apply time, dependency
  edge count, and dependency-resolution time.
- `gpu_kernels`: calls and CUDA-event total/average/maximum time for the
  currently instrumented launch sites.
- `gpu_transfers`: explicit ZPC HtoD, DtoH, DtoD and generic copy calls/bytes.
- `totals` and `findings`: aggregated values and evidence-bounded diagnostic
  observations.

The JSON schema identifier is
`zeno.aether.execution-report.v1`.

## Measurement boundaries

These metrics are deliberately not presented as a single additive frame time:

- Node apply timing measures the instrumented `apply()` body.
- Dependency-resolution timing wraps `requireInput()`; it can recursively
  execute upstream nodes, so it overlaps with node execution and must not be
  added to node apply time as an independent cost.
- GPU kernel timing covers only the CUDA launch paths already instrumented by
  the Aether kernel profiler, not every kernel in Zeno/ZPC.
- Transfer bytes are requested copy sizes from successful instrumented ZPC
  calls, not measured PCIe traffic or bandwidth. Unified Memory page migration
  and direct CUDA copies outside those wrappers are not counted.
- The profiler uses process-wide aggregate counters. A single outer graph
  execution on one thread defines the reset boundary; simultaneous independent
  graph executions are not isolated into separate concurrent reports.

The findings identify the highest measured node/kernel and describe counter
coverage. They do not claim a bottleneck when a measurement domain is absent
or incomplete. The report is intended as a reliable diagnostic input for a
future scheduler, not as a scheduler or an optimization by itself.


## Editor viewer and comparisons

In multiprocess editor builds, the runner sends each completed graph-substep report to the editor. Open **Tools → Aether Execution Diagnostics** to inspect the latest JSON and save it to disk. The viewer compares node-apply time, instrumented GPU-kernel time, and explicit transfer bytes against the previous captured report. The typed API `compareAetherExecutionReports(baseline, candidate)` additionally returns per-node and per-kernel deltas for callers that retain typed snapshots.

The UI comparison is against the previous captured report. When a timeline contains multiple substeps, this may be the preceding substep rather than a different scene or a controlled benchmark run.


## Aether Smart Scheduler (experimental)

Enable the history-based scheduler with:

```sh
-DZENO_AETHER_ENABLE_SMART_SCHEDULER=ON
```

This option also enables the CPU node profiler because scheduling needs timing
history. The scheduler keeps a constant-space EWMA per observed node name for
the process lifetime (alpha 0.20). At each graph application it estimates the
critical path of each requested root as that node's historical mean apply time
plus the longest known dependency path, then stably orders requested roots from
highest estimated path to lowest. Estimates are confidence-weighted (confidence ramps to 1.0 over eight
samples). Unknown nodes receive a zero own-node estimate; stable sorting keeps
ties deterministic. The existing recursive dependency
execution and visited-node reuse remain unchanged. The selected order and
estimates are included in the execution report's `schedule` array, and
`scheduler_mode` identifies the active policy.

This is deliberately a conservative first scheduler stage. It reorders
independent requested roots; it does not execute nodes concurrently, move
arbitrary node implementations between CPU and CUDA, fuse kernels, or
automatically change data residency. Zeno's current `INode::apply()` contract
does not expose a general dual-implementation/backend-capability interface,
and aggregate CUDA/transfer counters are not yet attributable to individual
nodes. Automatic CPU/GPU dispatch or transfer elimination without those
contracts would be guesswork and is intentionally not enabled.

The next safe extension requires explicit per-node backend capabilities,
node-scoped GPU/transfer telemetry, and a data-residency/lifetime contract.
Only then can the cost model compare CPU execution against CUDA execution plus
the actual transfer cost and make a defensible dispatch decision.
