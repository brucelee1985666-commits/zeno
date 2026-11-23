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
