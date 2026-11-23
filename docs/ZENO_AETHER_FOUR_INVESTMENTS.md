

## ZenoCompute increment 5 — CPU↔GPU transfer instrumentation audit

Source audit completed against the exact ZPC submodule revision pinned by this branch:
- ZPC repository: `zenustech/zpc`
- Pinned submodule commit: `2739e7b6a71407be26dbc1a9b66ffcf597f031f8`
- Zeno submodule path: `projects/CUDA/zpc`

### Central copy paths identified

ZPC exposes centralized memory operations in `include/zensim/cuda/memory/MemOps.hpp` and implements them in `include/zensim/cuda/memory/MemOps.cpp`. The CUDA backend has explicit operations for:
- host-to-device: `copyHtoD(device_mem_tag, ...)` and `copyHtoD(um_mem_tag, ...)`;
- device-to-host: `copyDtoH(device_mem_tag, ...)` and `copyDtoH(um_mem_tag, ...)`;
- device-to-device: `copyDtoD(...)`;
- generic device copy: `copy(...)`;
- managed-memory allocation and copies through `um_mem_tag`.

The current implementations use CUDA Driver API calls such as `cuMemcpyHtoD`, `cuMemcpyDtoH`, and `cuMemcpyDtoD`. The operation receives an exact byte count and a source-location argument, making these functions the appropriate place for low-overhead byte counters. ZPC's `Vector` and `TileVector` use the `Resource::copy` abstraction for many data movement operations, so instrumenting the backend is preferable to adding counters to individual Zeno nodes.

### Coverage limits that must remain explicit

- Managed-memory page migration caused by CPU/GPU access is not equivalent to an explicit copy call and will not be counted by these counters.
- Direct CUDA Driver/Runtime API calls outside ZPC's memory-resource functions bypass this accounting.
- Device-to-device bytes must be reported separately from PCIe/host-device traffic; they are not CPU↔GPU transfers.
- Allocation size is not peak resident VRAM. A separate allocation/lifetime tracker is required for that metric.
- Byte counters can be updated at the copy call without adding a device synchronization. Transfer latency must be measured separately and must not be inferred from bytes.

### Implementation boundary / next safe change

The ZPC implementation is a pinned Git submodule, not source owned by this repository's current tree. Editing its files as if they were ordinary Zeno files would not create a durable parent-repository change. The correct implementation is a compatible ZPC fork or upstreamable patch that adds opt-in, thread-safe counters for HtoD, DtoH, DtoD, and generic device copies, preserving the existing API and default behavior; then update the Zeno submodule pointer to that reviewed revision. Do not silently change the submodule URL or claim complete transfer coverage before this boundary is resolved.

**Status:** source paths mapped; transfer counters not yet implemented; no build or runtime measurement performed. This audit is deliberately a truthful prerequisite, not a fabricated profiler result.
