# ZPC transfer-profiler patch

Target: ZPC submodule revision `2739e7b6a71407be26dbc1a9b66ffcf597f031f8`.

This patch adds opt-in, process-wide atomic counters for explicit CUDA Driver API copy calls in the ZPC CUDA memory backend. It distinguishes host-to-device, device-to-host, and device-to-device and generic-copy bytes and call counts. Generic `cuMemcpy` traffic is kept in its own bucket because its pointer direction should not be assumed. It does not synchronize the GPU and does not alter copy ordering.

## Apply

From the root of a checkout of the pinned ZPC revision:

```sh
git apply 0001-zpc-explicit-transfer-counters.patch
```

Compile the ZPC CUDA backend with `ZS_ENABLE_TRANSFER_PROFILER=1` to enable counting. Leave it undefined or set to 0 for the default zero-instrumentation path.

Include `zensim/cuda/memory/TransferProfiler.hpp` and call:

- `zs::transfer_profile::snapshot()` to read counters;
- `zs::transfer_profile::reset()` to reset them.

## Important limits

- Counts explicit copies only; Unified Memory page migration is not observed.
- Counts copies routed through these ZPC functions only; direct CUDA calls elsewhere bypass it.
- Generic `copy()` calls are reported separately; they are not folded into HtoD, DtoH, or DtoD without pointer-direction proof.
- Counters are process-wide, not per-device or per-node. A future refinement can add device/context attribution.
- These counters measure requested bytes, not physical PCIe traffic, latency, or bandwidth.
- Snapshot loads are individually atomic but are not a globally consistent simultaneous snapshot while copies are running.
- No compile or runtime test has been run against this patch yet; it must pass ZPC's supported compiler matrix before the submodule pointer is advanced.