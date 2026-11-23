# Aether CPU/GPU execution backends: Taskflow and StarPU

Status: Taskflow adapter prototype added; StarPU evaluated but not linked. No C++ build, runtime test, or performance benchmark has been executed for this change.

## Findings from the current Zeno branch

- The root build selects C++17 (CMAKE_CXX_STANDARD 17).
- INode::requireInput resolves upstream nodes recursively through Graph::applyNode.
- Graph::applyNode mutates the shared Context::visited set and invokes node code that may mutate shared graph/session state.
- INode has no purity, thread-safety, side-effect, resource-access, or backend-capability contract.
- ZPC/CUDA already owns important device-memory and kernel paths. A second GPU scheduler must not silently take ownership of these resources.

These facts prevent a safe direct mapping of arbitrary Zeno nodes into concurrent tasks today.

## Taskflow: selected first integration

Repository: https://github.com/taskflow/taskflow
Compatibility target: v3.11.0 (MIT, C++17). Taskflow v4.x requires C++20; adopting it would require a separately reviewed Zeno-wide language-standard change.

The new AetherTaskflowExecutor is an opt-in standalone CPU task-graph adapter:
- Accepts explicit task callbacks and predecessor/successor edges.
- Validates edge indices, duplicate edges, self-dependencies, and cycles before dispatch.
- Uses Taskflow's executor for CPU parallel dispatch/work-stealing.
- Waits for completion and propagates callback exceptions.
- Keeps Taskflow types out of the public Zeno header through PIMPL.
- Does not dispatch arbitrary INode objects or change Graph::applyNodes order.

### Build

Taskflow is not downloaded by CMake. Install/check out Taskflow v3.11.x separately and set TASKFLOW_ROOT to its source root or ZENO_AETHER_TASKFLOW_INCLUDE_DIR directly.

    -DZENO_AETHER_ENABLE_TASKFLOW=ON
    -DTASKFLOW_ROOT=/path/to/taskflow-3.11

The option defaults OFF. Existing builds remain unchanged when disabled.

Optional smoke test:
    -DZENO_AETHER_ENABLE_TASKFLOW=ON
    -DZENO_AETHER_BUILD_TASKFLOW_TESTS=ON

## StarPU: complementary future backend, not enabled in this increment

Repository: https://github.com/starpu-runtime/starpu (GitHub mirror; main development is hosted on Inria GitLab). License: LGPL-2.1-or-later.

StarPU provides CPU/GPU task implementations, heterogeneous scheduling, data handles, replication/transfers, and asynchronous execution. It is most appropriate when each task has explicit CPU and CUDA implementations (codelets), declared read/write/reduction access modes, stable data handles/lifetimes, and measurable performance models per device.

Zeno does not currently expose these contracts for general nodes. Its zany/IObject inputs and recursive dependency resolution do not automatically provide StarPU data handles or safe access modes. Adding StarPU now risks duplicating ZPC memory ownership and does not make current nodes safely GPU-dispatchable.

Decision: do not link StarPU yet. First define an Aether node capability/resource descriptor and bridge selected ZPC buffers to StarPU data handles in a standalone experiment. Audit LGPL distribution obligations before shipping a linked runtime.

## Planned evolution

1. Benchmark the Taskflow adapter on explicit, side-effect-free task DAGs.
2. Add a node capability contract: thread-safe, pure/reorder-safe, resource reads/writes, CPU implementation, optional CUDA implementation.
3. Add a graph snapshot/immutable execution plan; do not share the current mutable Graph::ctx->visited across worker threads.
4. Keep Taskflow as CPU backend; retain ZPC/CUDA for device execution.
5. Prototype StarPU only for nodes with explicit CPU/CUDA codelets and registered data handles.
6. Compare baseline, Taskflow, and StarPU on identical deterministic graphs: wall time, scheduling overhead, node time, transfer bytes/latency, and peak VRAM.

## Acceptance and limitations

- This prototype is not yet the Zeno graph executor.
- No FPS, throughput, or latency improvement is claimed.
- No automatic network fetch occurs during CMake configure.
- Taskflow and StarPU are not enabled by default.
- The smoke test is authored but has not been compiled or run in this environment.