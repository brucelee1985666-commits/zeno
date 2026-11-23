# Aether scope reduction

Date: 2026-11-23
Branch: `aether/lean-core-cleanup`

## Removed

- `AetherTaskflowExecutor.h/.cpp`: standalone callback executor; it did not execute Zeno `INode` objects and had no benchmark evidence.
- `AetherTaskflowExecutorSmoke.cpp`: test for the removed standalone adapter, not for the real Zeno graph executor.
- Taskflow CMake options, package discovery, link wiring, and smoke-test target.
- Taskflow/StarPU integration note and the unexecuted Taskflow benchmark plan.
- `ZENO_AETHER_ENABLE_TBB_RUNTIME`: link-only integration switch; no Aether scheduler was implemented through it.
- `ZENO_AETHER_ENABLE_VULKAN`: link-only viewport switch; no Vulkan renderer/backend existed behind it.

## Kept

- Existing Zeno graph executor and default execution order.
- Opt-in per-node profiler and execution diagnostics.
- ZPC/CUDA and the existing GPU execution/memory ownership paths.
- Existing OpenVDB authoring/editing path.
- Existing OptiX renderer path.
- Optional NanoVDB header discovery switch, explicitly not represented as a completed volume node.
- ZPC transfer-profiler patch and documentation as an unvalidated, opt-in engineering artifact; it does not change the ZPC submodule pointer in this cleanup.

## Deliberately not deleted

- OpenMP, TBB already used by existing project components, ZPC, CUDA, OptiX, OpenVDB, or other existing Zeno subsystems. They are upstream/runtime dependencies or established capabilities, not disposable Aether experiments.
- The advisory historical cost model inside the profiler. It remains diagnostic only and does not reorder or concurrently execute graph nodes.

## Validation status

GitHub file operations confirm the branch edits and deletions. No local CMake configure, C++ compilation, unit test, CUDA test, runtime graph replay, or performance benchmark was executed in this environment. The cleanup is therefore source-level and repository-structure validated only; it is not a build certification.
