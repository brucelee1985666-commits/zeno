# Zeno Aether — four engineering investments

Status: integration foundation, not a claim that the four subsystems are complete.
Baseline inspected: fork branch `master`, commit
`56c74cc3ef5ca5d0cc3a9d713f6e4e7788c45979`.

## Guiding rule

Reuse the existing Zeno graph, ZPC/CUDA, OpenVDB, and OptiX code. Add optional
dependencies behind explicit CMake switches. Never download third-party code
during configure, silently change the default build, or make a new backend
mandatory for existing users.

## 1. ZenoCompute — first priority

**Existing assets:** C++17 core, ZPC, CUDA extension, OpenMP support, and TBB
already appears as a dependency in the VDB extension.

**Adopt:** oneTBB task arenas/task groups for CPU scheduling; retain ZPC/CUDA
for GPU work. Use CUDA events and explicit stream ownership before attempting
kernel fusion. Add a graph execution profiler and measure CPU↔GPU transfer
bytes, queue wait, node time, and peak VRAM.

**Do not do yet:** replace the graph executor or introduce a second GPU
abstraction. First map node evaluation, data ownership, synchronization, and
cache invalidation. A TBB link alone is not a scheduler implementation.

Build switch: `-DZENO_AETHER_ENABLE_TBB_RUNTIME=ON`.

## 2. ZenoVolume — second priority

**Existing asset:** `projects/zenvdb` embeds an OpenVDB submodule.

**Adopt:** NanoVDB as a read-mostly GPU representation of OpenVDB grids.
Keep OpenVDB as the authoring/editing and topology-changing representation;
convert or upload at well-defined simulation/render boundaries. NanoVDB's
tree topology is static, so it is not a drop-in replacement for mutable FLIP
or topology-changing volume operations.

**Next implementation:** add a conversion/upload node, device buffer lifetime
management, grid metadata validation, and sparse-volume benchmarks. Test
CUDA first; keep the data format API-agnostic for later Vulkan/HLSL readers.

Build switch: `-DZENO_AETHER_ENABLE_NANOVDB=ON` and provide a compatible
NanoVDB installation/include path. The build does not fetch dependencies.

NeuralVDB is a later optional research track, not a prerequisite for this
first GPU-volume milestone.

## 3. ZenoRender — third priority

**Existing asset:** `zenovis/xinxinoptix` and the existing
`ZENO_ENABLE_OPTIX` switch. Reuse and audit this path before adding another
ray-tracing backend.

**Adopt:** OptiX for NVIDIA ray tracing; evaluate Vulkan as a separate
cross-vendor compute/graphics backend only after the viewport's resource,
shader, and synchronization abstractions are understood. ReSTIR is a later
rendering algorithm, not a library that can simply be linked to produce GI.

Build switches: existing `-DZENO_ENABLE_OPTIX=ON`; optional Vulkan link
foundation: `-DZENO_AETHER_ENABLE_VULKAN=ON`.

Do not claim Vulkan ray tracing or ReSTIR is implemented by this dependency
wiring. Those require renderer code, shaders, denoising, validation scenes,
and GPU-specific tests.

## 4. ZenoGaussian — fourth priority

**Adopt:** treat Gaussian Splatting as an import/render representation, not a
replacement for editable meshes or simulation volumes. Prototype the data
contract first: position, covariance/scale+rotation, opacity, SH coefficients,
coordinate transform, color space, and LOD metadata.

Candidate for an isolated conversion tool: OpenSplat. Its repository is
AGPL-3.0; do not vendor or link it into the Zeno executable without a deliberate
license review. Prefer a process boundary or a separately distributed tool.
A native viewport renderer should be implemented only after format and
performance tests establish the target.

## Dependency and license policy

- Keep the current OpenVDB dependency and inspect the exact submodule revision
  before updating it.
- NanoVDB is part of the OpenVDB project; validate the version and license
  from the exact selected revision before vendoring.
- OptiX SDK use is governed by NVIDIA's SDK license; the existing bundled
  headers and third-party code need a separate audit before upgrading.
- OpenSplat is AGPL-3.0; isolate it unless the project intentionally accepts
  that license's obligations.
- No new dependency is fetched automatically by CMake.

## Acceptance gates (must be measured, not guessed)

1. Baseline and candidate built with the same compiler, CUDA, driver, and
   build flags.
2. CPU/GPU node graph: wall time, per-node time, scheduling overhead,
   transfer bytes, and peak VRAM.
3. Volume: OpenVDB→NanoVDB conversion time, upload time, memory footprint,
   random-access throughput, and correctness against CPU samples.
4. Rendering: same scene, resolution, samples, camera, and denoiser; report
   frame time and image error, not only FPS.
5. Gaussian: load time, memory, frame time, and visual parity on a documented
   dataset.
6. Existing default build must remain unchanged when all Aether switches are
   OFF.

## Immediate next coding increments

- Instrument the current graph executor before changing scheduling semantics.
- Implement a single OpenVDB→NanoVDB conversion/upload node.
- Audit the existing OptiX path and establish a reproducible reference scene.
- Add a standalone Gaussian import/conversion prototype and license boundary.
