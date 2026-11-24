# Optional tiny-cuda-nn neural volume decoder

CUDA inference is opt-in. Set `ZENO_WITH_TINY_CUDA_NN=ON` only when the parent
build has already provided the upstream `tiny-cuda-nn` CMake target and enabled
CUDA. The normal Zeno build and CPU `.znvm` reader remain independent.

## Runtime and lazy leaf decoding

`TinyCudaNNRuntime` recreates the HashGrid + CutlassMLP from the checkpoint's
model JSON, checks the parameter count and parameter precision, uploads weights
once, and converts float32 checkpoint weights to FP16 on the GPU when required.
It uses the exact batch granularity reported by the pinned tiny-cuda-nn API,
reuses CUDA scratch buffers between calls, and keeps the input/output layout
column-major as required by `tcnn::cpp::Module::inference`.

`NeuralVolumeDecoder` takes ownership of a complete `NeuralVolumeModel`.
Pass the parsed model with `std::move(model)` to avoid duplicating its CPU
checkpoint arrays. It indexes domain keys and leaf origins, but does not load
every domain's network onto the GPU. Only the requested leaf is decoded. A
two-entry least-recently-used domain cache keeps the classifier/regressor
weights for the current and most recently used domains resident; accessing a
third domain evicts the least-recently used pair. This reduces repeated model
construction and parameter uploads when traversal alternates across a domain
boundary, while bounding neural-weight VRAM to at most two domains per
decoder.

For each requested leaf the decoder:
1. Resolves the owning domain using floor division (including negative index
   coordinates).
2. Generates the same domain-local normalized xyz coordinates used by the
   Python trainer.
3. Runs the occupancy classifier for the 512 leaf voxels.
4. Applies exact occupancy flips before selecting the CPU regressor samples.
   This ordering is essential: a false-negative corrected to active must still
   receive a neural value prediction. The device path evaluates all 512
   positions, then restores values using the stored center and scale.
5. Applies exact-value corrections last, so exact values are never overwritten
   by the regressor.
6. The CPU convenience API returns one `DecodedNeuralLeaf` with 512 occupancy flags and values. For a device-resident result, use `decodeLeafDevice()` below.

A domain without a regressor must provide exact values for every active voxel;
the decoder checks this rather than silently returning background values.

## Scope and validation boundary

The legacy `decodeLeaf()` convenience API returns one leaf on the CPU. The
new `decodeLeafDevice()` path keeps the decoded leaf on the GPU and avoids
reconstructing the full volume as a CPU OpenVDB grid. Neither path is yet a
direct NanoVDB GPU-grid writer, a ZenoCompute device object, or a renderer node.
The caller can inspect the original source metadata, affine transform, and
preserved active tiles through `sourceModel()`. A subsequent integration must
stream these device leaves into a GPU-resident NanoVDB/volume representation
without expanding the full tree on the host. A decoder instance is not thread-safe because its two-entry domain cache and
tiny-cuda-nn modules are mutable; serialize calls per instance or use one
instance per worker.

The checkpoint format currently stores one shared model configuration for
classifier and regressor networks; this is valid because the trainer creates
both with the same HashGrid and CutlassMLP architecture. Partial checkpoints
are intentionally rejected by the complete-volume decoder.

Each domain is an independent local neural field. To prevent value seams at
domain boundaries, the trainer stores active values exactly for a one-leaf
(8-voxel) shell on all six faces of every domain. Interior values still use
the configured error tolerance and sparse exact corrections. This is a
conservative continuity fallback, not neural blending: it trades some
compression at boundaries for deterministic matching values across adjacent
domains. Occupancy remains losslessly corrected throughout each leaf.

The CUDA bridge has a separate compile-only CI job using the pinned
CUDA/tiny-cuda-nn/NanoVDB headers. That job can establish source-level CUDA
compilation but cannot establish GPU execution, numerical correctness,
throughput, or VRAM usage. Actual GPU inference and round-trip tests remain
unverified until run on NVIDIA hardware.


## Device-resident decoded leaves

The optional CUDA API also exposes `NeuralVolumeDecoder::decodeLeafDevice()`. It returns a move-only `DecodedNeuralLeafDevice` owner whose opaque device pointers address 512 occupancy bytes and 512 float32 values. The owner keeps both allocations and the producing stream alive. Consumers must retain the owner while using either pointer. `streamHandle()` exposes the opaque CUDA stream so downstream kernels can be enqueued in order without a host round trip; call `wait()` when the CPU needs completion and validation. `wait()` checks a compact device validation flag (one integer), not the 512-voxel arrays.

Classifier and regressor predictions are written directly to device buffers. The regressor evaluates all 512 leaf positions to avoid reading a variable active count back to the CPU. A final CUDA kernel applies occupancy flips and exact-value corrections and writes the final leaf arrays on the GPU. The leaf owner also retains intermediate buffers until destruction, so the returned CUDA stream can be consumed asynchronously. Only the small correction lists and normalized coordinates cross from host to device; the predicted occupancy/value arrays are never downloaded. This is a device-leaf API, not yet a NanoVDB grid writer or a ZenoCompute node. CUDA compilation and execution still require a CUDA-capable build with the pinned tiny-cuda-nn dependency and have not been validated by the CPU-only workflow.


## GPU NanoVDB bridge with native tiles

`NeuralVolumeDecoder::buildNanoVDBDevice()` builds a device-resident
`nanovdb::NanoGrid<float>` with the original active leaf occupancy and active
internal tiles represented in the native NanoVDB tree.

1. Each neural leaf is decoded on the GPU. Only its 512-byte occupancy mask is
   copied to the CPU to construct sparse topology; the 512 float values remain
   on the GPU.
2. The CPU-side NanoVDB `GridBuilder<ValueOnIndex>` creates active leaf entries
   and inserts each active tile at its recorded tree level (1, 2, or 3).
3. `CreateNanoGrid` serializes that sparse index topology. The index grid is
   uploaded to the GPU using NanoVDB's `cuda::copyTo`.
4. CUDA kernels map each decoded leaf coordinate and each active tile
   coordinate to its `ValueOnIndex` slot. Neural leaf values and original tile
   values are written into a GPU sidecar.
5. NanoVDB's upstream `IndexToGrid<float>` converts the indexed topology and
   sidecar into a FloatGrid on the GPU. The affine transform, grid name, grid
   class, and checksum are preserved/finalized.

The returned move-only `NanoVDBDeviceGrid` owns the device buffer. Keep it alive
while using `gridDevice()`; the pointer is an opaque device pointer to
`nanovdb::NanoGrid<float>`. `bufferSize()`, `activeVoxelCount()`,
`sourceActiveTileCount()`, and `wait()` expose storage size, leaf voxel
count, original active tile count, and completion. Active tiles are no longer
exposed as a separate sidecar: they are part of the NanoVDB tree itself.

The topology builder performs CPU work proportional to active leaf voxels and
tiles, but it does not reconstruct a CPU float volume or an OpenVDB grid. This
is deliberate: upstream NanoVDB's current `MergeGrids` CUDA operator rejects
inputs containing value tiles, so it is not used for this operation.

### Pinned NanoVDB headers

The legacy `projects/zenvdb/openvdb` submodule remains unchanged. A separate
optional submodule, `projects/zenvdb/nanovdb-upstream`, pins the official
`AcademySoftwareFoundation/openvdb` revision
`f28614d90ce02736865e5db1da10b931f78e67a3`. This avoids mixing legacy
OpenVDB headers with modern NanoVDB CUDA builder headers in the same
translation unit. Initialize submodules recursively before enabling
`ZENO_WITH_TINY_CUDA_NN`. CMake compiles the neural CUDA bridge as a separate
object target with only the pinned upstream NanoVDB include root. If that
submodule is absent, the neural leaf decoder can still be built, but the
NanoVDB conversion method reports that the required headers are unavailable.

### Validation boundary

The source update does not constitute a CUDA build. The new path still
requires compilation with the project's CUDA toolkit, tiny-cuda-nn, and the
pinned NanoVDB headers, followed by GPU tests containing leaf voxels and active
tiles at levels 1, 2, and 3. CPU-only CI cannot establish CUDA compilation or
runtime correctness. The code now constructs native tile topology, but a
compiled GPU round-trip is still required before claiming validated output.
Tile-only checkpoints are still rejected because upstream `IndexToGrid` has
a zero-leaf launch path that must be handled separately.

Minimal C++ usage:

```cpp
#include <zeno/TinyCudaNNRuntime.h>
#include <nanovdb/NanoVDB.h>

using namespace zeno::neuralvolume;
auto model = readZNVMFile(checkpointPath);
NeuralVolumeDecoder decoder(std::move(model));
auto deviceGrid = decoder.buildNanoVDBDevice();
auto* grid = static_cast<const nanovdb::NanoGrid<float>*>(deviceGrid.gridDevice());
// Keep deviceGrid alive while grid is used by CUDA kernels.
```

The API returns a device-only grid handle; it does not write a `.nvdb` file or
register a Zeno graph node. Those are separate integration tasks.


## Bounded multi-domain residency

The decoder now uses a two-entry LRU cache for domain runtimes. This is a
bounded working-set optimization, not a new neural hierarchy: each domain
still owns its own local classifier and optional regressor. The cache avoids
recreating networks and uploading parameters when traversal revisits either
of the two most recently used domains. A third distinct domain evicts the
least-recently used runtime, keeping memory bounded. The cache capacity is
currently a compile-time constant and should only become configurable after
VRAM measurements on target GPUs.
