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
every domain's network onto the GPU. Only the requested leaf is decoded, and
only the currently used domain's classifier/regressor weights remain resident.
Switching domains replaces that active runtime.

For each requested leaf the decoder:
1. Resolves the owning domain using floor division (including negative index
   coordinates).
2. Generates the same domain-local normalized xyz coordinates used by the
   Python trainer.
3. Runs the occupancy classifier for the 512 leaf voxels.
4. The CPU convenience path evaluates the regressor for predicted active
   voxels; the device path evaluates all 512 positions to keep the output on
   the GPU, then restores values using the stored center and scale.
5. Applies exact occupancy flips and exact-value corrections from the ZNVM.
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
without expanding the full tree on the host. A decoder instance is not thread-safe because it maintains one
active domain runtime; serialize calls per instance or use one instance per
worker.

The checkpoint format currently stores one shared model configuration for
classifier and regressor networks; this is valid because the trainer creates
both with the same HashGrid and CutlassMLP architecture. Partial checkpoints
are intentionally rejected by the complete-volume decoder.

No CUDA compilation or GPU inference is claimed by the existing CPU-only CI.


## Device-resident decoded leaves

The optional CUDA API also exposes `NeuralVolumeDecoder::decodeLeafDevice()`. It returns a move-only `DecodedNeuralLeafDevice` owner whose opaque device pointers address 512 occupancy bytes and 512 float32 values. The owner keeps both allocations and the producing stream alive. Consumers must retain the owner while using either pointer. `streamHandle()` exposes the opaque CUDA stream so downstream kernels can be enqueued in order without a host round trip; call `wait()` when the CPU needs completion and validation. `wait()` checks a compact device validation flag (one integer), not the 512-voxel arrays.

Classifier and regressor predictions are written directly to device buffers. The regressor evaluates all 512 leaf positions to avoid reading a variable active count back to the CPU. A final CUDA kernel applies occupancy flips and exact-value corrections and writes the final leaf arrays on the GPU. The leaf owner also retains intermediate buffers until destruction, so the returned CUDA stream can be consumed asynchronously. Only the small correction lists and normalized coordinates cross from host to device; the predicted occupancy/value arrays are never downloaded. This is a device-leaf API, not yet a NanoVDB grid writer or a ZenoCompute node. CUDA compilation and execution still require a CUDA-capable build with the pinned tiny-cuda-nn dependency and have not been validated by the CPU-only workflow.


## GPU NanoVDB bridge

`NeuralVolumeDecoder::buildNanoVDBDevice()` decodes the complete set of ZNVM leaf records one leaf at a time and builds a real device-resident `nanovdb::NanoGrid<float>`. It reuses NanoVDB's own CUDA `PointsToGrid<ValueOnIndex>` and `IndexToGrid<float>` builders:

1. Each decoded leaf is appended into GPU arrays of active index-space coordinates and their float values. The append kernel uses the leaf's exact occupancy and corrected values.
2. The append counter is checked against the ZNVM source active-voxel count before topology construction.
3. NanoVDB's CUDA PointsToGrid creates a sparse `ValueOnIndex` tree directly on the GPU.
4. A CUDA scatter kernel maps each coordinate into NanoVDB's assigned active index and fills the value sidecar; IndexToGrid bakes that sidecar into a `FloatGrid` on the GPU.
5. The original affine index-to-world transform and grid name are passed into the builder. The grid class and checksum are finalized before the handle is returned.

The returned move-only `NanoVDBDeviceGrid` owns the NanoVDB device buffer. Keep it alive while using `gridDevice()`; the returned pointer is an opaque device pointer to `nanovdb::NanoGrid<float>`. `bufferSize()`, `activeVoxelCount()`, and `wait()` expose storage size, topology count, and completion. No complete OpenVDB grid or full voxel array is materialized on the CPU; only scalar counters and NanoVDB's small internal builder metadata cross to the host.

CMake enables this bridge only when `ZENO_WITH_TINY_CUDA_NN=ON` and the selected NanoVDB headers contain both `nanovdb/tools/cuda/PointsToGrid.cuh` and `nanovdb/tools/cuda/IndexToGrid.cuh`. Older NanoVDB revisions still compile the neural runtime but the bridge method reports that the GPU builders are unavailable. This feature-detection is intentional: do not silently mix headers from a different NanoVDB/OpenVDB revision.

**Current topology boundary:** this first bridge builds all leaf-level active voxels into one sparse NanoVDB FloatGrid. Active internal tiles recorded in `model.activeTiles` are preserved in the source ZNVM and reported by `sourceActiveTileCount()`, but they are not yet inserted as NanoVDB internal tiles. A model containing only active tiles and no active leaf voxels is therefore rejected. This is not yet a complete topology round-trip for tile-heavy grids. The CUDA path has not been compiled or executed on a GPU in this environment; the existing CPU-only CI does not validate it.
