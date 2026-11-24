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
4. Runs the value regressor only for predicted active voxels, if that domain
   has one, then restores values using the stored center and scale.
5. Applies exact occupancy flips and exact-value corrections from the ZNVM.
6. Returns one `DecodedNeuralLeaf` with 512 occupancy flags and values.

A domain without a regressor must provide exact values for every active voxel;
the decoder checks this rather than silently returning background values.

## Scope and validation boundary

This is lazy GPU neural inference with one-leaf CPU output. It avoids
reconstructing the entire volume as a CPU OpenVDB grid, but it is not yet a
direct NanoVDB GPU-grid writer, a ZenoCompute device object, or a renderer node.
The caller receives decoded leaf arrays on the CPU and can inspect the original
source metadata, affine transform, and preserved active tiles through
`sourceModel()`. A subsequent integration must stream these leaf results into
a GPU-resident NanoVDB/volume representation without expanding the full tree
on the host. A decoder instance is not thread-safe because it maintains one
active domain runtime; serialize calls per instance or use one instance per
worker.

The checkpoint format currently stores one shared model configuration for
classifier and regressor networks; this is valid because the trainer creates
both with the same HashGrid and CutlassMLP architecture. Partial checkpoints
are intentionally rejected by the complete-volume decoder.

No CUDA compilation or GPU inference is claimed by the existing CPU-only CI.
