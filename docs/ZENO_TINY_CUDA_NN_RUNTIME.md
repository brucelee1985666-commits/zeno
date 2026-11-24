# Optional tiny-cuda-nn CUDA runtime

CUDA inference is opt-in. Set ZENO_WITH_TINY_CUDA_NN=ON only when the parent
build has already provided the upstream tiny-cuda-nn CMake target and enabled
CUDA. The normal Zeno build and CPU .znvm reader remain independent.

TinyCudaNNRuntime recreates the HashGrid + CutlassMLP from the model JSON,
checks parameter count and precision, uploads weights once, and converts
float32 checkpoint weights to FP16 on the GPU when required by the pinned
tiny-cuda-nn build. It batches inference in multiples of the upstream
256-element granularity. Input is domain-local normalized xyz; output is a raw
scalar prediction. Classifier logits are thresholded at zero. Regressor output
must be mapped using the domain's valueCenter and valueScale.

This is a low-level inference API only. It does not yet select domains, decode
whole volumes, apply leaf corrections, or register a Zeno renderer node.
No CUDA build or GPU inference is claimed by the existing CPU-only CI.
