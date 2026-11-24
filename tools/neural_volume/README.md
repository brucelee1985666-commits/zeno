# Zeno Neural Volume prototype

This folder contains the first offline neural-compression experiment for Zeno.

## Current pipeline

1. In Zeno, connect a FloatGrid to the `ExportNeuralVolumeDataset` node and write a `.znvds` file.
2. Install a PyTorch build matching the local CUDA toolkit/driver.
3. Fetch the pinned tiny-cuda-nn source **with its required submodules** and install its PyTorch extension. A direct `pip install git+...` is insufficient because the upstream build uses CUTLASS and fmt submodules:
   
   ```bash
   git clone https://github.com/NVlabs/tiny-cuda-nn.git
   cd tiny-cuda-nn
   git checkout 3daa6e5b8b8e0fa17971a1a20acfbb3754494b38
   git submodule update --init --recursive
   pip install ./bindings/torch
   cd ..
   ```

4. Install NumPy if it was not installed with PyTorch:
   
   ```bash
   pip install "numpy>=1.24"
   ```

5. Train a prototype checkpoint:
   
   ```bash
   python tools/neural_volume/train_znvds.py input.znvds output.znvm
   ```

Validate the output container with:

```bash
python tools/neural_volume/inspect_znvm.py output.znvm
```

## GPU-free validation path

You do not need an NVIDIA GPU to validate an exported `.znvds` dataset:

```bash
python tools/neural_volume/inspect_znvds.py input.znvds --domain-size 128
```

The standard-library CPU audit checks finite values, unique/aligned leaf origins,
occupancy/value consistency, domain assignment (including negative coordinates),
active samples on domain boundary planes, and a SHA-256 digest of the ordered
active coordinate/value records. It does not train or execute a neural network.

CI also builds a small FloatGrid with NanoVDB's upstream CPU `GridBuilder`,
converts it with `createNanoGrid`, and checks active leaf values and an active
tile on the CPU. This verifies the upstream NanoVDB CPU data path, not the
tiny-cuda-nn inference path.

The repository also contains a CPU-only end-to-end converter:

- `znvds_to_nanovdb.cpp` parses and validates ZNVDS v1, reconstructs active leaf
  voxels and active internal tiles with NanoVDB's upstream CPU `GridBuilder`,
  then emits a host `NanoVDB FloatGrid` with `createNanoGrid`.
- The conversion preserves the stored affine index-to-world transform, grid
  name, grid class, background, active topology and scalar values.
- CI writes a synthetic ZNVDS fixture and verifies voxel occupancy/values, active
  tiles at all three NanoVDB hierarchy levels, grid class/name and a non-uniform
  affine transform after conversion.
- The smoke test then serializes the grid to `.nvdb`, reopens that exact file with
  NanoVDB's `io::readGrid`, and repeats the topology/value/metadata/transform
  assertions against the reloaded grid. It also checks that truncated ZNVDS and
  NanoVDB files are rejected.
- This is a lossless CPU reference path for the ZNVDS interchange format; it is
  not a neural decoder and does not execute tiny-cuda-nn.

Build the standalone converter against the pinned upstream NanoVDB headers:

```bash
g++ -std=c++17 -pthread -O2 \
  -isystem projects/zenvdb/nanovdb-upstream/nanovdb \
  tools/neural_volume/znvds_to_nanovdb.cpp \
  tools/neural_volume/znvds_to_nanovdb_main.cpp \
  -o znvds-to-nanovdb
./znvds-to-nanovdb input.znvds output.nvdb
```

The CLI writes a standard NanoVDB file. It does not require CUDA or tiny-cuda-nn.

### Real OpenVDB FloatGrid end-to-end CPU proof

The CPU integration test constructs an actual `openvdb::FloatGrid` in C++,
including active leaf voxels, active tiles at all three tree levels, a non-uniform
affine transform, grid class, name and background. It passes that in-memory grid
through the same reusable serializer called by the Zeno
`ExportNeuralVolumeDataset` node, producing a real `.znvds` file. The test then
runs the production ZNVDS-to-NanoVDB converter, writes `.nvdb`, reopens the
serialized file with NanoVDB's reader, and compares every active OpenVDB value
and tile against the reloaded NanoVDB grid, as well as metadata and the
index-to-world transform.

The runtime smoke links against the Ubuntu `libopenvdb-dev` package to avoid
rebuilding the full OpenVDB library in CI. Separately, CI syntax-compiles the
Zeno exporter node and shared serializer against the fork's pinned OpenVDB
headers, so the runtime test and production-header compatibility are checked
as distinct layers.

Run it on a CPU-only machine with OpenVDB development headers and the pinned
NanoVDB submodule:

```bash
sudo apt-get install libopenvdb-dev pkg-config
cmake -S tools/neural_volume/openvdb_roundtrip -B build-openvdb-floatgrid \
  -DOPENVDB_SOURCE_DIR="$PWD/projects/zenvdb/openvdb" \
  -DNANOVDB_INCLUDE_DIR="$PWD/projects/zenvdb/nanovdb-upstream/nanovdb"
cmake --build build-openvdb-floatgrid --target openvdb-floatgrid-roundtrip-smoke -j2
./build-openvdb-floatgrid/openvdb-floatgrid-roundtrip-smoke
```

This exercises real OpenVDB CPU data structures and the full interchange
pipeline. It does not train or run the tiny-cuda-nn model and does not require
an NVIDIA GPU.

### Zeno node integration

When the pinned `projects/zenvdb/nanovdb-upstream` submodule is initialized,
the `projects/zenvdb` CMake configuration enables the CPU converter as an
isolated object target and registers the `ZNVDSFileToNanoVDB` node in the
`nanovdb` category. The isolated target keeps the modern pinned NanoVDB
headers away from the legacy NanoVDB translation units.

The node has no sockets; set its `inputpath` parameter to an existing
`.znvds` file and `outputpath` to the destination `.nvdb` file. It performs
the same validated CPU conversion as the standalone CLI. It does not invoke
the neural model or require CUDA/tiny-cuda-nn.

The tiny-cuda-nn trainer remains optional and NVIDIA-only. If no compatible GPU
is available, use the CPU audit and CPU NanoVDB tests; do not substitute a
CPU mock and call it a neural training/inference result.

### Remote NVIDIA training without a local GPU

For users without an NVIDIA GPU, `colab_remote_gpu_training.ipynb` provides an
optional Google Colab path. It checks that a real CUDA device was allocated,
builds the pinned tiny-cuda-nn revision, accepts a user-supplied ZNVDS export,
runs the existing trainer and validates the resulting ZNVM checkpoint. Open
the notebook in Colab and select a T4 GPU runtime when one is available.

This is an on-demand remote GPU session, not a promise of free or continuous
GPU availability. The user must start and authorize the session and provide a
real `.znvds` dataset. Do not interpret GitHub's CUDA compile-only job as
training or GPU inference.

## First-pass model

- Domain size: 128³ voxels by default.
- Per-domain occupancy classifier: tiny-cuda-nn HashGrid + CutlassMLP.
- Per-domain active-value regressor: tiny-cuda-nn HashGrid + CutlassMLP.
- Explicit leaf-block coordinates and active internal tiles.
- Sparse occupancy corrections make the predicted leaf masks exact under the same inference backend/configuration.
- Exact value corrections are stored when prediction error exceeds `--value-tolerance`.

Useful controls:

- `--steps 2000`: training iterations per model and domain.
- `--domain-size 128`: must be a positive multiple of 8.
- `--value-tolerance 0.001`: maximum accepted absolute value prediction error before storing an exact correction.
- `--max-domains 1`: quick, bounded first-domain experiment.

## Important limitations

- This is an experimental hybrid encoder/decoder, not a production NeuralVDB replacement.
- `.znvm` is a portable little-endian binary container with tiny-cuda-nn model parameters and correction records. A C++ reader and optional CUDA leaf decoder are present; the CUDA bridge is not yet integrated as a ZenoCompute/render node.
- Exact occupancy reconstruction depends on matching tiny-cuda-nn inference behavior and numeric precision between encoding and decoding.
- Values not covered by corrections are lossy within the requested tolerance. Active values on the six domain-boundary voxel planes are stored exactly to avoid seams between adjacent local fields.
- FloatGrid and linear/affine transforms only.
- The VDB hierarchy is retained explicitly: active internal tiles remain native NanoVDB tiles, and leaf blocks are represented by local neural domains. The parent-level topology is not learned by a separate neural classifier; this is a deliberate first hybrid design, not full NeuralVDB parity.
- The optional runtime can decode one leaf on demand and can build a device-resident NanoVDB FloatGrid. Building the full FloatGrid materializes decoded scalar values and therefore gives up much of the checkpoint's neural-memory savings; it is a compatibility/fallback path, not the compressed streaming path.
- The decoder keeps up to two domain model pairs resident in an LRU cache. It is not thread-safe per instance.
- Training requires an NVIDIA GPU, CUDA-enabled PyTorch and tiny-cuda-nn. The Python extension does not run on CPU.
- CI validates the binary reader/parser, NanoVDB API and native tile topology, and performs a compile-only CUDA bridge check. No actual GPU training/inference, full Zeno build, renderer integration, or performance benchmark has been completed.

## Reproducibility and license

The tiny-cuda-nn dependency is pinned to upstream commit
`3daa6e5b8b8e0fa17971a1a20acfbb3754494b38`. Its upstream `LICENSE.txt`
contains a BSD 3-Clause-style license with NVIDIA attribution and non-endorsement
conditions. Preserve that notice if redistributing the dependency or binaries.

The trainer reports IoU, precision/recall, classifier accuracy, correction
counts and pre-correction regression RMSE. Do not compare only checkpoint size:
the final codec must include decoder/model memory, exact corrections, tiles,
metadata and runtime buffers in its footprint.
