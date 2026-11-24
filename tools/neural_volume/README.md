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

Validate the output container with:\n\n```bash\npython tools/neural_volume/inspect_znvm.py output.znvm\n```\n\n## First-pass model

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

- This is an experimental encoder checkpoint, not a production NeuralVDB replacement.
- `.znvm` is a portable little-endian binary container with tiny-cuda-nn model parameters and correction records. There is no C++ Zeno decoder yet.
- Exact occupancy reconstruction depends on matching tiny-cuda-nn inference behavior and numeric precision between encoding and decoding.
- Values not covered by corrections are lossy within the requested tolerance.
- FloatGrid and linear/affine transforms only.
- The first version retains the leaf block list and upper active tiles explicitly; it does not yet learn the parent-level VDB child/tile classifier. Domains are currently disjoint and are not blended, so boundary continuity is not yet guaranteed.
- Training requires an NVIDIA GPU, CUDA-enabled PyTorch and tiny-cuda-nn. The Python extension does not run on CPU.
- The binary inspector validates container structure and counts; it does not prove model reconstruction quality.

## Reproducibility and license

The tiny-cuda-nn dependency is pinned to upstream commit
`3daa6e5b8b8e0fa17971a1a20acfbb3754494b38`. Its upstream `LICENSE.txt`
contains a BSD 3-Clause-style license with NVIDIA attribution and non-endorsement
conditions. Preserve that notice if redistributing the dependency or binaries.

The trainer reports IoU, precision/recall, classifier accuracy, correction
counts and pre-correction regression RMSE. Do not compare only checkpoint size:
the final codec must include decoder/model memory, exact corrections, tiles,
metadata and runtime buffers in its footprint.
