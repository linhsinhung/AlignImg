# alignimg-gpu

Optional native-CUDA and CuPy backends for AlignImg 2.3. The native path uses a
compiled persistent CUDA transform session and a fused Fourier M-step;
FFT, NCC reduction, controller stages, and final raw-output accumulation use
CuPy. The workflow controller
and polar bootstrap remain CPU-authoritative, so all engines share the same
public contracts. Adaptive refinement also shares the CPU posterior-mass
controller; ragged coarse/fine cells are packed into flat buffers and
Fourier-NCC scored in VRAM-bounded CuPy chunks. Native CUDA or CuPy supplies
transforms, while the GPU soft M-step consumes the complete packed fine
posterior rather than only the public top-L shortlist.

Both GPU transform paths accept even-sized square
images and rotate/mirror about the integer origin `(size//2, size//2)`.

## Installation

From the repository root, install the core first, then build the GPU package
with the extra matching your CUDA runtime:

```bash
python -m pip install .
python -m pip install -v './packages/alignimg-gpu[cuda12]'
# CUDA 13 alternative: './packages/alignimg-gpu[cuda13]'
```

These are source builds, not downloads of a prebuilt CUDA wheel. Both GPU
backends require Linux x86-64, CuPy 14+, and a compatible NVIDIA driver. Install
only one CuPy distribution in an environment. Native compilation additionally
requires CUDA Toolkit 12+, C++17, and CMake 3.24+; pip manages the Python build
dependencies during a normal isolated build.

With `nvcc`, the build compiles `_native`; without it, only the CuPy fallback is
installed. For example, explicitly target an RTX 3090:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS='-DCMAKE_CUDA_ARCHITECTURES=86' \
  python -m pip install -v './packages/alignimg-gpu[cuda12]'
```

Adjust the path and architecture for your host. See the
[installation guide](../../docs/INSTALLATION.md) for explicit fallback builds,
in-place upgrades, and package/native version checks. Rebuilding the native
extension is required when upgrading its sources or version stamp; an editable
Python installation alone does not refresh an already compiled binary.

## Backend selection and execution

Use `backend="cuda"` to require the native engine, `backend="cupy"` to require
the fallback, or `backend="auto"` for CUDA → CuPy → CPU selection. Explicit
requests never silently fall back. The legacy `backend="gpu"` name selects the
best installed GPU engine.

VRAM allocation is budgeted from current free memory, with an 80% default hard
limit. Automatic batching uses a performance-oriented soft cap of 256;
explicitly requesting 512 or 1024 bypasses that soft cap but never the VRAM
budget. An OOM halves the batch and retries down to one item.

The default 2.x path uses the Fourier-native complex rotation/translation and
candidate-NCC engine validated in 1.8, followed by the Fourier-domain M-step
validated in 1.9. It uses the same
native CUDA or CuPy transform convention, VRAM-bounded particle-DFT caching,
complex weighted accumulation, and one output IFFT per reference. Spatial
reference update and raster candidate scoring remain explicit frozen comparison
paths.

The quadratic refinement preset uses bounded Fourier correlation maps followed by
continuous independent x/y and angular quadratic fitting. Native CUDA performs
bounded-window peak selection and fitting without downloading correlation maps;
CuPy provides the portable GPU implementation. Correlation-map chunks obey the
same VRAM budget and are recorded in result metadata.

AlignImg 2.3 adds the opt-in `fast3` and configurable `fast_hard` presets. The
native polar path samples bounded translation centers and selects angular peaks
on the GPU, using CuPy batched angular FFT/IFFT. Only final winners, scores, and
margins return to the host; full correlation maps stay on the device. The same
Fourier reference updater and final raw-average accumulator serve hard and soft
inference. `fast3` runs three iterations and enables final raw reconstruction.
The core, GPU package, and native build stamp must all be `2.3.1`.

In the original 2.3.1 freeze, polar inference chooses resident particles only when the full stack
and requested batch workspace fit the shared budget. Otherwise it streams
particle batches, accounting for live Fourier caches without counting them
twice. Recovery tries streaming, evicts rebuildable caches, then halves the
batch; an infeasible one-particle plan fails before upload. Full correlation
maps remain on device. See the [memory policy](../../docs/POLAR_231_T4_MEMORY_POLICY.zh-TW.md)
and [2.3.1 release status](../../docs/RELEASE_FREEZE_2_3_1.md).

The separately frozen [spatial-cache snapshot](../../docs/RELEASE_FREEZE_POLAR_SPATIAL.md)
retains prepared FP32 spatial particles across iterations of one polar workflow
with Fourier reference updates, when the shared budget permits. Source residency
does not require the entire requested solver batch to fit at once. Admission is
rechecked each iteration; Fourier workspace has priority and may evict this
optional cache. Eviction or upload OOM disables it for the rest of the workflow,
which then uses the original per-call resident/streaming paths. References and
translation grids still refresh each iteration, and workflow exit releases the
cache. This is not a raw-image cache and cannot replace final raw-output uploads.
The package/native stamps remain `2.3.1`; use the snapshot manifest and checksums
to distinguish it from the original release artifacts.

When `apply_final_pose_to_raw=True`, the input NumPy stack is still streamed to
the selected GPU backend in VRAM-bounded batches. Final-pose transforms and
FP64 hard-class accumulation remain on the device; only the final K class
averages return to the host. This removes the previous N-image aligned-stack
download but does not add disk streaming or remove the required input H2D
transfer.

Empirical radial whitening, introduced in 1.10, remains optional and applies
CPU-authoritative
weights during both native-CUDA and CuPy translation search and Fourier-NCC
reduction. Weight estimation is deterministic and performed once from the input
particle stack; the additional GPU storage is one image-sized real array.
