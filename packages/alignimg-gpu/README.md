# alignimg-gpu

Optional native-CUDA and CuPy backends for AlignImg 2.x. The native path uses a
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

Install the matching wheel extra:

```bash
python -m pip install -e './packages/alignimg-gpu[cuda12]'
# or: python -m pip install -e './packages/alignimg-gpu[cuda13]'
```

On Linux with `nvcc`, the build compiles `_native`; without a CUDA compiler it
installs the CuPy-only fallback. For the RTX 3090 server, build with:

```bash
CMAKE_ARGS='-DCMAKE_CUDA_ARCHITECTURES=86' \
  python -m pip install -v './packages/alignimg-gpu[cuda12]'
```

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

The 2.2 refinement preset uses bounded Fourier correlation maps followed by
continuous independent x/y and angular quadratic fitting. Native CUDA performs
bounded-window peak selection and fitting without downloading correlation maps;
CuPy provides the portable GPU implementation. Correlation-map chunks obey the
same VRAM budget and are recorded in result metadata.

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
