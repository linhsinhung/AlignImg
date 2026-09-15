# AlignImg

**2-D cryo-EM particle alignment with reference-free initialization,
multi-reference alignment, and continuous pose refinement.**

AlignImg 2.2 estimates in-plane rotations and translations, then reconstructs
reference/class-average images. Its Python API can be used before or between
steps of an external classification workflow. EMAN2 and RELION are not required.

A complete CPU backend is included. An optional GPU package provides native CUDA
acceleration and a CuPy fallback. The default engine uses Fourier NCC scoring,
cached particle Fourier transforms, and soft Fourier reference updates.

[Features](#features) · [Installation](#installation) ·
[Quick start](#quick-start-align-a-stack-to-a-reference) ·
[RF and class feedback](#other-alignment-workflows) ·
[Outputs](#results-and-raw-particle-averages) · [GUI](#optional-gui)

## Features

| Task | Public API | Inputs |
| --- | --- | --- |
| Reference-free alignment (RF) | `reference_free_align()` | Particles and a chosen number of components `K`; references are initialized from the particles. |
| Single-reference alignment / multi-reference alignment (MRA) | `align_to_references()` | Particles and one or more initial references; no prior poses needed. |
| Local pose refinement | `refine_alignment()` | Particles, references, and existing poses. |
| Classification feedback | `make_class_priors()` + `refine_alignment()` | External labels or soft class probabilities, class averages, and existing poses. |
| Apply saved poses | `transform_images()` | An image stack and a `PoseSet`. |

K=1 and K>1 use the same alignment engine. References are updated during
iterations; supplied references are initial models, not immutable targets.
Class priors can fix each particle to one reference or allow reassignment.

Additional capabilities include:

- Subpixel/subdegree refinement using quadratic peak fitting.
- Soft class responsibilities, pose candidates, uncertainty, and robust inlier weights.
- Optional mirror search, disabled by default.
- Reference history, half-set Fourier ring correlation (FRC), and execution diagnostics.
- Optional output averages reconstructed from input particles using final poses.
- GPU memory budgeting and batch processing on a single NVIDIA GPU.

AlignImg handles **2-D in-plane alignment**. Particle picking, CTF
estimation/correction, denoising, biological classification, and 3-D
reconstruction belong in the surrounding workflow. MRA assignments describe
alignment components; they are not a validated biological classification.

## Installation

### Requirements

- **Python 3.10 or newer**; Python 3.12 is a tested choice.
- **CPU:** NumPy, SciPy, and OpenCV headless, installed automatically.
- **MRC/MRCS input/output:** `mrcfile`, included by the optional `[io]` extra.
- **GPU:** Linux x86-64, a single NVIDIA CUDA GPU, a compatible NVIDIA driver,
  and CuPy 14 or newer. Both AlignImg GPU backends currently require Linux
  x86-64; use the CPU backend on other platforms.
- **Native CUDA build:** CUDA Toolkit 12 or newer with `nvcc`, a compatible
  C++17 host compiler, and CMake 3.24 or newer. The build uses scikit-build-core
  and pybind11; pip manages the Python build dependencies.

The 2.2 release was tested on Linux x86-64 with Python 3.12, an RTX 3090
(24 GB), CUDA Toolkit 12.8, and CuPy 14.2.0. This is a tested configuration,
not a minimum GPU-memory requirement. Host RAM must accommodate the particle
array and working copies; GPU batching does not provide disk streaming.

### 1. Install the CPU package

Clone the repository or download and extract its source archive. Run the
installation commands from the repository root:

```bash
git clone https://github.com/linhsinhung/AlignImg.git
cd AlignImg

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[io]"
```

The activation command above is for a POSIX shell. An existing Conda environment
also works. Use `python -m pip install .` if you only need the NumPy-array API,
or `python -m pip install -e ".[io]"` for an editable development installation.

The CPU package does not install CUDA, CuPy, or GUI dependencies.

### 2. Optionally install GPU support

Install the CPU package first, then choose one of the following paths.

**Native CUDA**

With CUDA 12.x and `nvcc` discoverable on the build host:

```bash
python -m pip install -v "./packages/alignimg-gpu[cuda12]"
```

On CUDA 13.x, use `[cuda13]` instead. Select the extra for the CUDA
toolkit/runtime you intend to use, not just the maximum CUDA version displayed
by `nvidia-smi`. Install only one CuPy distribution in an environment; see the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

By default, native compilation targets the build machine's GPU. To specify
the compiler and architecture explicitly, for example for an RTX 3090:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" \
python -m pip install -v "./packages/alignimg-gpu[cuda12]"
```

Adjust the compiler path and architecture for your host. Without a discoverable
CUDA compiler, the package builds only the CuPy fallback.

**CuPy fallback without compiling the native extension**

To explicitly skip the native CUDA build:

```bash
CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=NOTFOUND" \
python -m pip install "./packages/alignimg-gpu[cuda12]"
```

This still requires a working CuPy CUDA runtime and NVIDIA driver. It uses
CuPy's runtime-compiled kernels. The package build still configures CMake and
a host C++ compiler. See the [GPU package guide](packages/alignimg-gpu/README.md)
for execution and memory details.

### 3. Check the installation and select a backend

```python
import alignimg as ai

print(ai.__version__)
print(ai.available_alignment_backends())
```

Every alignment and transform call accepts `backend=`:

| Value | Behavior |
| --- | --- |
| `"cpu"` | CPU execution; the default when the argument is omitted. |
| `"cuda"` | Require the compiled native CUDA backend, with CuPy for FFTs and other GPU operations. |
| `"cupy"` | Require the portable CuPy GPU backend. |
| `"auto"` | Select the available backend in the order CUDA → CuPy → CPU. |

Explicit `"cuda"` and `"cupy"` requests report an error if unavailable.
The compatibility alias `"gpu"` selects the best installed GPU backend.
GPU work is constrained by `memory_fraction` (default `0.8`); a requested
`batch_size` may be reduced to fit the available budget.

## Input data

The API accepts **in-memory NumPy arrays**, not filenames:

- Particles: shape `(N, H, H)`, with `N > 0`.
- References: shape `(H, H)` for one reference, or `(K, H, H)`.
- `H` must be even. Particles and references must have matching box sizes and
  pixel sampling.
- Values must be finite; inputs are converted to `float32`.
- Prepare CTF-corrected data externally when needed. Supply references with
  compatible preprocessing.
- Poses are in degrees and pixels. Pixel size is used for physical-unit
  reporting and file metadata, not as an alignment parameter.

## Quick start: align a stack to a reference

This example reads `particles.mrcs` and `reference.mrc` from your working
directory. It performs global alignment, refines the resulting poses, and saves
averages and pose metadata. It requires the `[io]` installation above.

```python
from dataclasses import replace

import mrcfile
import numpy as np
import alignimg as ai

backend = "cpu"  # Use "cuda" or "cupy" after installing GPU support.

with mrcfile.open("particles.mrcs") as mrc:
    particles = np.array(mrc.data, dtype=np.float32, copy=True)
    pixel_size = float(mrc.voxel_size.x)

if particles.ndim == 2:  # A one-image MRC file may be loaded without a stack axis.
    particles = particles[None, :, :]

with mrcfile.open("reference.mrc") as mrc:
    reference = np.array(mrc.data, dtype=np.float32, copy=True)

# Find poses without prior alignment parameters.
global_config = replace(
    ai.AlignmentConfig.preset("global_accurate"),
    max_iterations=3,
    batch_size=512,
    memory_fraction=0.8,
)
initial = ai.align_to_references(
    particles, reference, config=global_config, backend=backend,
)

# Improve the existing poses using continuous quadratic refinement.
refine_config = replace(
    ai.AlignmentConfig.preset("refine"),
    max_iterations=2,
    batch_size=512,
    memory_fraction=0.8,
)
result = ai.refine_alignment(
    particles,
    initial.references,
    initial.poses,
    config=refine_config,
    backend=backend,
)

with mrcfile.new("class_averages.mrcs", overwrite=True) as mrc:
    mrc.set_data(result.class_averages.astype(np.float32))
    mrc.set_image_stack()
    mrc.voxel_size = pixel_size

np.savez_compressed(
    "alignment.npz",
    alignimg_version=np.asarray(ai.__version__),
    center_convention=np.asarray(result.metadata["center_convention"]),
    pixel_size_angstrom=np.asarray(pixel_size),
    angle_deg=result.poses.angle_deg,
    shift_y_px=result.poses.shift_y_px,
    shift_x_px=result.poses.shift_x_px,
    mirror=result.poses.mirror,
    assignments=result.reference_assignments,
    responsibilities=result.responsibilities,
    inlier_weights=result.inlier_weights,
    references=result.references,
    class_averages=result.class_averages,
)
```

For MRA, load a reference stack of shape `(K, H, H)` instead; the calls are
identical. Without `class_priors`, particles may switch references during
alignment and refinement.

Use a workflow preset and `dataclasses.replace()` when changing parameters.
Passing `AlignmentConfig(max_iterations=2)` directly to refinement selects
that object's default `"proposal"` strategy, not the refine preset.

The refine preset uses `quadratic_refine`: angles within ±7° of the current
pose, screened at 1° intervals, and translations within ±3 pixels. It fits
continuous peak positions and verifies them with Fourier NCC plus pose priors.
Refinement needs an initial alignment within a useful local neighborhood.
The example iteration counts are starting points, not convergence guarantees.

## Other alignment workflows

These examples reuse `particles`, `backend`, and the imports from the quick
start. Reference indices and class labels are **zero-based**.

### Reference-free alignment

Choose `n_components=1` for a homogeneous view, or a larger K for several views:

```python
rf_config = replace(
    ai.AlignmentConfig.preset("reference_free"),
    max_iterations=15,
    temperature_anneal_iterations=10,
    batch_size=512,
    memory_fraction=0.8,
    random_seed=0,
)
rf = ai.reference_free_align(
    particles, n_components=10, config=rf_config, backend=backend,
)
```

RF initializes references from the particles and uses global proposal search
throughout. This example anneals for 10 iterations and holds the final
temperature for five more. Choose K explicitly; AlignImg does not automatically
estimate K or merge/split classes. RF results have an arbitrary overall
alignment frame.

### Class-average feedback: fixed or corrective refinement

Here, `class_averages` is a `(K, H, H)` array, `labels` contains N integer
class indices, and `initial_poses` is a `PoseSet` for the same particles in
the same order. References and poses must use a consistent coordinate frame.

```python
fixed_priors = ai.make_class_priors(
    assignments=labels,
    n_components=len(class_averages),
    trust=1.0,
)
fixed = ai.refine_alignment(
    particles, class_averages, initial_poses,
    class_priors=fixed_priors, config=refine_config, backend=backend,
)
```

One-hot priors with `trust=1.0` keep assignments fixed. To allow corrective
reassignment, use `trust < 1`, optionally starting from external soft
probabilities of shape `(N, K)`:

```python
corrective_priors = ai.make_class_priors(
    responsibilities=class_probabilities,
    trust=0.9,
)
corrective = ai.refine_alignment(
    particles, class_averages, initial_poses,
    class_priors=corrective_priors, config=refine_config, backend=backend,
)
```

Supply exactly one of `assignments` or `responsibilities`. The helper
normalizes soft rows and mixes the prior as
`trust * prior + (1 - trust) / K`. This allows reassignment; it does not
require any particle to change class.

### Apply or reload poses

Apply final poses to a matching stack, including raw particles when poses were
estimated from externally denoised images:

```python
# raw_particles must match the estimation stack's order, box size, and frame.
aligned_raw = ai.transform_images(raw_particles, result.poses, backend=backend)

# Reload the poses saved by the quick-start example.
with np.load("alignment.npz", allow_pickle=False) as saved:
    poses = ai.PoseSet(
        angle_deg=saved["angle_deg"],
        shift_y_px=saved["shift_y_px"],
        shift_x_px=saved["shift_x_px"],
        mirror=saved["mirror"],
    )
```

`transform_images()` returns the complete aligned stack. It does not average
or classify the particles.

## Results and raw-particle averages

All three alignment workflows return an `AlignmentResult`:

| Field | Meaning |
| --- | --- |
| `poses` | One final MAP pose per particle: `angle_deg`, `shift_y_px`, `shift_x_px`, `mirror`. |
| `references` | Updated soft Fourier reference models, shape `(K, H, H)`. |
| `class_averages` | Output averages, shape `(K, H, H)`; equal to `references` by default. |
| `reference_assignments` | Final hard reference indices, shape `(N,)`. |
| `responsibilities` | Soft class probabilities, shape `(N, K)`, with rows summing approximately to one. |
| `inlier_weights` | Per-particle weights used for robust averaging. |
| `candidates`, `pose_entropy`, `map_posterior` | Retained pose hypotheses and uncertainty summaries. |
| `diagnostics`, `reference_history`, `metadata` | Iteration diagnostics, model history, actual backend, configuration, and GPU memory plans when applicable. |

To reconstruct the final average using each input particle's final MAP pose,
hard class assignment, and inlier weight, enable this option before running:

```python
raw_output_config = replace(refine_config, apply_final_pose_to_raw=True)
raw_output = ai.refine_alignment(
    particles, initial.references, initial.poses,
    config=raw_output_config, backend=backend,
)
raw_averages = raw_output.class_averages
soft_references = raw_output.references
```

This adds a final reconstruction step with the configured circular mask while
preserving the soft references used during inference. On GPU, accumulation is
batched and only K averages return to the host.

Here, **raw means the array supplied as `images` to that call**. The option
does not load another raw file. If alignment used denoised inputs, apply the
poses to your separately held raw stack with `transform_images()`.

### Pose convention

A `PoseSet` maps input images into reference coordinates in this order:

1. Optional periodic left-right mirror about the integer x origin.
2. Counter-clockwise rotation about `(H // 2, H // 2)`.
3. Translation by `(shift_y_px, shift_x_px)`; positive shifts move toward
   increasing row/column indices.
4. Periodic (wrap) boundaries.

Angles and shifts use `float32` and may contain fractions. Returned poses
are complete input-to-reference transforms, not increments to add to prior
poses. The integer center matches RELION's center convention, but external pose
formats may still need angle/sign/translation conversion. Legacy pose adapters
are documented in the [API guide](docs/API.md).

## Optional GUI

[AlignImg Workbench](packages/alignimg-gui/README.md) provides MRC-stack input,
RF/MRA workflows, optional refinement, and inspection of averages, convergence,
uncertainty, and logs. Install it after the core package:

```bash
python -m pip install ./packages/alignimg-gui
alignimg-gui
```

The GUI requires PyQt6 and a graphical desktop. It works with the CPU backend or
the separately installed GPU package.

## Documentation and validation

- [Public API and configuration](docs/API.md)
- [Unified alignment model](docs/UNIFIED_ALIGNMENT_FRAMEWORK.zh-TW.md) (Traditional Chinese)
- [Continuous quadratic refinement](docs/CONTINUOUS_QUADRATIC_REFINEMENT_2_2.md)
- [GPU installation and execution](packages/alignimg-gpu/README.md)
- [2.2 release and validation summary](docs/RELEASE_FREEZE_2_2.md)

For development tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

For a backend smoke report from the source checkout:

```bash
python tools/server_validation.py \
  --suite quick --backend cpu \
  --output validation-results/cpu-quick.json
```

Use `--backend cuda` or `--backend cupy` to validate an installed GPU backend.

## License

AlignImg is distributed under **GPL-3.0-or-later**. See [LICENSE](LICENSE) and
[third-party notices](THIRD_PARTY_NOTICES.md).
