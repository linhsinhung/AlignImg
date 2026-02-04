# AlignImg
A self-made program for aligning a stack of typical cryo-EM images.

## Installation

AlignImg is a small, self-contained module. Install the Python dependencies and keep the repo on your `PYTHONPATH` (or install it in editable mode).

### 1) Install minimum dependencies

```bash
pip install numpy scipy opencv-python tqdm
```

### 2) Full dependencies (for running tests or demos)

```bash
pip install numpy scipy opencv-python matplotlib mrcfile pandas tqdm
```

### 3) (Optional) Enable GPU acceleration

If you want to use the GPU implementation, install CuPy for your CUDA version (example below uses CUDA 11.x):

```bash
pip install cupy-cuda11x
```

> **Note**: GPU mode requires `cupy` and the local `align_utils_gpu.py` module in this repo. If GPU setup fails, the API automatically falls back to CPU.

## Using AlignImg as a Module

The main entry point is `run_alignment` in `alignimg_api.py`.

### Basic usage (CPU, parallel by default)

```python
import numpy as np
from alignimg_api import run_alignment

# X: (N, H, W) stack of 2D particle images
# initial_ref: (H, W) reference image

final_ref, history, params, meta = run_alignment(
    X,
    initial_ref,
    num_iterations=4,
    mask_diameter=None
)
```

### Force single-core (serial CPU)

```python
final_ref, history, params, meta = run_alignment(
    X,
    initial_ref,
    num_iterations=4,
    n_jobs=1
)
```

### Use GPU (with automatic fallback to CPU)

```python
final_ref, history, params, meta = run_alignment(
    X,
    initial_ref,
    num_iterations=4,
    use_gpu=True,
    batch_size=512
)
```

### Inputs and outputs

- **Inputs**
  - `X`: `np.ndarray` of shape `(N, H, W)` (particle stack).
  - `initial_ref`: `np.ndarray` of shape `(H, W)` (initial reference image).
  - `num_iterations`: number of alignment iterations.
  - `mask_diameter`: circular mask diameter in pixels (or `None` for full image).
  - `use_gpu`: enable GPU processing (requires CuPy; otherwise falls back to CPU).
  - `n_jobs`: number of CPU workers (`1` for serial, `None`/`-1` for all cores).
  - `batch_size`: GPU batch size.

- **Outputs**
  - `final_ref`: the aligned reference image.
  - `history`: list of reference images per iteration.
  - `params`: per-particle alignment parameters `[angle, dy, dx, score]`.
  - `meta`: metadata dict with `com_offsets`, `engine`, `num_iterations`, and `mask_diameter`.

### Example: quick synthetic test

```python
from alignimg_api import generate_synthetic_data, run_alignment

gt, X = generate_synthetic_data(N=200, H=128, W=128)
final_ref, history, params, meta = run_alignment(X, gt)
```

## Notes

- AlignImg is designed for cryo-EM 2D particle stacks but can be used with any 2D image stacks of the same size.
- CPU parallel execution is used by default; set `n_jobs=1` for single-core behavior.
- GPU acceleration is optional and auto-fallbacks to CPU if CuPy is not installed.
