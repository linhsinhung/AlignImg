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
    batch_size=512,
    profile_gpu=True
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
  - `batch_size`: GPU batch size (recommended starting point on Linux servers: `batch_size=4096`, then tune with `profile_gpu=True`).
  - `profile_gpu`: if `True` in GPU mode, reports CUDA event timing summary in `meta["gpu_profile"]`.

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


## Minimal GPU benchmark

Use this command to compare throughput before/after changes (adjust to your GPU memory):

```bash
python - <<'PY'
import time
import numpy as np
from alignimg_api import run_alignment

rng = np.random.default_rng(0)
N, H, W = 1024, 128, 128
X = rng.normal(size=(N, H, W)).astype(np.float32)
init_ref = X.mean(axis=0).astype(np.float32)

t0 = time.perf_counter()
_, _, _, meta = run_alignment(
    X,
    init_ref,
    num_iterations=2,
    use_gpu=True,
    batch_size=256,
    profile_gpu=True,
)
print(f"elapsed_s={time.perf_counter() - t0:.3f}")
print(meta.get("gpu_profile", {}))
PY
```
