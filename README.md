# AlignImg

AlignImg is a CPU-based 2D image alignment package for robust MAP-EM workflows.
It includes single-process and multicore backends, optional warm-start
refinement, and an experimental batched CPU local scan path.

The public package API is available as:

```python
import alignimg as ai
import alignimg.api as api
```

Legacy root-level imports are still supported while existing scripts migrate:

```python
import alignimg_api as api
import align_utils as au
import align_multicore
import align_batch_cpu
```

## Installation

Install locally in editable mode:

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e .[dev]
```

For real-data examples and plotting:

```bash
python -m pip install -e .[io,plot]
```

Core installation depends only on `numpy`, `scipy`, and `opencv-python`.
`mrcfile`, `matplotlib`, and `pandas` are optional example dependencies.

## Quick Start

```python
import numpy as np
import alignimg as ai

cfg = ai.make_mapem_config(
    phase=3,
    weight_mode="sigmoid",
    lambda_shift=0.01,
    lambda_angle=0.0,
)

final_ref, history, params, meta = ai.run_alignment(
    X,
    initial_ref,
    num_iterations=4,
    mask_diameter=80.0,
    backend="multicore",
    algorithm="mapem",
    config=cfg,
    n_jobs=24,
)

corrected = ai.run_transform(X, params)
```

`X` is an image stack with shape `(N, H, W)`. `initial_ref` is a reference image
with shape `(H, W)`. Output `params` uses the stable convention
`[angle, dy, dx, score]`.

## Cold-Start Alignment

Cold start uses the global search schedule:

```python
cfg = ai.make_mapem_config(
    phase=3,
    weight_mode="sigmoid",
    lambda_shift=0.01,
    lambda_angle=0.0,
)

final_ref, history, params, meta = ai.run_alignment(
    X,
    initial_ref,
    num_iterations=4,
    mask_diameter=80.0,
    backend="multicore",
    algorithm="mapem",
    config=cfg,
    search_mode="global",
    n_jobs=24,
)
```

## Warm-Start Refine

Warm-start refinement reuses parameters from a previous call and runs local
refinement. This path is preserved and remains opt-in through the warm-start
parameters and `search_mode="refine"`.

```python
final_ref2, history2, params2, meta2 = ai.run_alignment(
    X,
    final_ref,
    num_iterations=1,
    mask_diameter=80.0,
    backend="multicore",
    algorithm="mapem",
    config=cfg,
    previous_params=params,
    search_mode="refine",
    n_jobs=24,
    use_shared_memory=True,
)
```

Warm-start inputs may be passed as `previous_params`, `initial_params`, or
`warm_start_params`. Use only one of these in a single call.

## Shared Memory

`use_shared_memory=True` is an optional multicore MAP-EM mode for large
workloads. It shares input arrays between worker processes to reduce copy
overhead. It defaults to `False` and does not change the transform convention,
parameter format, or algorithm objective.

## Experimental Batched Scan

`use_batched_scan=True` enables an experimental CPU batched local angle scan.
It is not the default and is intended for explicit benchmarking or local
refinement experiments:

```python
cfg = ai.make_mapem_config(use_batched_scan=True)
```

## Current Limitations

- The implemented backends are CPU-only: `single` and `multicore`.
- The `gpu` backend is reserved for future implementation.
- No PyTorch or CuPy dependency is included.
- Large MRC/MRCS/NPY/NPZ data files are not part of the package.
- Example scripts may require optional `io` and `plot` dependencies.

## Testing

Run the pytest suite:

```bash
python -m pytest tests
```

The script-style tests also support direct execution:

```bash
python tests/test_api_smoke.py
python tests/test_io_contract.py
python tests/test_batch_scan_cpu.py
```

Optional benchmark smoke check:

```bash
python examples/benchmark_cpu_optimization.py --n 64 --n-jobs 2 --refine-iters 1
```

Build a local distribution:

```bash
python -m build
```
