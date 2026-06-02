# AlignImg

AlignImg is a CPU-based 2D image alignment package for robust MAP-EM
workflows. It provides a small public API, single-process and multicore CPU
backends, warm-start refinement, and an experimental batched local scan option.

Use the package namespace:

```python
import alignimg as ai
```

The supported public API is:

```python
ai.MAPEMConfig
ai.available_backends()
ai.make_mapem_config()
ai.run_alignment()
ai.run_transform()
```

Implementation modules with underscore names are internal and may change.

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

Core installation depends on `numpy`, `scipy`, and `opencv-python-headless`.
`mrcfile`, `matplotlib`, and `pandas` are optional example dependencies.

## Quick Start

```python
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
    config=cfg,
    n_jobs=24,
)

corrected = ai.run_transform(X, params, backend="multicore", n_jobs=24)
```

`X` is an image stack with shape `(N, H, W)`. `initial_ref` is a reference
image with shape `(H, W)`. Output `params` uses the stable convention
`[angle, dy, dx, score]`.

## Warm-Start Refinement

Warm-start refinement reuses pose parameters from a previous call and runs a
local search:

```python
final_ref2, history2, params2, meta2 = ai.run_alignment(
    X,
    final_ref,
    num_iterations=1,
    mask_diameter=80.0,
    backend="multicore",
    config=cfg,
    initial_params=params,
    search_mode="refine",
    n_jobs=24,
    use_shared_memory=True,
)
```

`initial_params` is the only warm-start argument. It accepts `(N, 3)` or
`(N, 4+)` arrays with columns `[angle, dy, dx, score optional]`.

## Backends

Implemented backends:

- `single`: single-process CPU MAP-EM
- `multicore`: process-parallel CPU MAP-EM

`use_shared_memory=True` is an optional multicore mode for large workloads. It
does not change the transform convention, parameter format, or MAP-EM objective.

## Experimental Batched Scan

`use_batched_scan=True` enables an experimental CPU batched local angle scan. It
is off by default and intended for benchmarking or local refinement experiments:

```python
cfg = ai.make_mapem_config(use_batched_scan=True)
```

## Data And Outputs

Large MRC/MRCS/NPY/NPZ data files and generated real-data outputs are local
artifacts. They are ignored by Git and excluded from package builds.

## Testing

Run the pytest suite:

```bash
python -m pytest tests
```

Run the optimization benchmark smoke check:

```bash
python examples/benchmark_cpu_optimization.py --n 64 --n-jobs 2 --refine-iters 1
```

Build a local distribution:

```bash
python -m build
```
