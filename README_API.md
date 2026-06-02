# AlignImg Public API

This document describes the public API for AlignImg 0.2.0. Prefer the package
namespace:

```python
import alignimg as ai
```

Supported public names:

```python
ai.__version__
ai.MAPEMConfig
ai.available_backends()
ai.make_mapem_config()
ai.run_alignment()
ai.run_transform()
```

Underscore-prefixed modules are internal implementation details.

## Data Shapes

Image stack `X`:

```text
(N, H, W)
```

Initial reference `initial_ref`:

```text
(H, W)
```

Alignment parameters `params`:

```text
(N, 4)
[angle, dy, dx, score]
```

For MAP-EM output, `score` is the posterior score.

Warm-start `initial_params` accepts:

```text
(N, 3)
(N, 4+)
```

Columns are `[angle, dy, dx, score optional]`.

## `ai.available_backends()`

Return implemented backend metadata.

```python
backends = ai.available_backends()
```

Implemented backends:

- `single`
- `multicore`

## `ai.make_mapem_config()`

Create a MAP-EM configuration object.

```python
cfg = ai.make_mapem_config(
    phase=3,
    weight_mode="sigmoid",
    keep_fraction=0.75,
    lambda_shift=0.01,
    sigma_shift_y=8.0,
    sigma_shift_x=8.0,
    lambda_angle=0.0,
)
```

`phase` controls the MAP-EM variant:

- `1`: hard-MAP pose inference with unweighted template update
- `2`: robust MAP-EM with inlier weights
- `3`: robust MAP-EM with pose priors

`use_batched_scan=True` is experimental and off by default. It only affects
local MAP-EM angle scans.

## `ai.run_alignment()`

Run MAP-EM alignment through a CPU backend.

```python
final_ref, history, params, meta = ai.run_alignment(
    X,
    initial_ref,
    num_iterations=4,
    mask_diameter=80.0,
    backend="multicore",
    config=cfg,
    initial_params=None,
    search_mode="global",
    n_jobs=24,
)
```

Inputs:

- `X`: `(N, H, W)`, `float32` preferred
- `initial_ref`: `(H, W)`, `float32` preferred
- `num_iterations`: number of MAP-EM iterations
- `backend`: `"single"` or `"multicore"`
- `config`: optional `MAPEMConfig`; defaults to `ai.make_mapem_config()`
- `initial_params`: optional warm-start poses
- `search_mode`: `"auto"`, `"global"`, or `"refine"`
- `use_shared_memory`: optional multicore data-transfer mode

Returns:

- `final_ref`: `(H, W)`
- `history`: list of `(H, W)` references, including the initial masked reference
- `params`: `(N, 4)`, columns `[angle, dy, dx, posterior_score]`
- `meta`: backend, config, search mode, warm-start status, timing, scores, weights

`search_mode="refine"` requires `initial_params`.

## `ai.run_transform()`

Apply final transform parameters to an image stack.

```python
corrected = ai.run_transform(
    X,
    params,
    backend="multicore",
    n_jobs=24,
)
```

Inputs:

- `X`: `(N, H, W)`, `float32` preferred
- `params`: `(N, 4)` or compatible array using columns `[angle, dy, dx, score]`
- `backend`: `"single"` or `"multicore"`

Returns:

- `corrected`: `(N, H, W)`

## Real-Data Examples

`examples/test_realdata.py` is a Spyder/server-friendly script for real-data
experiments. It writes CSV, PNG, and NPY outputs to `examples/realdata_Multi/`.
Those generated outputs, along with large MRC/MRCS/NPY/NPZ data files, are local
artifacts and are excluded from package builds.
