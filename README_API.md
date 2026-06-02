# AlignImg Public API

This document describes the current public API exposed by the `alignimg`
package. Prefer the package imports:

```python
import alignimg as ai
import alignimg.api as api
```

Compatibility wrappers remain at the project root for older scripts:

```python
import alignimg_api as api
import align_utils as au
import align_multicore
import align_batch_cpu
```

The current production package files are:

- `src/alignimg/utils.py`: CPU primitives and single-process CPU backends
- `src/alignimg/multicore.py`: multi-core CPU MAP-EM backend
- `src/alignimg/api.py`: public interface and backend dispatcher
- `src/alignimg/batch_cpu.py`: experimental batched CPU local scan helper

Examples live in `examples/`. The standalone prototype lives in `prototypes/`
and is not part of the public package API.

## Data Shapes

### Image Stack `X`

Input particle/image stack:

```text
(N, H, W)
```

`float32` is preferred.

### Initial Reference `initial_ref`

Input reference image:

```text
(H, W)
```

`float32` is preferred.

### Alignment Parameters `params`

Canonical output alignment parameters:

```text
(N, 4)
```

Columns:

```text
[angle, dy, dx, score]
```

For MAP-EM output, the fourth column is the posterior score:

```text
[angle, dy, dx, posterior_score]
```

### Warm-Start Parameters

The warm-start arguments are aliases for the same input contract:

- `previous_params`
- `initial_params`
- `warm_start_params`

Accepted shapes:

```text
(N, 3)
(N, 4)
```

Columns:

```text
[angle, dy, dx, score optional]
```

Only one warm-start argument should be provided in a single call.

## `ai.available_backends()`

Return backend metadata as a dictionary.

```python
backends = ai.available_backends()
```

Currently implemented production backends:

- `single`
- `multicore`

The `gpu` backend is reserved for future implementation.

## `ai.make_mapem_config()`

Create a MAP-EM configuration object with the current package defaults.

```python
cfg = ai.make_mapem_config()
```

Common options include:

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

`use_batched_scan=True` is experimental and defaults to `False`. It only
affects local MAP-EM angle scans when explicitly enabled.

## `ai.run_alignment()`

Run alignment through the public backend dispatcher.

```python
final_ref, history, params, meta = ai.run_alignment(
    X,
    initial_ref,
    num_iterations=4,
    backend="multicore",
    algorithm="mapem",
    config=cfg,
)
```

Inputs:

- `X`: `(N, H, W)`, `float32` preferred
- `initial_ref`: `(H, W)`, `float32` preferred
- `num_iterations`: number of MAP-EM iterations
- `backend`: `"single"` or `"multicore"`
- `algorithm`: `"mapem"` for the recommended robust MAP-EM path
- `config`: optional `MAPEMConfig` from `ai.make_mapem_config()`
- `use_shared_memory`: optional `False`/`True` flag for the multicore MAP-EM backend

Returns:

- `final_ref`: `(H, W)`
- `history`: list of `(H, W)` references, including the initial masked reference
- `params`: `(N, 4)`, columns `[angle, dy, dx, posterior_score]`
- `meta`: dictionary containing at least:
  - `last_weights`: `(N,)`
  - `last_image_scores`: `(N,)`
  - `last_posterior_scores`: `(N,)`

The metadata also includes backend, algorithm, search mode, warm-start status,
configuration, timing, and per-iteration summaries.

`use_shared_memory=True` is an opt-in multicore MAP-EM mode that shares the
input stack and per-iteration matching reference between worker processes. It
only affects internal multicore data transfer and does not change the alignment
objective, transform convention, output format, or returned values.

## `ai.run_transform()`

Apply final transform parameters to an image stack.

```python
corrected = ai.run_transform(
    X,
    params,
    backend="single",
)
```

Inputs:

- `X`: `(N, H, W)`, `float32` preferred
- `params`: `(N, 4)` or compatible parameter array using columns `[angle, dy, dx, score]`

Returns:

- `corrected`: `(N, H, W)`

## Recommended Usage

### Cold Start

Use a global search over several iterations:

```python
cfg = ai.make_mapem_config()

final_ref, history, params, meta = ai.run_alignment(
    X,
    initial_ref,
    backend="multicore",
    algorithm="mapem",
    search_mode="global",
    num_iterations=4,
    config=cfg,
)
```

### Warm-Start / Repeated Call

Reuse parameters from a previous call and run local refinement:

```python
cfg = ai.make_mapem_config()

final_ref, history, params, meta = ai.run_alignment(
    X,
    initial_ref,
    backend="multicore",
    algorithm="mapem",
    previous_params=params_prev,
    search_mode="refine",
    num_iterations=1,
    config=cfg,
)
```

Warm-start refinement accepts `previous_params`, `initial_params`, or
`warm_start_params`; pass only one of them per call.
