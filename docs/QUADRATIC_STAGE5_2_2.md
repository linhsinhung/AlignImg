# AlignImg 2.2 Stage 5: user-scale validation and release candidate

Stage 5 starts from the accepted `2.2.0.dev4` Stage 4 engine. The numerical
solver is unchanged. Candidate `2.2.0.dev5` corrects two workflow-composition
issues before the final release gate:

- `tools/align_known_reference.py` now uses the validated refine preset without
  replacing its 1-degree, plus/minus 7-degree, plus/minus 3-pixel search window
  with the older tool-local grid.
- AlignImg Workbench 0.6 displays an independent refinement selector. Continuous
  quadratic is the default; adaptive posterior remains an explicit 2.1
  compatibility choice. Global search remains independently set to proposal.

## Frozen 3,050-particle A/B

`tools/quadratic_stage5_validation.py` loads `data/local/test_align.mrcs` and
`data/local/mu_aligned_mean.mrc`, computes one shared global initializer, and
runs two refinement iterations from the exact same poses and reference. The
only search differences are the frozen adaptive grid and the accepted quadratic
profile. Scoring, M-step, temperature, robust weighting, raw-average output,
backend, batch, and memory policy are identical.

The gate requires:

- quadratic refine time no more than 105% of adaptive;
- final raw-average correlation no more than 0.005 below adaptive relative to
  the supplied reference;
- soft-reference coherence with that strategy's final raw average no more than
  0.005 below adaptive; soft-to-supplied correlation remains diagnostic because
  posterior entropy changes how strongly the soft M-step smooths the model;
- finite normalized K=1 output using the raw MAP-pose estimator;
- the requested backend, no GPU fallback, and the correct native CUDA or CuPy
  quadratic peak implementation.

The runner records input and artifact hashes, per-iteration timing and NCC,
reference trajectories, pose deltas, memory plans, map chunks, both output
estimators, and cross-strategy correlations.

```bash
python tools/quadratic_stage5_validation.py \
  --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/quadratic-refine/\
dev5-cuda-local-n3050-ab.json
```

Run unprofiled for the release timing gate. Detailed kernel profiling is not
mixed into the formal wall-time comparison.

## Backend regression matrix

The candidate must also pass the complete quick suite on CPU, CuPy, and native
CUDA. These protect RF/global outputs, quadratic synthetic accuracy, fixed MRA,
class feedback, final raw-average reconstruction, Fourier scoring/M-step, and
backend transform conventions.

```bash
python tools/server_validation.py --suite quick --backend cuda \
  --batch-size 512 \
  --output validation-results/performance/quadratic-refine/dev5-cuda-quick.json

python tools/server_validation.py --suite quick --backend cupy \
  --batch-size 512 \
  --output validation-results/performance/quadratic-refine/dev5-cupy-quick.json

python tools/server_validation.py --suite quick --backend cpu \
  --batch-size 512 \
  --output validation-results/performance/quadratic-refine/dev5-cpu-quick.json
```

Passing dev5 is a release-candidate gate, not the final release itself. Only
after inspecting these four reports will the package versions move to `2.2.0`,
the final distributions be rebuilt, and the release source/results frozen.

## Accepted result

The CUDA, CuPy, and CPU quick reports each passed all 21 cases with AlignImg
`2.2.0.dev5`. The 3,050-particle CUDA run completed two quadratic refinement
iterations in 65.159 seconds versus 459.070 seconds for the frozen adaptive
path, a 7.045-fold speedup. Final expected Fourier NCC increased from 0.22346
to 0.26110. The final raw-average correlation to the supplied reference changed
from 0.95935 to 0.95437, a -0.00498 boundary pass under the unchanged 0.005
tolerance.

The initial report also gated the stored soft M-step reference directly against
the supplied reference. Review showed that this rewarded adaptive posterior
blur: adaptive pose entropy was 6.42 with 0.0072 median MAP mass, while
quadratic pose entropy was 2.59 with 0.1104 median MAP mass. The release gate
therefore retains soft-to-supplied correlation as an observation, gates the
external reference quality on the final raw average, and checks each soft
reference against its own raw average. Quadratic soft-to-raw correlation was
0.99546 versus 0.97768 for adaptive. No numerical threshold was loosened and
the engine was not changed during this review.

The reviewed gate and original artifact hashes are recorded in
`dev5-cuda-local-n3050-ab.gate-review.json`. These results authorize the formal
`2.2.0` source and distribution freeze described in `RELEASE_FREEZE_2_2.md`.
