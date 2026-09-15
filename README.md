# AlignImg

AlignImg 2.x is a workflow-oriented 2-D cryo-EM alignment library. A common
top-L soft-inference engine supports reference-free multi-view alignment,
global single/multi-reference alignment, and robust pose refinement.

The default 2.x pipeline keeps shortlisted rotation, translation, Fourier NCC,
and the soft reference update in Fourier space. Each particle is transformed to
Fourier space once per iteration; references return to real space for output and
diagnostics. The validated raster scorer and spatial M-step remain available as
explicit regression modes.

The new engine accepts even-sized square, externally prepared images. It does not perform
particle picking, CTF estimation/correction, denoising, biological
classification, or 3-D reconstruction.

## Installation

```bash
python -m pip install -e .
python -m pip install -e .[dev]
```

## Reference-free alignment

```python
import alignimg as ai

config = ai.AlignmentConfig(max_iterations=10, top_l=8)
result = ai.reference_free_align(
    particles,
    n_components=20,
    config=config,
)

references = result.references
poses = result.poses
assignments = result.reference_assignments
responsibilities = result.responsibilities
```

`n_components` is required. Assignments identify alignment/reference
components and are not claimed to be final biological classes.

## Global alignment and refinement

```python
global_result = ai.align_to_references(
    particles,
    references,
    class_priors=optional_n_by_k_priors,
)

refined = ai.refine_alignment(
    particles,
    global_result.references,
    global_result.poses,
    class_priors=optional_n_by_k_priors,
)

aligned_raw = ai.transform_images(raw_particles, refined.poses)
```

This last pattern allows poses estimated from externally denoised images to be
applied to the corresponding raw particles.

For a complete MRC-stack workflow with one known reference, use:

```bash
python tools/align_known_reference.py \
  --particles data/local/test_align.mrcs \
  --reference data/local/mu_aligned_mean.mrc \
  --output-directory alignment-results/test-align \
  --backend cuda \
  --batch-size 512 \
  --global-iterations 1 \
  --refine-iterations 3 \
  --angle-samples 256 \
  --proposal-angles 8 \
  --translation-range 8 \
  --apply-final-pose-to-raw
```

This performs K=1 global alignment followed by optional prior-centered robust
refinement. It writes `class_average.mrc`, `result.npz`, and `report.json`.
The NPZ contains the final MAP poses, top-L pose candidates and posteriors,
uncertainty, inlier weights, global-stage poses, and reference history. The
class average is the soft Fourier M-step estimate by default, so it need not be
exactly equal to an arithmetic average transformed by only the MAP pose of each
image. `--apply-final-pose-to-raw` instead reconstructs only the final output
average from the raw particles using each particle's final MAP pose, hard class
assignment, and inlier weight. The iterative soft references remain available
in `result.npz` and are not replaced for subsequent refinement.

`reference_free_align()` and `align_to_references()` default to the fast polar
`proposal` search. When its config is omitted, `refine_alignment()` defaults to
the continuous prior-centered `quadratic_refine` search:

```python
refine_config = ai.AlignmentConfig.preset("refine")
# search_strategy="quadratic_refine"
# coarse_angle_step=1.0, local_angle_range=7.0
# local_shift_range=3.0
refined = ai.refine_alignment(
    particles, references, initial_poses, config=refine_config
)
```

Quadratic refinement screens a bounded angle profile, obtains the integer
translation surface from Fourier correlation-map IFFTs, and fits continuous
x/y shifts and angular modes with independently validated one-dimensional
parabolas. The full screened profile, rather than only the fitted MAP pose,
continues into responsibilities and the soft Fourier M-step.

The frozen 2.1 adaptive search remains available explicitly for comparison:

```python
adaptive_config = ai.AlignmentConfig(
    search_strategy="adaptive_posterior",
    coarse_angle_step=6.0,
    local_angle_range=15.0,
    local_shift_range=3.0,
    adaptive_fraction=0.999,
    oversampling_order=1,
    max_adaptive_cells=None,
)
```

Adaptive refinement selects coarse cells by cumulative pseudo-posterior mass,
then fine-samples only those cells. The current pseudo-posterior uses Fourier
NCC, temperature, and pose/class priors. This is RELION-like search behavior,
not RELION's full noise-spectrum/CTF likelihood or regularized reconstruction.
The public `CandidateSet` remains a compact top-L shortlist, while class
responsibilities, reference updates, and uncertainty use the complete internal
fine posterior. Consequently `top_l` controls output size, not M-step accuracy.
When adaptive rescue is enabled, a local-boundary hit or an optional normalized-entropy
or low-MAP trigger schedules a bounded subset per current component for one
global proposal review. The proposal replaces the local result only when its
Fourier-NCC score improves by `rescue_min_score_improvement`; rejected particles
keep their local result. The final iteration is always a local confirmation, so
rescue requires at least three iterations. This is intentionally a one-pass
safety check rather than a repeated outlier scheduler.

## Classification feedback refinement

Use external class assignments and class averages without coupling AlignImg to
the classification algorithm:

```python
fixed_priors = ai.make_class_priors(
    assignments=class_labels,
    n_components=len(class_averages),
    trust=1.0,
)
fixed = ai.refine_alignment(
    particles, class_averages, initial_poses, class_priors=fixed_priors
)

corrective_priors = ai.make_class_priors(
    responsibilities=classification_responsibilities,
    trust=0.9,
)
corrective = ai.refine_alignment(
    particles, class_averages, initial_poses, class_priors=corrective_priors
)
```

The first workflow cannot change class membership. For corrective refinement,
soft classification responsibilities are recommended because they preserve
which particles were originally ambiguous. Hard assignments remain supported
when no probabilities are available, but `trust=0.9` with one-hot feedback is
deliberately conservative and may produce no reassignment. Such assignments
remain alignment components, not validated biological classes.

## Transform convention

New `PoseSet` transforms map input particles into reference coordinates:

1. periodic left-right mirror about the integer x origin (when enabled)
2. counter-clockwise rotation about `(size // 2, size // 2)`
3. `(shift_y_px, shift_x_px)` translation
4. periodic/wrap boundary handling

The integer origin matches RELION and the DC index of an `fftshift`-centered
even DFT. Alignment workflows reject odd image sizes. Results record
`center_convention="integer-origin-floor-N-over-2"` in their metadata.

Saved 1.4 `PoseSet` values used the former geometric center. Convert them
explicitly before applying them with 2.x:

```python
poses_1_5 = ai.convert_v1_4_poses_to_integer_center(poses_1_4, image_size=128)
```

The adapter preserves the continuous affine mapping, including mirror poses;
it cannot reproduce the exact pixels produced by the former two-pass OpenCV
interpolation.

The explicit `poses_from_legacy_params()` and `poses_to_legacy_params()`
adapters remain available for historical pose arrays. The AlignImg 0.2
`run_alignment`/`run_transform` engine is not part of the 2.x package.

## Optional NVIDIA GPU backend

The separate distribution keeps CUDA dependencies out of the CPU package:

```bash
python -m pip install -e './packages/alignimg-gpu[cuda12]'
# use [cuda13] on CUDA 13 hosts
```

Then pass `backend="cuda"` (compiled native path), `backend="cupy"` (portable
fallback), or `backend="auto"` (CUDA → CuPy → CPU) to a 2.x workflow. Explicit
GPU requests never silently fall back. See `packages/alignimg-gpu/README.md` for the execution split
and installation notes.

With `apply_final_pose_to_raw=True`, CUDA/CuPy applies the final poses in
VRAM-bounded batches and accumulates FP64 class sums on the GPU. Only K class
averages are downloaded; the input in-memory NumPy particles still require one
H2D upload per batch. AlignImg does not reopen or stream a raw stack from disk.

## Testing

The final pre-2.0 scientific baseline is recorded in
[`docs/FINAL_RF_MRA_BASELINE_1_10.md`](docs/FINAL_RF_MRA_BASELINE_1_10.md).
The earlier raster baseline remains in
[`docs/BASELINE_1_7_1.md`](docs/BASELINE_1_7_1.md) for regression comparisons.
The AlignImg 2.1 performance release-candidate procedure is recorded in
[`docs/PERFORMANCE_STAGE_7_2_1.zh-TW.md`](docs/PERFORMANCE_STAGE_7_2_1.zh-TW.md).

```bash
python -m pytest
python -m build
```

For staged validation on a Linux/NVIDIA server, write an incremental JSON
report with:

```bash
python tools/server_validation.py --suite standard --backend cuda \
    --output alignimg-validation-standard.json
```

Use `--suite quick` for the first smoke run, `--suite tuning` for controlled
proposal/top-L/temperature/batch ablations, and `--suite full` for the
10,000-particle K=20/K=50 scalability run. A failed case records its traceback
and GPU-memory snapshot without discarding earlier results. An interrupted run
can continue with the same output path plus `--resume`.

The quick and standard suites include `adaptive_refine`, which checks synthetic
pose improvement, uncertainty output, adaptive diagnostics, CPU/backend parity,
and GPU memory-plan records. It can be run alone with
`--only adaptive_refine` during the first server smoke test.

AlignImg 2.x defaults to the Fourier-native candidate scorer validated in 1.8.
Select `AlignmentConfig(candidate_scoring="raster")` only for a
1.7.1-compatible regression run.
Its mathematical contract and CUDA/CuPy validation results are documented in
[`docs/FOURIER_NATIVE_1_8.md`](docs/FOURIER_NATIVE_1_8.md).

AlignImg 2.x also defaults to the Fourier-domain soft M-step validated in 1.9;
`reference_update="spatial"` remains the frozen comparison path. See
[`docs/FOURIER_MSTEP_1_9.md`](docs/FOURIER_MSTEP_1_9.md).

Empirical radial whitening remains opt-in:
`AlignmentConfig(score_model="whitened_fourier_ncc")`. The established
`score_model="fourier_ncc"` remains the default. See
[`docs/WHITENED_SCORING_1_10.md`](docs/WHITENED_SCORING_1_10.md).

### RE2DC 70S real-data validation

Build the reproducible 1000-particle, 128-pixel phase-corrected stack once:

```bash
python tools/prepare_re2dc_70s_benchmark.py
```

The command writes an MRCS stack, a matching subset STAR, and a JSON manifest
under `data/re2dc_70s_testdata/prepared/`. The original pixel size is 2.82
angstrom/pixel. CTF phase flipping is performed before a 130-to-128 centered
Fourier crop; the manifest records the resulting sampling. Do not apply CTF
correction to this prepared stack again.

Run two-seed reference-free validation with ten alignment/view components:

```bash
python tools/re2dc_70s_rf_validation.py --backend cuda \
    --output validation-results/re2dc-70s-rf.json
```

The real-data defaults use 10 annealing iterations followed by 5 iterations at
the final temperature, with a batch size of 512. Reference history from every
iteration is saved in each seed's result NPZ so convergence can be assessed
retrospectively; 10--20 or more total iterations may be appropriate depending
on the dataset.

This reports bootstrap diversity, component occupancy, FRC, convergence, and
cross-seed stability. It intentionally does not interpret alignment components
as biological classes or attempt to repair class-attraction/degeneracy.

After the 1000-particle workflow is stable, run the formal 5000-particle,
three-seed RF baseline with 10 annealing rounds followed by 10 hold rounds:

```bash
python tools/re2dc_70s_rf_validation.py \
    --stack data/re2dc_70s_testdata/prepared/re2dc_70s_n5000_s128.mrcs \
    --backend cuda \
    --components 10 \
    --iterations 20 \
    --anneal-iterations 10 \
    --seeds 0 1 2 \
    --angle-samples 128 \
    --batch-size 512 \
    --memory-fraction 0.8 \
    --output validation-results/re2dc-70s-rf-n5000-k10-i20.json
```

Each seed writes references, result arrays, and a `*.seed-N.report.json` file
that can be loaded directly in AlignImg Workbench. Analyze this baseline before
choosing temperature, top-L, bootstrap, or K ablations; this avoids committing
the full dataset to a speculative parameter grid.

After RF has produced references and a result NPZ, compare fixed-class and
corrective feedback refinement with:

```bash
python tools/re2dc_70s_feedback_validation.py \
    --backend cuda \
    --references validation-results/re2dc-70s-rf-iter15.seed-1.references.mrcs \
    --rf-result validation-results/re2dc-70s-rf-iter15.seed-1.result.npz \
    --corrective-prior-source responsibilities \
    --output validation-results/re2dc-70s-feedback.json
```

The report records assignment transitions, pose changes, reference
correlations, stable FRC, throughput, and GPU memory plans. Both modes also
write final references and result arrays. RF is provisionally usable for this
downstream development; the formal 5000-particle baseline above is the entry
point for scientific tuning.

### RELION-defined homogeneous pose benchmark

Use the RELION 2-D classification result to test alignment accuracy without
asking AlignImg to rediscover the full heterogeneous partition. The benchmark
uses `_rlnClassNumber` (not `_rlnGroupNumber`, which records acquisition/image
groups), selects high-confidence particles from RELION classes 4, 7, and 9,
and creates two disjoint sets per class:

- 64 particles build an external RELION-pose class reference;
- 128 different particles are retained for alignment evaluation.

Build the deterministic 384-particle benchmark with:

```bash
python tools/prepare_re2dc_70s_pose_benchmark.py
```

The conversion uses `angle_deg = -rlnAnglePsi`; origins in angstrom are first
converted with the prepared-stack pixel size and then expressed as AlignImg's
post-rotation shift. The output manifest records the integer-center convention,
source indices, confidence values, converted poses, and all split artifacts.

First run the classification-independent alignment checks:

```bash
python tools/re2dc_70s_pose_benchmark.py \
    --backend cuda \
    --batch-size 512 \
    --only oracle known_reference fixed_mra \
    --output validation-results/re2dc-70s-pose-known.json
```

`known_reference` performs three independent K=1 global alignments.
`fixed_mra` processes all three references together but uses one-hot class
priors, so it tests the multi-reference pose path without allowing a class
change. Deterministic modes run twice by default and report assignment,
pose, responsibility, and reference reproducibility.

After the proposal initializer is established, run the adaptive-posterior
refinement comparison:

```bash
python tools/re2dc_70s_pose_benchmark.py \
    --backend cuda \
    --batch-size 512 \
    --only adaptive_known_reference adaptive_fixed_mra \
    --global-iterations 3 \
    --mra-iterations 3 \
    --adaptive-iterations 3 \
    --coarse-angle-step 6 \
    --coarse-shift-step 1 \
    --local-angle-range 15 \
    --local-shift-range 3 \
    --adaptive-fraction 0.999 \
    --oversampling-order 1 \
    --max-adaptive-cells 32 \
    --output validation-results/re2dc-70s-pose-adaptive.json
```

These modes generate their own proposal initializer and then refine it; RELION
poses are used only for evaluation and never as AlignImg input. The report
nests initializer and final pose errors, records pose changes, full-posterior
entropy/MAP probability, posterior-mass retention, boundary/cap/rescue counts,
runtime, and GPU memory plans. `adaptive_fixed_mra` keeps one-hot class priors,
so it must be numerically equivalent to the corresponding independent K=1
refinements apart from execution batching.

Then test homogeneous K=1 RF:

```bash
python tools/re2dc_70s_pose_benchmark.py \
    --backend cuda \
    --batch-size 512 \
    --only homogeneous_rf \
    --rf-iterations 15 \
    --rf-seeds 0 \
    --output validation-results/re2dc-70s-pose-rf.json
```

For K=1, the spectral bootstrap's medoid update has one final component, so
different seeds produce identical alignment poses, references,
responsibilities, and histories. The seed still changes the diagnostic
halfset membership and therefore FRC. Additional seeds should only be used to
measure that halfset-split sensitivity, not alignment robustness.

Pose errors are evaluated after fitting one global rotation/translation gauge
per RELION class, because reference-free averages have no absolute in-plane
frame. `open_mra` is available as a separate diagnostic, but its assignment
accuracy is not a pose benchmark and is not interpreted as biological
classification quality. RELION classes and poses are an external practical
reference, not literal ground truth, so this runner records measurements and
artifacts without imposing a scientific pass/fail threshold.

AlignImg is distributed under GPL-3.0-or-later. See `LICENSE` and
`THIRD_PARTY_NOTICES.md`.

## Development GUI

An optional PyQt6 workbench is maintained as the independent
`packages/alignimg-gui` package. Its two pages compose either Reference-Free
global alignment or reference-based/MRA global alignment with an optional local
refinement stage. It can also resume refinement from a saved result. Each stage
runs in an isolated process and records its own artifacts, while the top-level
result preserves a combined reference history for inspecting class averages,
convergence, occupancy, Fourier execution mode, rescue diagnostics, GPU memory
plans, and logs. Install it with
`python -m pip install -e packages/alignimg-gui` and start it with
`alignimg-gui`. CTF correction and other data preparation remain outside the
GUI and alignment core.
