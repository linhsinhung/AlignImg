# AlignImg 2.x public API

The supported 2.x workflows are:

- `reference_free_align(images, *, n_components, config, backend)`
- `align_to_references(images, references, *, class_priors, config, backend)`
- `refine_alignment(images, references, initial_poses, *, class_priors, config, backend)`
- `transform_images(images, poses, *, backend)`
- `make_class_priors(*, assignments, responsibilities, n_components, trust)`
- `convert_v1_4_poses_to_integer_center(poses, image_size)`

The public contracts are `AlignmentConfig`, `PoseSet`, `CandidateSet`, and
`AlignmentResult`. See their Python docstrings for field definitions.
`proposal_angles_per_reference` independently controls the polar angular
shortlist; `None` preserves the automatic policy.
`temperature_anneal_iterations` optionally ends annealing before
`max_iterations`; remaining iterations hold `temperature_end`.
`candidate_scoring="fourier"` is the 2.x default and performs shortlisted rotation, mirror,
translation, and NCC scoring directly from each particle's cached DFT.
`candidate_scoring="raster"` preserves the frozen 1.7.1 comparison path.
`score_model="fourier_ncc"` preserves uniform weighting inside the configured
frequency band. `score_model="whitened_fourier_ncc"` instead applies inverse
radial empirical particle-power weights to translation search and NCC scoring.
It is opt-in and does not alter the polar proposal or reference update.
`reference_update="fourier"` is the 2.x default: it accumulates aligned particle
DFTs and performs
one output IFFT per reference per iteration. Half-set diagnostics require their
own two reference IFFTs. `reference_update="spatial"` preserves the frozen 1.8
comparison path.
`apply_final_pose_to_raw=False` keeps the final soft M-step reference as
`AlignmentResult.class_averages`. When enabled, inference still uses and
returns the same soft `AlignmentResult.references`, but `class_averages` is
reconstructed once after the final iteration from raw particles using their
final MAP poses, hard class assignments, and inlier weights. Reconstruction
uses bounded batches, the selected transform backend, and the configured
circular mask without an additional low-pass filter. On CUDA and CuPy backends,
the transformed batches are accumulated in FP64 on the device and only the K
final averages are downloaded. `images` is still the caller-provided in-memory
NumPy stack; this option does not read an original stack from disk.

## Pose-search strategies

`search_strategy="proposal"` is the global, high-throughput path. It uses
Fourier-polar correlation to propose multiple angular peaks, performs Cartesian
Fourier-NCC reranking, and retains `top_l` normalized hypotheses. With
`candidate_scoring="fourier"`, reranking does not rasterize or FFT each
shortlisted rotation. Reference-free
and global workflows select this strategy when their config is omitted.

`search_strategy="adaptive_posterior"` is the prior-centered refinement path.
For every particle it:

1. constructs a regular angle/translation grid around `initial_poses`;
2. scores all coarse cells and adds pose and class priors;
3. keeps the smallest stable-ranked cell set covering `adaptive_fraction`;
4. expands only those cells by `2**oversampling_order`, including their centers;
5. re-scores and normalizes the fine hypotheses before the shared soft M-step.

`coarse_angle_step` and `coarse_shift_step` set coarse resolution;
`local_angle_range` and `local_shift_range` bound the prior-centered window.
`max_adaptive_cells` is an optional safety cap, not the primary selection rule.
When `rescue_uncertain_particles=True`, a MAP cell on a local boundary schedules
that particle for one proposal-search iteration. Optional
`rescue_normalized_entropy_threshold` and `rescue_map_posterior_threshold`
schedule particles with diffuse fine-pose posteriors even when their MAP cell is
not on a boundary. `rescue_max_fraction` deterministically limits the fraction
scheduled within each current component. A global proposal is accepted only if
its raw Fourier-NCC exceeds the local result by
`rescue_min_score_improvement`; otherwise the local result is retained.
Rescued particles are excluded from immediate rescheduling, and the final
iteration is reserved for local confirmation. Rescue therefore needs at least
three iterations.

`search_strategy="quadratic_refine"` is the AlignImg 2.2 continuous local
refinement path. It screens `coarse_angle_step` samples inside
`local_angle_range`, obtains the bounded integer translation surface from one
weighted Fourier correlation-map IFFT per angle, and independently fits x and y
with three-point parabolas. It then fits each interior angular local maximum.
Every fitted translation and angle is accepted only when its curvature and
half-step bounds are valid and an exact Fourier-NCC plus pose-prior rescore
improves the discrete pose. A fitted angular mode replaces its grid sample,
and the complete screened angle profile across active references and mirror
states forms the internal soft posterior used by responsibilities and the
Fourier M-step. The public `CandidateSet` remains top-L normalized.
`quadratic_refine` requires `initial_poses`, Fourier candidate scoring, and the
Fourier M-step, and intentionally does not run the adaptive rescue scheduler.

The `refine` preset uses `quadratic_refine` after the CPU, CuPy, and
native-CUDA K=1 precision, conformance, and resolution-aware RELION gates
passed. The frozen 2.1 path remains available by explicitly setting
`search_strategy="adaptive_posterior"` for A/B comparison or compatibility.
A caller passing a hand-built config controls the strategy explicitly; AlignImg
does not silently infer it from the input. Both local strategies require
`initial_poses`.

The Fourier candidate scorer and Fourier M-step passed their CUDA/CuPy workflow
and RELION-defined pose benchmarks before becoming the 2.x defaults. Both
selected modes are recorded in result metadata. GPU
Fourier mode caches particle DFTs in
VRAM only when they fit within `memory_fraction` while preserving the configured
reserve; otherwise it streams bounded batches from host memory.

The current score is Fourier NCC divided by temperature, combined with pose and
class log-priors. The optional whitened model estimates total radial empirical
power from the masked normalized particle stack; it does not separate signal
from noise. This remains a **RELION-like adaptive posterior search**, not the
full RELION likelihood: it does not estimate a per-particle noise spectrum,
include CTF likelihood, or perform Fourier-regularized reconstruction. Inputs
are expected to have been CTF-corrected externally.

`AlignmentConfig.preset(name)` provides explicit `global_balanced`,
`global_accurate`, `reference_free`, and `refine` configurations. Workflows use
their matching preset only when `config` is omitted.

## Execution profiling (2.1 development)

`AlignmentConfig(profile_execution=True)` adds hierarchical stage timings,
explicit transfer counters, complete iteration wall times, and GPU memory
accounting to `result.metadata["performance"]`. The default is `False`; this
flag does not change default refinement selection or numerical outputs.
Legacy timing fields retain their original meanings. Inclusive stage times
overlap, CUDA event spans are not GPU utilization, and sampled device memory
peaks are not a hard-limit guarantee. Measure throughput in separate unprofiled
runs. See [stage-0 validation](PERFORMANCE_STAGE_0_2_1.zh-TW.md).

## Shapes

- images: `(N, H, H)`
- references: `(K, H, H)` or `(H, H)`
- class priors/responsibilities: `(N, K)`
- candidate fields: `(N, L)`
- assignment and inlier weights: `(N,)`

`AlignmentResult.candidates` contains the retained top-L hypotheses and
top-L-normalized weights; `poses` is their MAP projection. Adaptive inference
keeps a separate internal ragged fine posterior: responsibilities, reference
M-step, `pose_entropy`, and `map_posterior` use all fine hypotheses, so changing
`top_l` only changes the public shortlist rather than the scientific update.
`AlignmentResult.references` always contains the soft inference references.
`AlignmentResult.class_averages` contains the selected final-output estimator;
with the default configuration it is identical to `references`.
Adaptive diagnostics additionally report coarse/selected/fine candidate counts,
selected posterior mass, retained fine mass, safety-cap hits, local-boundary
hits, normalized entropy, uncertainty-trigger counts, and rescue counts.

The 2.x workflows require an even `H` and use the integer origin
`(H//2, H//2)` for rotation and periodic mirror. This matches the centered DFT
origin and RELION's even-box center convention. The result metadata records the
center convention. Use `convert_v1_4_poses_to_integer_center()` before applying
saved 1.4 `PoseSet` values with 2.x.

Class-prior rows are normalized. Values must be finite and non-negative, each
row must contain a positive value, and zero forbids the corresponding
reference. One-hot rows implement fixed-class refinement.

`make_class_priors` accepts exactly one feedback representation. Hard
assignments require `n_components`; soft responsibilities infer K from their
second dimension. The returned prior is
`trust * normalized_feedback + (1 - trust) / K`. Thus `trust=1` with hard
assignments fixes class membership, while values below one allow corrective
multi-reference reassignment. Soft responsibilities are the recommended input
for corrective refinement because they retain classification uncertainty. A
high-trust one-hot prior is intentionally much more conservative and may not
change any hard assignments.

## Backends

`available_alignment_backends()` reports `cpu`, native `cuda`, fallback `cupy`,
the compatibility alias `gpu`, and `auto`. Explicit `cuda` or `cupy` requests
raise when unavailable; only `auto` is allowed to descend CUDA → CuPy → CPU.

## Legacy compatibility

`poses_from_legacy_params()` and `poses_to_legacy_params()` explicitly convert
pose storage. They do not make the old and new interpolation/boundary semantics
pixel-equivalent, and the latter rejects mirror poses because legacy arrays
cannot represent them.

The AlignImg 0.2 `run_alignment()`/`run_transform()` engine and its
`single`/`multicore` backends were removed from the 2.x package. The frozen
1.10 source distribution remains the recovery point for historical execution.
