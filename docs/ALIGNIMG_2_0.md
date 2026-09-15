# AlignImg 2.0 consolidation

AlignImg 2.0 promotes the validated Fourier-native pipeline to the supported
default without changing the 1.10 alignment mathematics or tuned workflow
presets.

## Supported pipeline

- `candidate_scoring="fourier"` is the default. Rotation, optional mirror,
  translation phase, and Fourier NCC operate from cached particle DFTs.
- `reference_update="fourier"` is the default. The soft M-step accumulates
  aligned Fourier coefficients and performs one output IFFT per reference per
  iteration, plus the IFFTs required by half-set diagnostics.
- `score_model="fourier_ncc"` remains the default. Empirical radial whitening
  remains an explicit option rather than a new scientific default.
- `candidate_scoring="raster"` and `reference_update="spatial"` remain
  available for controlled regression comparisons.
- VRAM use remains bounded by `memory_fraction`, with 0.8 as the default hard
  limit. An explicit batch size of 512 remains supported when the memory plan
  permits it.

The frozen pre-2.0 real-data measurements and artifact hashes are recorded in
[`FINAL_RF_MRA_BASELINE_1_10.md`](FINAL_RF_MRA_BASELINE_1_10.md).

## Public API

The supported workflow API is:

```python
reference_free_align(images, *, n_components, config=None, backend="cpu")
align_to_references(images, references, *, class_priors=None, config=None, backend="cpu")
refine_alignment(images, references, initial_poses, *, class_priors=None, config=None, backend="cpu")
transform_images(images, poses, *, backend="cpu")
make_class_priors(*, assignments=None, responsibilities=None, n_components=None, trust=1.0)
```

`AlignmentConfig`, `PoseSet`, `CandidateSet`, and `AlignmentResult` are the
public data contracts. Signature tests freeze the four workflow entry points.

The deprecated 0.2 `run_alignment`, `run_transform`, `MAPEMConfig`, and
`single`/`multicore` backends were removed. The explicit
`poses_from_legacy_params`, `poses_to_legacy_params`, and
`convert_v1_4_poses_to_integer_center` adapters remain available.

## Linux/CUDA installation

Install the core package from the checkout:

```bash
python -m pip install --force-reinstall --no-deps -e .
```

For the RTX 3090 validation host, build the native extension for compute
capability 8.6 from the 2.0 GPU source distribution:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" \
python -m pip install -v \
  --force-reinstall \
  --no-build-isolation \
  --no-deps \
  --no-cache-dir \
  packages/alignimg-gpu/dist/alignimg_gpu-2.0.0.tar.gz
```

Verify the installed code before validation:

```bash
python - <<'PY'
import alignimg
import alignimg_gpu

print("AlignImg:", alignimg.__version__, alignimg.__file__)
print("AlignImg GPU:", alignimg_gpu.__version__)
print("Backends:", alignimg.available_alignment_backends())
print("Defaults:", alignimg.AlignmentConfig().candidate_scoring,
      alignimg.AlignmentConfig().reference_update)
PY
```

Then run the CUDA smoke suite:

```bash
python tools/server_validation.py \
  --suite quick \
  --backend cuda \
  --batch-size 512 \
  --output validation-results/alignimg-2.0-cuda-quick.json
```

The RTX 3090 validation completed all 16 quick cases without fallback or batch
reduction. Candidate-score error was below `6e-8`; CPU/CUDA poses agreed at the
reported precision, and all 133 recorded memory plans retained batch size 512.

## Known-reference K=1 alignment

`tools/align_known_reference.py` is the reproducible command-line path for a
stack with no prior poses and one trusted reference. It first performs global
pose inference and can then use those poses for adaptive robust refinement.
Its class assignment is necessarily zero and its class responsibility is one
for every particle.

The pre-freeze `data/local/test_align.mrcs` check aligned 3050 images of size
100 using `mu_aligned_mean.mrc`. One CPU global iteration improved correlation
between the stack average and the supplied reference from `0.7322` to `0.9767`.
The stable half-set FRC cutoff was `0.308 cycles/pixel`, or about `10.9 Å` at
`3.36 Å/pixel`. The run is recorded under
`validation-results/local-known-reference-2.0-global/`.

## Workbench 0.4

The GUI exposes the Fourier/raster candidate scorer, uniform/whitened score
model, and Fourier/spatial M-step. Its run-summary page displays the actual
engine and backend, FFT/GPU policy, rescue totals and trajectories, and the
VRAM plan used by each recorded stage.
