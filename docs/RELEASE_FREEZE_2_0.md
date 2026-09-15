# AlignImg 2.0.0 release freeze

AlignImg 2.0.0 is the frozen consolidation release for the completed 1.7.1 to
2.0 alignment-engine upgrade. No parameter retuning or new scientific model was
introduced during the 2.0 consolidation.

## Frozen defaults and contracts

- Fourier-native candidate scoring is the default.
- Uniform band-limited Fourier NCC is the default score model.
- The Fourier-domain soft M-step is the default reference update.
- Raster scoring, the spatial M-step, and empirical whitening remain explicit
  comparison modes.
- The integer-origin, periodic mirror/rotation/translation convention is
  unchanged from the validated 1.5--1.10 implementation.
- RF, global alignment, robust refinement, class feedback, transforms, and
  class-prior construction have signature tests.
- The deprecated 0.2 execution engine is absent; explicit pose-format and
  v1.4-center adapters remain.

## Validation gates

- Final local test suite: 137 passed and 13 environment-dependent tests skipped.
- CPU quick validation: 16/16 cases passed.
- RTX 3090 CUDA quick validation: 16/16 cases passed using
  `alignimg-soft-fourier-cuda`, with no backend fallback or batch reduction.
- All 133 CUDA memory plans retained the requested batch size 512.
- The frozen 5000-particle, K=10, 20-iteration, three-seed real-data baseline is
  documented in `FINAL_RF_MRA_BASELINE_1_10.md`. It already used the Fourier
  candidate scorer and Fourier M-step promoted by 2.0.

## Pre-freeze known-reference check

`tools/align_known_reference.py` was exercised on
`data/local/test_align.mrcs`: 3050 particles of size 100 at 3.36 Å/pixel,
aligned to `data/local/mu_aligned_mean.mrc` with K=1.

One CPU global iteration improved the correlation of the stack average with
the supplied reference from 0.7322 to 0.9767. Responsibilities were normalized
within `1.2e-7`; stable half-set FRC was 0.308 cycles/pixel (about 10.9 Å).
The class average, pose arrays, posterior metadata, JSON report, and comparison
figure are under `validation-results/local-known-reference-2.0-global/`.

## Scope boundary

The native CUDA backend intentionally remains hybrid: compiled CUDA kernels
perform the native transforms while CuPy supplies cuFFT, device arrays, NCC
reduction, and Fourier accumulation. A fully custom CUDA/C++ replacement for
those library operations is not part of the 2.0 contract.

Publishing to a package index and creating a repository tag are operational
release actions, not changes to this frozen source or scientific baseline.
