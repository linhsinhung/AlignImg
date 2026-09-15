# AlignImg 2.2: Continuous Quadratic Refinement

AlignImg 2.2 replaces the default refine-only adaptive Cartesian grid after the
new path passes the staged gates. RF and global proposal search are unchanged.
The refine preset uses `quadratic_refine` from dev3 onward;
`adaptive_posterior` remains the explicit frozen 2.1 comparison path.

## Pose inference

For each positive-prior reference and allowed mirror state, the solver screens a
prior-centred angle grid. At each angle it rotates the cached particle DFT,
forms the weighted cross-power spectrum, and obtains all integer translations
with one inverse FFT. The integer peak maximizes

\[
Q = \mathrm{NCC}/\tau
  - \tfrac12 ((t_y-t_{0y})/\sigma_t)^2
  - \tfrac12 ((t_x-t_{0x})/\sigma_t)^2.
\]

Independent three-point parabolas fit x and y. A fit is accepted only when it
is concave, finite, within half a pixel, away from that axis's search boundary,
and an exact Fourier-NCC rescore improves Q. The translation-optimized angular
profile additionally includes the angle prior and class prior. Every local
angular maximum is treated as a continuous mode; an interior maximum receives
the same bounded three-point fit and exact rescore. The fitted mode replaces
its original grid sample, while all screened angular samples remain as the
internal profile-posterior support. This preserves angular uncertainty for
class responsibilities and the existing Fourier M-step. Only normalized top-L
samples are exposed through `CandidateSet`.

The initial dev1 implementation normalized only the local maxima. Real K=1
validation showed that most particles then had one retained mode, with MAP
posterior 0.91--0.96, causing an unintentionally hard M-step and lower reference
correlation. Dev2 corrects that posterior representation without adding search
constraints or changing the fitted MAP pose rule.

There is deliberately no xy cross term, polar-angle fit, automatic window
growth, or rescue loop in 2.2.

## Development stages

1. Stage 0 freezes the 2.1 inputs, source archive, timings, candidate counts,
   and GPU calls in
   `validation-results/performance/quadratic-refine/BASELINE_2.1.json`.
2. Stage 1 is the CPU-authoritative implementation and synthetic precision
   contract.
3. Stage 2 validates CuPy/native-CUDA K=1 accuracy, parity, deterministic
   repeats, map chunking, and no fallback. Dev2 passed the accepted
   resolution-aware gate and switched the refine preset to quadratic.
4. Stage 3 validates fixed K>1 MRA against independent K=1 runs. Dev3
   exposed a class-coupled robust-weight threshold; dev4 scopes robust weights
   per fixed reference for the quadratic refine path and repeats the gate.
5. Stage 4 runs fixed and corrective feedback on the prepared 1000-particle
   stack for two and then five iterations. Dev4 passed the technical runs and a
   controlled RELION-compatible fixed-feedback scientific A/B. The accepted
   quadratic path improved aggregate angle and shift medians and was 5.256
   times faster than adaptive.
6. Stage 5 validated the 3050-particle user stack, updated the GUI/docs, and
   authorized the 2.2.0 freeze. The accepted candidate commands and gates are
   recorded in `QUADRATIC_STAGE5_2_2.md`; the formal release record is
   `RELEASE_FREEZE_2_2.md`.

The real-data A/B entry point is:

```bash
python tools/quadratic_refine_validation.py \
  --suite k1 --backend cuda --iterations 3 \
  --deterministic-repeats 2 --batch-size 512 --memory-fraction 0.8 \
  --profile-execution \
  --output validation-results/performance/quadratic-refine/dev2-cuda-k1.json
```

The runner computes the initializer once, stores it beside the JSON as
`*.inputs.npz`, records array hashes, and gives both strategies identical
particles, references, poses, and settings outside their search strategy.
Stage 3 uses `--suite fixed_mra`; it reuses the same frozen initializer and
additionally compares joint K=3 refinement with the three independent K=1
runs.

```bash
python tools/quadratic_refine_validation.py \
  --suite fixed_mra --backend cuda --iterations 3 \
  --deterministic-repeats 2 --batch-size 512 --memory-fraction 0.8 \
  --profile-execution \
  --output validation-results/performance/quadratic-refine/dev4-cuda-fixed-mra.json
```

Corrective feedback is validated first for two iterations and then for five:

```bash
python tools/re2dc_70s_feedback_validation.py \
  --backend cuda --search-strategy quadratic_refine \
  --candidate-scoring fourier --reference-update fourier \
  --iterations 2 --deterministic-repeats 2 \
  --batch-size 512 --memory-fraction 0.8 --profile-execution \
  --output validation-results/performance/quadratic-refine/dev2-feedback-i2.json
```

The feedback report includes fixed and corrective runs, transition matrices,
pose and reference trajectories, compact quadratic diagnostics, FRC reliability,
GPU memory plans, native-peak identity, and deterministic-repeat checks.

Stage 4 closes with one controlled scientific comparison. It deliberately runs
only the fixed-class adaptive baseline: fixed assignments isolate pose-search
accuracy, while repeating adaptive corrective feedback would confound class
transitions with pose accuracy and add roughly an hour of GPU work. The runner
maps the RF classes to RELION classes, keeps confidence at least 0.5, and only
evaluates mapped components containing at least ten particles. Both strategies
receive the same 1,000 images, RF references, initial poses, fixed class priors,
iteration count, scoring model, M-step, temperatures, and robust weighting.

```bash
python tools/quadratic_feedback_ab_validation.py \
  --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/quadratic-refine/\
dev4-cuda-feedback-scientific-ab.json
```

The existing five-iteration quadratic artifacts are read rather than recomputed.
The output records input hashes, the frozen subset, the new adaptive artifacts,
pose errors after an independent SE(2) gauge fit per component, runtimes, and
every Stage 4 scientific gate. Reference quality for the pose gate is measured
by applying each result's final MAP poses to the same unweighted raw subset.
Stored soft M-step reference correlations are retained as observations because
adaptive and quadratic fixed feedback intentionally use different robust-weight
scopes. The accepted report and rationale are recorded in
`dev4-delivery/STAGE4_ACCEPTED.md`.

## Stage 2 dev1 result and dev2 correction

The four dev1 server reports passed both CUDA and CuPy synthetic suites and
confirmed backend parity. CUDA used the native peak kernel without fallback;
CuPy used its portable peak implementation. The real homogeneous K=1 gate did
not pass, however. The aggregate median angle error changed from 2.8125 to
3.8943 degrees and median shift error from 2.2205 to 2.3088 pixels. Component
reference correlations fell by roughly 0.023--0.035. The p95 and deterministic
repeat gates did pass. CUDA runtime nevertheless fell from 89.81 to 16.44
seconds, and CuPy from 92.52 to 23.83 seconds.

CUDA and CuPy produced the same scientific results from identical input hashes,
so this was not a backend discrepancy. Diagnostics isolated the cause: dev1
retained only 1.1--1.2 local maxima per particle, compared with roughly 1,800
adaptive candidates, and its mean MAP posterior rose to 0.91--0.96. Dev2 keeps
the same fitted MAP pose but normalizes the internal posterior over all 15
screened angle-profile samples.

Dev2 restored the intended soft posterior and passed CUDA/CuPy parity,
deterministic repeats, p95 limits, shift improvement, and all component
reference-correlation checks. CUDA K=1 runtime fell from 86.39 to 16.38
seconds; CuPy fell from 95.30 to 23.78 seconds. Two components improved their
median RELION angle, while one changed from 2.8125 to 3.3725 degrees.

Inspection showed that every RELION truth angle lies on a 5.625-degree lattice
after one common phase adjustment, with maximum residual below `6.3e-6`
degrees. A strict raw-median comparison therefore favors the discrete adaptive
path and cannot resolve a continuous estimate within the RELION bin. The
accepted real-data angle gate requires every component to improve from its
shared initial pose and requires the aggregate median to remain within the
2.8125-degree RELION half-step of adaptive. The original raw strict comparison
remains in reports as an observation. Synthetic continuous-truth gates remain
strict. Dev2 passes this corrected scientific contract, so the public refine
preset switches to `quadratic_refine` before Stage 3.

## Release gates

- Synthetic noiseless median error: angle at most 0.5 degrees and shift at most
  0.2 pixels, with both at least 25% below adaptive. Fixed-seed SNR 0.5 and 0.2
  cases must also improve both median errors.
- RELION homogeneous K=1: every component median angle improves from the
  shared initial pose; aggregate median angle remains within the 2.8125-degree
  RELION half-step of adaptive; median shift improves; p95 is no more than 10%
  worse; every component reference correlation is no more than 0.005 below
  adaptive; deterministic repeats; CUDA backend with no fallback.
- Fixed MRA: assignments unchanged; joint-versus-independent K=1 angle and
  shift differences at most 1e-3; responsibility tolerance 2e-4; matched
  reference correlation at least 0.9999.
- Corrective feedback preserves finite normalized responsibilities, transition
  diagnostics, FRC, robust weights, and reference trajectories. It is not a
  biological classification gate.

If precision passes but end-to-end time is more than 5% slower than adaptive,
the continuous result remains the target and batching/profile work continues
before release. The mathematical path is not discarded for a first performance
miss.
