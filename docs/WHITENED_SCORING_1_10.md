# AlignImg 1.10 empirical-whitened Fourier NCC

AlignImg 1.10 adds `score_model="whitened_fourier_ncc"` as an opt-in candidate
score. The default remains `score_model="fourier_ncc"`; no existing workflow
changes unless the new model is requested explicitly.

## Mathematical contract

Let \(F_i(q)\) be the cached DFT of masked, mean-subtracted, unit-norm particle
\(i\). The empirical stack power is

\[
P(q)=\frac{1}{N}\sum_i |F_i(q)|^2.
\]

AlignImg averages this power in integer radial rings, applies the fixed
three-point kernel \([1/4,1/2,1/4]\), and floors it at \(10^{-3}\) times the
median active-ring power. Within the configured frequency band, the score weight
is

\[
W(q) \propto \frac{1}{\max(P_{\mathrm{radial}}(q),P_{\mathrm{floor}})}.
\]

The active weights are normalized to mean one. The weighted score is

\[
\operatorname{NCC}_W(A,B)=
\frac{\Re\sum_q W(q)A(q)B(q)^*}
{\sqrt{\sum_q W(q)|A(q)|^2\sum_q W(q)|B(q)|^2}}.
\]

The same weights are used for translation cross-correlation and final candidate
NCC. Because radial Fourier magnitude is invariant to in-plane rotation and
translation, one stack-level estimate is shared by every pose candidate.

## Scope and limitations

- This is empirical spectral whitening, not a fitted RELION likelihood.
- The estimate contains both signal and noise; it is not a per-particle noise PSD.
- CTF correction remains an external preprocessing responsibility.
- Polar angular proposal, soft posterior, class/pose priors, rescue, robust
  weights, and spatial/Fourier M-steps are unchanged.
- No public floor, smoothing, or whitening-strength tuning parameter is added in
  this stage.

The CPU computes the authoritative weights once. Native CUDA and CuPy consume the
same array in their VRAM-bounded score reductions, so backend differences cannot
come from independently fitted spectra. Result metadata records the selected
model, estimator description, and active-weight summary.

## Validation sequence

Run the isolated synthetic and backend gate first:

```bash
python tools/server_validation.py \
  --suite quick \
  --backend cuda \
  --only whitened_scoring_ab \
  --batch-size 512 \
  --output validation-results/alignimg-1.10-whitened-cuda.json

python tools/server_validation.py \
  --suite quick \
  --backend cupy \
  --only whitened_scoring_ab \
  --batch-size 512 \
  --output validation-results/alignimg-1.10-whitened-cupy.json
```

The controlled case uses strong radially colored noise and requires whitening to
recover all 24 known poses while uniform NCC recovers at most 20. It also checks
finite normalized weights and CPU/backend pose and posterior agreement.

### Initial GPU gate result

Both gates passed on the RTX 3090 host with AlignImg and alignimg-gpu 1.10.0.
Uniform NCC recovered 19 of 24 poses within 5 degrees and had a 20.625-degree
mean error; empirical whitening recovered all 24 with zero mean error. Native
CUDA and CuPy selected exactly the CPU-authoritative poses, and their maximum
posterior error was approximately `1.2e-15`. Active weights were finite and had
mean one. Batch 512 was honored without an OOM retry.

- `validation-results/alignimg-1.10-whitened-cuda.json`
  (`sha256:755ceccd3f04389386df7cbe6db94da7cd382110e5c4542053d1b35f60da6823`)
- `validation-results/alignimg-1.10-whitened-cupy.json`
  (`sha256:58d1b5c7cb34e4171a8dffe58ba938b73259a968b2b6e1109168c07a130e7c3f`)

Only after both gates pass, compare against the frozen 1.9 384-particle result:

```bash
python tools/re2dc_70s_pose_benchmark.py \
  --backend cuda \
  --candidate-scoring fourier \
  --score-model whitened_fourier_ncc \
  --reference-update fourier \
  --batch-size 512 \
  --rescue-uncertain-particles \
  --rescue-normalized-entropy-threshold 0.967 \
  --rescue-min-score-improvement 0.02 \
  --only adaptive_known_reference adaptive_fixed_mra \
  --output validation-results/re2dc-70s-pose-whitened-1.10.json
```

This is a one-factor scientific A/B comparison against 1.9. Whitening is retained
as opt-in unless the real-data result gives a clear, broad improvement; a small
change in the final outlier tail is not sufficient reason to change the default.

## RELION-defined 384-particle result

The one-factor comparison completed successfully on the RTX 3090. All JSON,
NPZ, and MRCS values were finite, fixed assignments remained unchanged, and the
deterministic repeat was exact.

| Measurement | 1.9 uniform NCC | 1.10 empirical whitening |
| --- | ---: | ---: |
| adaptive fixed-MRA runtime | 21.519 s | 20.010 s |
| angle errors above 11.25 degrees | 11 / 384 | 14 / 384 |
| shift errors above 2 px | 16 / 384 | 21 / 384 |
| angle within 2.8125 degrees | 81.51% | 80.99% |
| angle within 5.625 degrees | 95.31% | 93.49% |
| shift within 1 px | 76.56% | 75.26% |
| mean aligned-particle NCC | 0.233313 | 0.231593 |
| mean result-reference/oracle correlation | 0.900166 | 0.900011 |
| mean aligned-mean/oracle correlation | 0.867655 | 0.866228 |
| median reliable stable FRC cutoff | 0.213734 | 0.270723 cycles/pixel |

Whitening substantially increased half-set spectral reproducibility, but did not
produce a broad improvement in the primary RELION pose benchmark. Four new angle
outliers appeared while one was recovered; nine new shift outliers appeared while
four were recovered. The runtime difference is small enough to be ordinary run
variation. Therefore `fourier_ncc` remains the default and empirical whitening is
retained only as an explicit option for data with demonstrably colored noise.

The empirical spectrum is estimated independently for every workflow invocation.
Consequently, a class subset run independently with K=1 does not use exactly the
same weights as that class inside a full-stack fixed-MRA call, and the strict K=1 /
fixed-MRA numerical equivalence of uniform NCC does not apply. This is an explicit
data-adaptive score-model property, not a backend discrepancy. Adding an external
or frozen noise-spectrum contract is deferred until a real workflow requires it.

- `validation-results/re2dc-70s-pose-whitened-1.10.json`
  (`sha256:5bdfd2bc777e9461104ac9c69d247cbd867d6dc24f1ec89285a24e39016d85c2`)
- fixed-MRA result NPZ
  (`sha256:cf65edec9beec4f8c7c0ed7e3075d3e6e5cb0bc3f6ff28a0376a45f6b2f90344`)
- fixed-MRA references MRCS
  (`sha256:ab64c6809d11cffbc47c1f19638d71d5f49d70f4eb02a7462730491a0d4d337b`)

## Final closure

The complete quick regression suite passed on both production GPU paths. Each
report recorded 16 of 16 cases passing, including all frozen 1.9 workflows and
the new whitening A/B gate. Both reports were finite. Batch 512 was honored,
approximately 5.06 GB remained reserved by the 80% VRAM policy, and no OOM retry
occurred.

- native CUDA quick report
  (`sha256:6ec2904675ab76d8fa7b70db82bf1c570ead0c5c2419fdae0f56e259ca13b2ce`)
- CuPy fallback quick report
  (`sha256:11b27731945b3ceaa1e4c4b0f6f0d13777bd257f907adc7a5d7e6d247b7b36c3`)

The final local suite completed with 138 passed and 14 skipped tests. The eight
warnings are the intentional legacy 0.2 API deprecation notices. All modified
files except the pose-benchmark runner pass Ruff; that runner retains its six
pre-existing F401/E731 findings and was not refactored in this feature release.

The following release artifacts passed `twine check` and freeze AlignImg 1.10.0:

| Artifact | SHA256 |
| --- | --- |
| `dist/alignimg-1.10.0-py3-none-any.whl` | `12332e6e75c5d35cc5112cf446a6e29c4db00f10780ac7884a6cf8d3af9d36db` |
| `dist/alignimg-1.10.0.tar.gz` | `6362fa980dce66294da0439c2c4e20d35cc8131e45457957ca56056141889b44` |
| `packages/alignimg-gpu/dist/alignimg_gpu-1.10.0.tar.gz` | `8947ab5e948ebd75996ef2be79f85f2e95aa066ca1b8dddc8a216f99b7c37d0a` |

AlignImg 1.10 closes with `fourier_ncc` as the default score and
`whitened_fourier_ncc` as a validated, explicit option. No automatic scorer
selection is introduced.
