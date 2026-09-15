# AlignImg 1.9 Fourier-domain M-step

## Scope

AlignImg 1.9 adds an explicit `reference_update="fourier"` mode without changing
the RF/global/refine workflow APIs. The frozen 1.8 path remains available as
`reference_update="spatial"` until the new M-step passes CPU, CuPy, native-CUDA,
and RELION-defined validation.

For component `k`, the update is

```text
R_k(q) = sum_(i,g) w_igk T_g I_i(q) / sum_(i,g) w_igk
```

where `w_igk` is the pose/class posterior multiplied by the robust inlier weight.
The implementation transforms and accumulates particle spectra directly. It
performs one IFFT for each non-empty output reference; two additional sets of
reference IFFTs are required when half-set FRC diagnostics are enabled.

The same ragged full posterior used by the spatial M-step is retained. Class priors,
robust weights, mirror poses, center correction, low-pass filtering, empty-component
re-seeding, and half-set effective weights are unchanged.

## Particle spectra

The 1.8 scorer uses a mean-subtracted, circularly masked, unit-norm particle DFT.
The established M-step instead averages the original particle intensities. Reusing
the scoring spectrum would silently change that scientific meaning and reduced
spatial/Fourier reference correlation to approximately 0.94--0.96 in the development
case. Version 1.9 therefore computes and caches two representations once at entry:

- the normalized scoring DFT used by pose inference;
- the raw update DFT used by the M-step.

This removes every per-candidate and per-iteration particle FFT while preserving the
1.8 reference-update meaning. Collapsing the two representations into one would
require an explicit change to the input-normalization contract and is not part of
this stage.

## Backends and memory

- CPU is the NumPy-authoritative complex accumulation and inverse-FFT path.
- Native CUDA performs Fourier pose transforms; CuPy performs weighted complex
  accumulation and inverse FFT.
- The CuPy fallback performs both transform and accumulation through CuPy.

Both GPU paths use the existing `memory_fraction` planner. The raw update DFT cache
is placed on the device only when the full cache and at least one work item fit while
preserving the configured reserve. Otherwise bounded particle batches stream from
host memory. OOM retry continues to halve the batch down to one item.

## Initial validation

Run the isolated production-workflow comparison on the Linux GPU host:

```bash
python tools/server_validation.py \
  --suite quick \
  --backend cuda \
  --only fourier_mstep_workflow_ab \
  --batch-size 512 \
  --output validation-results/alignimg-1.9-fourier-mstep-cuda.json

python tools/server_validation.py \
  --suite quick \
  --backend cupy \
  --only fourier_mstep_workflow_ab \
  --batch-size 512 \
  --output validation-results/alignimg-1.9-fourier-mstep-cupy.json
```

The gate covers global proposal and adaptive refinement with Fourier candidate
scoring, robust weights, fixed class priors, and half-set FRC. It requires unchanged
assignments, responsibilities and top-L posterior; spatial/Fourier reference
correlation above 0.99; CPU/GPU Fourier-reference correlation above 0.9999; finite
FRC; and matching half-set effective weights. Runtime is recorded but is not an
initial scientific pass/fail threshold.

After both GPU gates pass, repeat the 384-particle RELION-defined benchmark with
`--candidate-scoring fourier --reference-update fourier`. Do not tune rescue or pose
search during this comparison.

### Initial GPU gate results

Both corrected GPU gates passed on the RTX 3090 host with AlignImg and alignimg-gpu
1.9.0. Assignments, responsibilities, poses, top-L posterior, and half-set effective
weights were unchanged. The minimum spatial/Fourier reference correlation was
`0.9930` for global proposal and `0.9932` for adaptive refinement. Native CUDA and
CuPy references each correlated with CPU authority above `0.99999999999999`; all FRC
and JSON values were finite. Batch 512 was honored, the raw update spectra used the
device cache, and approximately 5.06 GB of VRAM remained reserved.

| Workflow | Native CUDA spatial/Fourier time ratio | CuPy spatial/Fourier time ratio |
| --- | ---: | ---: |
| global proposal | 0.125x | 3.872x |
| adaptive refinement | 1.192x | 1.089x |

The small native-CUDA global case includes a roughly one-second first-use cost. This
is recorded rather than optimized from a 16-particle workload; the 384-particle gate
will determine whether it remains relevant at the intended scale.

- `validation-results/alignimg-1.9-fourier-mstep-cuda-v2.json`
  (`sha256:d52b776d189c0350ee4f263439e2e56ccf70976efe70ab778aa758c92089a440`)
- `validation-results/alignimg-1.9-fourier-mstep-cupy-v2.json`
  (`sha256:7f007b1b3205d0741c03517d454f8707839d559eadf352c316f09d71c63cdda9`)

## RELION-defined 384-particle result

The one-factor comparison against the frozen 1.8 Fourier-candidate/spatial-M-step
result completed successfully. All artifacts were finite, deterministic repeats were
exact, and independent K=1 runs were identical to their fixed-MRA component slices.

| Measurement | 1.8 spatial M-step | 1.9 Fourier M-step |
| --- | ---: | ---: |
| refinement runtime | 43.789 s | 21.519 s |
| initializer + refinement runtime | 46.604 s | 24.360 s |
| angle errors above 11.25 degrees | 9 / 384 | 11 / 384 |
| shift errors above 2 px | 13 / 384 | 16 / 384 |
| angle within 2.8125 degrees | 83.59% | 81.51% |
| shift within 1 px | 78.39% | 76.56% |
| mean aligned-particle NCC | 0.236069 | 0.233313 |
| mean result-reference/oracle correlation | 0.907516 | 0.900166 |
| mean aligned-mean/oracle correlation | 0.867728 | 0.867655 |
| median reliable stable FRC cutoff | 0.218160 | 0.213734 cycles/pixel |

Refinement is `2.03x` faster and the complete measured path is `1.91x` faster. The
small tail grows by two angle and three shift outliers, concentrated mainly in one
component. The aligned-mean/oracle correlation is effectively unchanged, while the
direct result-reference metrics show the expected Fourier-interpolation smoothing.
This is accepted as a bounded interpolation tradeoff rather than a reason to add
outlier constraints or gridding corrections in this stage.

- `validation-results/re2dc-70s-pose-fourier-mstep-1.9.json`
  (`sha256:c0baa3566aa33786e95796fe4025df13b3f1ffc237571833cd256787a0fb4713`)
- fixed-MRA result NPZ
  (`sha256:acbeb697b48db86c8c0d57e30ebf1adfb0f8c009610bed32338e5f1beaf10f4f`)
- fixed-MRA references MRCS
  (`sha256:1a0b9158c52c6bde310ad182559b2669155c4c996f2327d1593db44c2df40513`)

## Final closure

The complete quick regression suite passed on both production GPU paths. Each
report recorded 15 of 15 cases passing, including the integer-center contract,
CPU/GPU parity, Fourier-native conformance, raster/Fourier workflow A/B,
Fourier-M-step workflow A/B, RF, robust refinement, class feedback, adaptive
refinement and rescue, low-SNR global alignment, and K=20 scaling. Both reports
were finite. Batch 512 was honored and the 80% VRAM budget retained approximately
5.06 GB of reserve without an OOM retry.

- native CUDA quick report
  (`sha256:bd2d2f35dd40828df0c795e8dbeb465475f5b2fa2fc15b1076bc04455e66adda`)
- CuPy fallback quick report
  (`sha256:024ea67a9f9f78bc5ecb4f24947b8be000e98f7fcc139d7f21086272a6ce083d`)

The final local suite completed with 134 passed and 12 skipped tests. The skipped
tests require unavailable optional runtime facilities; the eight emitted warnings
are the intentional deprecation notices for the legacy 0.2 API scheduled for
removal in AlignImg 2.0.

The following release artifacts passed `twine check` and freeze AlignImg 1.9.0:

| Artifact | SHA256 |
| --- | --- |
| `dist/alignimg-1.9.0-py3-none-any.whl` | `ec796117ee92f6d8c8541133a214f225ee840de2371d7ffaf52ae13261bef0a1` |
| `dist/alignimg-1.9.0.tar.gz` | `a899c74b2118f9de897c1a4c3ec0723da898eb4cb694ba58bde4e76b8be80213` |
| `packages/alignimg-gpu/dist/alignimg_gpu-1.9.0.tar.gz` | `a7528890890848f0ed95c34169278fcf1d929222413aa3ef9ed00f9ad066720b` |

These artifacts preserve `reference_update="spatial"` as the compatibility
default. The accepted Fourier-native path is selected explicitly with
`candidate_scoring="fourier"` and `reference_update="fourier"` until a later
release changes the public default deliberately.
