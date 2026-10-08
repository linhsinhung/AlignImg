# AlignImg 2.3.1 spatial cache freeze

Frozen on 2026-10-08 following the accepted 2026-10-07 server validation. Retain the workflow-scoped polar
spatial cache as a modest transfer optimization, and close this performance
workstream. No further search, precision, classification or scheduling changes
are included.

Freeze ID: `2.3.1-spatial-cache-09774528`. This is an internal source/evidence
identifier, not a new package version. Core, GPU and native stamps remain
`2.3.1`; Workbench remains `0.7.0`. The original
[2.3.1 release record](RELEASE_FREEZE_2_3_1.md), artifacts and historical evidence
remain unchanged. No new packages are built or published for this freeze.

## Accepted behavior

Multi-iteration polar workflows with Fourier reference updates may retain the
prepared, normalized/masked FP32 particle stack in the workflow GPU workspace.
The source is uploaded once and reused while references and translation grids
are rebuilt each iteration. Public APIs, defaults, pose conventions, mirror
sampling, candidate ordering, CPU mathematics and native kernels are unchanged.

The optional cache shares the existing VRAM budget with Fourier caches. Source
residency and solver batch size are independent; every iteration rechecks
admission. Fourier/M-step workspace takes priority. Eviction or allocation OOM
disables spatial retention for the remaining workflow; existing resident/
streaming recovery remains available. Normal and exceptional exits close the
workspace. No cross-workflow cache, disk streaming or raw-image reuse is added.

## Server validation

The RTX 3090 server loaded the expected core/GPU Python sources from
`align-dev/site-packages` and native CUDA 2.3.1. Validation compares the current
backend to the pinned original 2.3.1 backend on identical input/config hashes;
it does not simulate the baseline by merely disabling the new cache.

- Product tests: **515 passed, 0 failed, 0 skipped**.
- CUDA and CuPy quick: **22/22 each**.
- CUDA and CuPy smoke/representative A/B: all reported gates passed.
- All 30 returned NPZ files match their report hashes and contain finite arrays.
  Assignments and mirrors remain exact; pose/reference differences satisfy the
  unchanged tolerances. Across the frozen/cached comparisons the maximum final
  angle-array difference is `6.103515625e-5` degrees and the maximum shift-axis
  difference is `4.0531158447265625e-6` pixels.
- Same-allocation-schedule repeats pass exactness checks. Natural batch schedules
  vary in some cases; separate controlled repeats establish determinism without
  calling cross-batch results bitwise equal. Cached controlled runs keep caching
  enabled. They are not common-batch causal performance experiments.
- Every cached A/B case reports one spatial upload and two cache hits across
  three iterations. No full correlation-map download, backend fallback or
  unclosed workspace was observed. Allocation plans and measured resource
  checks pass. The A/B runs do not themselves exercise eviction/OOM; failure
  paths are covered separately by the product tests. Pool tracking excludes
  native/cuFFT allocations outside the pool; sampled device usage is not
  continuous total-memory peak proof.

## Performance conclusion

Times include three Fast 3 iterations and final raw averaging, with one warm-up
and three unprofiled repeats. Profiling is performed in a separate run.

| Workload | CUDA original | CUDA cached | CuPy original | CuPy cached |
| --- | ---: | ---: | ---: | ---: |
| Fixed K=3, 384 particles | 1.367 s | 1.367 s | 1.427 s | 1.297 s |
| Open K=10, 1000 particles | 3.718 s | 3.618 s | 3.740 s | 3.630 s |
| K=1, 3050 particles | 7.544 s | 7.584 s | 7.636 s | 7.313 s |

The whole-workflow H2D reduction is approximately 33% in these representative
cases: 50,331,648 / 131,072,000 / 244,000,000 bytes respectively, exactly two
spatial-stack uploads. D2H bytes are unchanged. All cases pass the 5% slowdown
limit. CUDA is largely neutral; the observed CuPy fixed-K3 median reduction is
9.1%, but variable schedules and timing of unchanged stages prevent attributing
all of that difference to the cache. There is no universal speedup claim.

The user accepted this as a bounded, modest optimization, not a reason to add
more mechanisms or continue tuning. This acceptance does not change any test
threshold or rewrite a historical failure.

## Frozen package and evidence

The canonical local archive directory is
`validation-results/performance/polar-spatial-cache/freeze-09774528/`.
Its acceptance index is `freeze-manifest.json`; `checksums.sha256` covers every
payload file. The portable bundle is `alignimg-2.3.1-spatial-cache-09774528-freeze.tar.gz`.

It retains the tested source snapshot and manifest, the documentation-only
closure snapshot, unchanged core wheel/sdist, tested cache-enabled GPU sdist,
original baseline source snapshot, exact server return archive with JSON/NPZ/
JUnit evidence, input hashes, and the original experiment delivery package.
The large input datasets remain in their existing locations; this evidence
archive is not a duplicate dataset distribution.

Identity anchors:

- Tested source (192 files):
  `09774528727e5d98add63247bcf4bbc8a45604aac6002f41193940afa81376f7`.
- Original 2.3.1 comparison source:
  `00656c02547d8174287c5d839472787e5fc7f3949e37d083f3002308105c6dd6`.
- Returned server archive (37 files, including 30 NPZ):
  `c2faade2987b3f64d2e2fe1c23d9e7f1d7596d82f0976b0c6c7b6913d8be4dcf`.
- Native binary identity recorded by the server:
  `f03ddd1a2e7c7ac53de63e9206e99d9847eed5ec784001ed38e6a66111a4a09a`.

The exact rebuilt Linux wheel/native binary was not returned with this run;
its reported identity is evidence, not a retained executable. The GPU sdist can
be rebuilt on the target server, but byte-identical native rebuilds are not
claimed. Do not relabel the original release's Linux wheel as this snapshot.
The server already running the accepted build needs no reinstall or new run.

Only documentation changes follow the tested source. The closure manifest
distinguishes those bytes without suggesting they were the tested runtime.
No files were deleted, no runtime code changed during freeze, and no Git
commit/tag/push or package publication was performed.
