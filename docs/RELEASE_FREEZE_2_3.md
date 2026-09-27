# AlignImg 2.3.0 release freeze

Status (2026-09-27): **sealed delivery and cleaned-repository server regression
accepted**.
Core and GPU packages target `2.3.0`; Workbench targets `0.7.0`.

## Scope

AlignImg 2.3 is a general-purpose 2-D image alignment library with an even-square
image contract. The existing RF, single-reference, MRA, and local-refinement APIs
share the same engine and result model. Cryo-EM supplies the principal validation
datasets. External applications manage classification and workflow orchestration.

The opt-in `fast3` preset runs three global polar-hard iterations, with one-hot
responsibilities, the shared Fourier M-step, fast-stage half-set diagnostics
disabled, and final raw-average reconstruction enabled. It supports frequent
classification/alignment review cycles. The configurable `fast_hard` preset
retains its ten-iteration default and existing output policy.

Balanced global/RF presets remain the defaults for checkpoints and final
convergence. Continuous quadratic refinement remains available for local pose
updates. Workbench 0.7 exposes Balanced + precise and Fast 3 exploration; the
Fast 3 selection disables local refinement and fixes three iterations.

## Accepted evidence and limits

Stage 0–4 established the frozen baseline, CPU-authoritative polar search,
CPU/CuPy/native-CUDA conformance, deterministic hard inference, class-prior and
shared M-step contracts, GPU resource accounting, and public API/GUI behavior.
Native polar sampling and peak selection keep full correlation maps on the GPU.
The existing bounded-memory policy and explicit backend selection remain intact.

On the frozen 3,050-particle K=1 workload, Fast 3 produced raw-average correlation
`0.948872` in `7.514 s`; balanced 12 produced `0.965926` in `18.674 s`. The
`2.485x` comparison describes complete practical schedules with different
iteration counts. It is not an equal-iteration kernel speed claim. Fast 3 was
both fastest and highest in correlation among the measured 3/5/8/10/12 fast
points; each point's three repeats matched exactly.

The strict `0.95` threshold and original Stage 5 fast-to-fixed-precise gates
remain failed. The user accepted the measured `0.001128` shortfall for opt-in
exploration; this is recorded as limited practical acceptance. Two fixed
quadratic iterations do not reliably recover balanced external accuracy.
Correlation is not classification accuracy. The K=1 operating point gives no
general quality guarantee for other SNRs, K>1, or other image domains.

## Records

- `validation-results/fast-hard/stage-3/` and `stage-4/`: accepted backend and
  public-contract evidence.
- `validation-results/fast-hard/stage-5/FAST95_SWEEP0_REJECTED.md`: strict
  threshold result and complete accuracy/time curve.
- `validation-results/fast-hard/stage-5/FAST3_LIMITED_ACCEPTANCE.md`: product
  decision, local tests, and server source/preset/GUI verification.
- `validation-results/fast-hard/2.3.0-delivery/`: formal source manifest,
  artifact/checksum audit, server instructions, and release record.
- `validation-results/fast-hard/2.3.0-main-cleanup-r2/`: cleaned-repository
  manifest, artifact/handoff audits, corrected product regression, and acceptance.

The formal core/GPU/native version stamp must be `2.3.0`, and GUI module/metadata
must be `0.7.0`. The original sealed delivery passed server installation
identity and CUDA quick verification on 2026-09-27: 156 source files, core/GPU/native
`2.3.0`, GUI `0.7.0`, and `22 passed / 0 failed` on native CUDA. Fast 3 K1,
fixed MRA, and RF each passed three-iteration deterministic/native/raw-output
and workspace checks. The source digest was
`397caef3148c97d9be69ff0d24c38999c43e3139dbd3a2e7c48a6fa52c761481`.

The later test-layout, README, ignore, and packaging cleanup was accepted through
the separate 162-file r2 manifest. Its corrected server regression passed all
252 current-product cases with zero failures, errors, or skips while loading the
installed GPU package and native CUDA module from `site-packages`. The first
cleanup run's ten skips were traced to pytest source-path precedence, not a CUDA
failure. Compute/GUI/native/validation-tool code was unchanged, so another CUDA
rebuild, quick report, or scientific experiment was intentionally not required.
The detailed reviews are recorded in the sealed-delivery and cleanup-r2 server
acceptance records. Historical validation artifacts remain unchanged.

## Downstream integration

RE2DC adapters belong to the RE2DC repository. AlignImg supplies the public
array, pose, prior, result, and backend contracts documented in
`DOWNSTREAM_INTEGRATION_2_3.zh-TW.md`. Original Stage 6 is documentation handoff
and is not a prerequisite for this release. Automatic classification, K
estimation, application-specific controllers, and the stopped 5,000-particle
K10 experiment are outside the revised 2.3 scope.
