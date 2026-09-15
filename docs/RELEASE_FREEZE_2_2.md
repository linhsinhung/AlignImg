# AlignImg 2.2.0 release freeze

AlignImg 2.2.0 completes Continuous Quadratic Refinement. The public workflow
API is unchanged. `reference_free_align()` and global alignment retain the 2.1
proposal search; `refine_alignment()` now receives `quadratic_refine` from the
default refine preset. Callers can explicitly select `adaptive_posterior` to
reproduce the frozen 2.1 local-search behavior.

## Frozen refinement contract

The refine preset screens a prior-centred plus/minus 7-degree window at a
1-degree step and searches translations inside plus/minus 3 pixels. A Fourier
correlation map supplies every integer translation for each angle. Independent
three-point x/y and angular parabola fits propose continuous poses; concavity,
half-step bounds, boundary checks, and exact Fourier-NCC rescoring guard every
fit. The complete screened angular profile remains the internal soft posterior
for responsibilities and the Fourier M-step. Pose values remain float32.

The same solver handles K=1, fixed K>1 MRA, and corrective class feedback.
Class priors determine which references are active; there is no separate MRA
engine. Robust weights, half-set FRC, Fourier M-step, final raw-average output,
integer-centre convention, and the 80-percent VRAM limit remain intact.

## Accepted validation

Stages 1 through 4 established CPU-authoritative fitting, CPU/CuPy/native-CUDA
conformance, deterministic K=1 behavior, fixed-MRA equivalence, and two- and
five-iteration class-feedback behavior. The controlled Stage 4 RELION subset
improved both aggregate pose-error medians and ran 5.256 times faster than the
adaptive baseline.

The final Stage 5 candidate passed 21 of 21 quick cases independently on CPU,
CuPy, and native CUDA. On the 3,050-particle, 100-by-100 K=1 user stack, two
quadratic iterations took 65.159 seconds versus 459.070 seconds for adaptive,
a 7.045-fold speedup. Expected Fourier NCC increased from 0.22346 to 0.26110.
Final raw-average correlation to the supplied reference changed by -0.00498
and passed the frozen 0.005 tolerance. The CUDA run used a 512-particle map
chunk and the native peak backend without fallback.

Soft-reference correlation to the supplied reference is retained as a
diagnostic, not an external quality gate: the adaptive candidate posterior is
far more diffuse and consequently smooths its soft M-step reference. The
strategy-neutral external gate uses final MAP poses applied to raw particles;
soft-reference coherence is checked against the corresponding raw average.
This interpretation and the unchanged numerical thresholds are recorded in
the Stage 5 gate-review JSON.

## Release evidence

The formal delivery directory contains the exact source archive, source
manifest, regression fixtures, distribution checksums, validation-report
checksums, and the Stage 5 acceptance note. Core and GPU package versions and
the native extension build stamp are all `2.2.0`. AlignImg Workbench 0.6
exposes independent global and refine strategy controls and defaults refine to
continuous quadratic.

Automatic window expansion, a full xy Hessian fit, gradient optimizers,
additional rescue constraints, RF/global quadratic search, automatic K
selection, and biological classification remain outside the 2.2.0 scope.
