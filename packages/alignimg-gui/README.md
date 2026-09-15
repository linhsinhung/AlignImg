# AlignImg Workbench

AlignImg Workbench 0.6 is a development GUI for the AlignImg 2.x 2D cryo-EM
alignment library. It keeps PyQt out of the core package and runs each
alignment in an isolated worker process.

## Install

Install AlignImg first, then install the GUI package:

```bash
python -m pip install -e .
python -m pip install -e packages/alignimg-gui
```

The optional `alignimg-gpu` package supplies native CUDA and CuPy backends.
The GUI remains usable with the CPU backend when it is absent.

## Start

```bash
alignimg-gui
```

or:

```bash
python -m alignimg_gui
```

The workbench has two primary pages:

- **Reference-Free** bootstraps `K` references from the particle stack, runs
  global RF alignment, and can follow it with local refinement.
- **Reference-Based / MRA** runs global alignment against a reference stack
  (`K >= 1`) and can follow it with local refinement. It can also skip global
  search and refine a prior AlignImg result NPZ against the supplied references.

Local refinement defaults to two iterations and can use soft responsibilities,
fixed assignments, free reassignment, or trusted hard assignments. The global
and refinement stages use the same AlignImg engine; the GUI only composes the
workflow and does not implement another alignment method.

The GUI accepts particle/reference MRC stacks and prior AlignImg result NPZ
files. It does not perform STAR parsing, CTF correction, denoising,
classification, or 3D reconstruction.

Result NPZ files written by AlignImg 1.5 and later record their integer-origin center
convention. Older files without that field are treated as 1.4 geometric-center
poses and converted when used as refinement input.

Workbench 0.6 separates the global pose-search selector from the refinement
selector. Global alignment defaults to polar proposal; refinement defaults to
continuous quadratic search, while adaptive posterior remains available as a
2.1 compatibility choice. It also exposes
Fourier/raster candidate scoring, uniform/whitened Fourier NCC, and
Fourier/spatial reference updates. The run summary records the actual backend,
FFT policy, rescue statistics, and GPU memory plans.
The optional **Reconstruct final average from raw particles** control keeps
soft references for inference while writing final class averages reconstructed
from the raw stack with final MAP poses, hard class assignments, and inlier
weights.
Global search uses the selected RF or reference-based preset; the optional
refinement stage uses the selected quadratic or adaptive refinement settings. Saved result
NPZ files include retained pose entropy and MAP posterior for uncertainty
inspection.

Each run writes a self-contained directory containing:

```text
spec.json
report.json
references.mrcs
result.npz
run.log
global/
    report.json
    references.mrcs
    result.npz
refine/
    report.json
    references.mrcs
    result.npz
```

Only stages that ran are present. The top-level result is the final pipeline
result, with a combined global/refine reference history. Completed
`report.json` files can be loaded later without rerunning alignment.
