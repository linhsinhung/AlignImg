# AlignImg Workbench

AlignImg Workbench 0.7 is a development GUI for the AlignImg 2.3 2D image
alignment library. It keeps PyQt out of the core package and runs each
alignment in an isolated worker process.

## Install

From the repository root, install AlignImg first, then the GUI package:

```bash
python -m pip install .
python -m pip install ./packages/alignimg-gui
```

The GUI installs `mrcfile`, PyQt6, and pyqtgraph as dependencies and requires a
graphical desktop. The core stays independent of these GUI dependencies.
For editable development installs or upgrades in an existing environment, see
the [installation guide](../../docs/INSTALLATION.md). Workbench module and
distribution versions should both be `0.7.0`, with core `2.3.0`.

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

Workbench 0.7 provides two explicit pipeline strategies. **Balanced + precise**
remains the default. **Fast 3 exploration** runs the validated three-iteration
polar-hard schedule, reconstructs final averages from raw particles, and
disables local refinement; it is intended for repeated classification/alignment
feedback rather than final accuracy-sensitive output. It is never selected
from the input or data size. The advanced global selector retains **Fast hard
(custom)** for deliberately configured schedules.

The refinement selector defaults to continuous quadratic search, while
adaptive posterior remains available as a 2.1 compatibility choice. Appending
two fixed-class refine iterations to a fast-hard pass is not an automatic
guarantee that balanced-soft accuracy will be recovered. The workbench also exposes
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
