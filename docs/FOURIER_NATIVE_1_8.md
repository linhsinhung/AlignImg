# AlignImg 1.8 Fourier-native candidate scoring

## Scope

This stage introduces conformance-tested Fourier rotation, translation, mirror, and
candidate-NCC primitives. Native CUDA and CuPy passed the isolated Linux/NVIDIA
conformance gate, so the scorer is connected to RF/global/refine through the explicit
`AlignmentConfig(candidate_scoring="fourier")` option. Production workflow and
RELION-defined A/B validation have passed. The default remains `"raster"` for
backward-comparable runs.

The backward-compatible raster path is:

```text
spatial particle -> raster rotation -> FFT -> translation phase -> Fourier NCC
```

The Fourier-native production option is:

```text
cached particle FFT -> Fourier rotation interpolation
                    -> translation phase -> Fourier NCC
```

No per-candidate FFT is performed by the experimental scorer.

## Integer-center transform

Let `F(k)` be the unshifted DFT and `c=(N/2,N/2)` the integer spatial origin. The
centered spectrum is

```text
Fc(k) = exp(+i 2 pi k.c / N) F(k).
```

For the inverse spatial map `A` used by AlignImg rotation/mirror, the transformed
centered spectrum is sampled at `A^-T k`. Rotation and reflection are orthogonal, so
`A^-T=A`. Translation `(dy,dx)` then contributes

```text
exp(-i 2 pi (ky dy + kx dx) / N).
```

The implementation bilinearly interpolates the real and imaginary parts of `Fc` on
the periodic DFT grid, converts back to the unshifted convention, and applies the
translation phase. For even boxes, the integer-center conversion is the exact
checkerboard phase `(-1)^(ky+kx)`.

Identity, pure Fourier translation, and integer-origin mirror are exact up to floating
point error. Arbitrary rotations differ from the raster baseline because interpolation
is performed in frequency rather than real space. Conformance is therefore measured
inside the configured scoring band, not by requiring pixelwise equality.

## Implementations

- CPU: NumPy authoritative primitive in `alignimg._fourier_native`.
- CuPy: runtime-compiled complex bilinear transform plus device NCC reduction.
- CUDA: compiled persistent `TransformSession` complex transform plus CuPy NCC
  reduction.

The GPU candidate scorers use the existing memory planner and split flat candidate
buffers into bounded batches. Particle DFTs are cached on the device only when the
whole cache fits inside the configured budget while preserving the VRAM reserve;
otherwise they are host-streamed. Posterior reduction, workflow control, and the
real-space M-step remain unchanged.

## Conformance gate

Run:

```bash
python tools/server_validation.py \
  --suite quick \
  --backend cuda \
  --only fourier_native_conformance \
  --batch-size 512 \
  --output validation-results/alignimg-1.8-fourier-native-cuda.json

python tools/server_validation.py \
  --suite quick \
  --backend cupy \
  --only fourier_native_conformance \
  --batch-size 512 \
  --output validation-results/alignimg-1.8-fourier-native-cupy.json
```

The test covers identity, subpixel translation, mirror, arbitrary positive/negative
rotations, and combined poses on centered 128-pixel cryo-EM-like images. It requires:

- minimum Fourier-native/raster NCC of 0.995 within 0.35 cycles/pixel;
- GPU/CPU relative transform error no greater than `3e-5`;
- GPU/CPU candidate-score absolute error no greater than `2e-5`;
- finite JSON metrics.

The isolated CUDA and CuPy gate passed on the RTX 3090 host with transform relative-L2
error `8.65e-8`, score maximum-absolute error `5.96e-8`, and minimum raster-band NCC
`0.999755`. The next gate is production workflow A/B:

```bash
python tools/server_validation.py \
  --suite quick \
  --backend cuda \
  --only fourier_workflow_ab \
  --batch-size 512 \
  --output validation-results/alignimg-1.8-fourier-workflow-cuda.json

python tools/server_validation.py \
  --suite quick \
  --backend cupy \
  --only fourier_workflow_ab \
  --batch-size 512 \
  --output validation-results/alignimg-1.8-fourier-workflow-cupy.json
```

After that passes, use `tools/re2dc_70s_pose_benchmark.py --candidate-scoring
fourier` for the 384-particle RELION-defined pose benchmark. The frozen 1.7.1
results remain the raster comparison baseline.

## Accepted real-data results

The 384-particle RELION-defined fixed-MRA benchmark also passed the comparison
contract:

| Metric | 1.7.1 raster | 1.8 Fourier-native |
| --- | ---: | ---: |
| adaptive fixed-MRA runtime | 68.401 s | 43.789 s |
| initializer + refinement runtime | 71.574 s | 46.604 s |
| angle errors above 11.25 degrees | 8 / 384 | 9 / 384 |
| shift errors above 2 px | 14 / 384 | 13 / 384 |
| median angle error | 2.625 degrees | 2.8125 degrees |
| median shift error | 0.6681 px | 0.6647 px |
| mean aligned-particle NCC | 0.236407 | 0.236069 |
| median reliable stable FRC cutoff | 0.218726 | 0.218160 cycles/pixel |

All deterministic repeats were exact. Independent K=1 and joint fixed-MRA results
were identical within each class, including poses, responsibilities, and references.
The two engines did not fail on the same individual outliers, but the total tail size
remained effectively unchanged; this is accepted as interpolation-dependent basin
selection rather than justification for more rescue constraints. The 1.8 scorer is
therefore scientifically accepted as an explicit production option.

The production workflow gate passed on both GPU engines. Relative to the raster
path on the same 16-particle synthetic workload, Fourier scoring measured:

| Workflow | Native CUDA speedup | CuPy speedup | Maximum angle delta | Minimum reference correlation |
| --- | ---: | ---: | ---: | ---: |
| global proposal | 7.08x | 6.51x | 0.0 degrees | 0.999631 |
| adaptive refine | 1.39x | 1.38x | 1.5 degrees | 0.999521 |

Both engines preserved fixed-class assignments, kept responsibility MAE below
`1.7e-7`, and selected the device particle-DFT cache while preserving about
5.06 GB of the configured VRAM reserve. Reports:

- `validation-results/alignimg-1.8-fourier-workflow-cuda-final.json`
  (`sha256:cd51b7cd06495ed953baa007ed3280551431a4fc175b92190dc60890c1be6999`)
- `validation-results/alignimg-1.8-fourier-workflow-cupy-final.json`
  (`sha256:432ae276479c89642b6eca11f8170cd9768f76e2c20909b73e0943a29b231527`)

The workflow gate also matches raster and Fourier top-L candidates one-to-one after
accounting for the accepted 3-degree and one-pixel pose tolerance. It requires full
top-L support matching, mean posterior total variation no greater than 0.15, maximum
per-particle total variation no greater than 0.40, and matched candidate-score error
no greater than 0.05. Candidate order is not part of the contract. Both final GPU
reports matched every top-L candidate. Global mean/maximum posterior total variation
was `0.1141/0.3572`; adaptive refinement measured `0.0182/0.0318`. AlignImg 1.8 is
therefore closed as the accepted Fourier-native candidate-scoring baseline.

## Frozen 1.8 artifacts

The final local suite passed with 128 tests and 10 GPU-only skips. The Linux RTX 3090
host then passed the native-CUDA and CuPy primitive gates, the final workflow gates,
and the RELION-defined 384-particle comparison. The frozen artifacts are:

| Artifact | SHA-256 |
| --- | --- |
| `dist/alignimg-1.8.0-py3-none-any.whl` | `96111215bcfa4c13a8fe74762211550c85c8446f2927b4e915509b916a35a092` |
| `dist/alignimg-1.8.0.tar.gz` | `4bd04a45156239b622131ee7d52d3c394caddb6e74705085bc586edf19fd86ca` |
| `packages/alignimg-gpu/dist/alignimg_gpu-1.8.0.tar.gz` | `22e9b7e8f053724a1b703a8bee1925487573be7e4d366e00d37aa5084aa837f5` |
| Native CUDA primitive report | `0a635ec09a4412045e1cc1385c869b9e65c9c6e1f3e553104fcacf669b624e82` |
| CuPy primitive report | `16379583752b7592be8b6817ab5e45e9f1e0887b8d179de7581dedd803e024fe` |
| Native CUDA final workflow report | `cd51b7cd06495ed953baa007ed3280551431a4fc175b92190dc60890c1be6999` |
| CuPy final workflow report | `432ae276479c89642b6eca11f8170cd9768f76e2c20909b73e0943a29b231527` |
| RELION-defined pose report | `058caf28d5b1265797808b7180affd138db735fbe7166b64a2d5b4f33cbfd9ef` |

These files define the 1.8 comparison baseline. Further changes to candidate scoring,
reference update, or GPU residency require a new version and must not overwrite these
artifacts.

## Deliberate limitations

- Bilinear Fourier interpolation is initially evaluated on approximately centered,
  externally prepared particles, matching the intended cryo-EM input contract.
- No new outlier, rescue, centering, or class-management rule is added.
- No CTF likelihood or whitening is introduced in this stage.
- The M-step remains real-space, so this stage alone does not yet provide a completely
  Fourier-resident iteration.
