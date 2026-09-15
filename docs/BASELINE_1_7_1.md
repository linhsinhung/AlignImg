# AlignImg 1.7.1 frozen baseline

本文件固定 Fourier-native engine 開發前的 raster-transform 科學與工程基準。
1.7.1 從此只作比較基準，不再調整 rescue、pose search 或 reference update。
任何演算法變更都必須使用新的版本號，且不得覆寫本文件列出的 artifacts。

## Frozen implementation

- AlignImg：1.7.1
- alignimg-gpu：1.7.1
- transform center：`integer-origin-floor-N-over-2`
- reference-free/global search：polar proposal + Cartesian Fourier NCC
- refinement search：adaptive posterior + bounded one-pass rescue
- rescue policy：per-component 5% cap、Fourier-NCC improvement gate、final local confirmation
- CTF policy：alignment 輸入已由外部 preprocessing 完成 CTF correction

### Distribution SHA-256

| Artifact | SHA-256 |
| --- | --- |
| `dist/alignimg-1.7.1-py3-none-any.whl` | `60e241a35b10260f164cdf283767d9f8c41673c8b6e48110d9548088141c6e5b` |
| `dist/alignimg-1.7.1.tar.gz` | `6ac1b66cb2bd81cdc0805fbc771add3f8c480645b790b283f00aa1bb86efb428` |
| `packages/alignimg-gpu/dist/alignimg_gpu-1.7.1.tar.gz` | `3dc697e609edc333f426322ee3c1ad61287b494953e3b52952c0a4d5c9e6973f` |

這三個 archives 是 1.7.1 source snapshot。後續若 checksum 改變，該內容不得仍稱為
1.7.1 baseline。

## Validation status

| Check | Result | Artifact |
| --- | --- | --- |
| Full local pytest | PASS: 117 passed, 4 GPU-only skipped | local run, 2026-09-01 |
| CPU quick suite | PASS: 12/12 | `validation-results/alignimg-1.7.1-cpu-quick.json` |
| CUDA adaptive-rescue smoke | PASS: 1/1 | `validation-results/alignimg-1.7.1-adaptive-rescue-cuda.json` |
| RELION-defined 384-particle pose benchmark | PASS | `validation-results/re2dc-70s-pose-rescue-1.7.1.json` |
| Full CUDA quick suite | PASS: 12/12 | `validation-results/alignimg-1.7.1-cuda-quick.json` |
| Full CuPy quick suite | PASS: 12/12 | `validation-results/alignimg-1.7.1-cupy-quick.json` |

### Validation artifact SHA-256

| Artifact | SHA-256 |
| --- | --- |
| CPU quick JSON | `d76ced3b643c55d40f23a7e499efdc2bb92a4232f1349f82f57da052fb288e31` |
| CUDA rescue JSON | `40729e253ed2390bf0ac4e541e520bbadbe92b49647602bee0a453ff99d16c17` |
| CUDA quick JSON | `3c0de5c49e3f255e6af49d83f79b5589c40731ff4d2b6588c7b9316c339ce51d` |
| CuPy quick JSON | `d592d8336dbc2044eaa13b6d1c81e0da7e5f2d78813c6a619709db91d94eda18` |
| Pose benchmark JSON | `96fe796177f431c1a1707858e73f306a9efa38ccea3f487feed28e2a3e8c344a` |
| Fixed-MRA result NPZ | `535e2f1d55456b775ca26a706df870264c77aa698c4f137dd272ae8b74e39e8e` |
| Fixed-MRA references MRCS | `3273da96e6da939f1b2dfd82d39384026049a85a7ef9432d4e6ad1bc19998495` |

## Frozen scientific measurements

384 particles are split equally across three RELION-defined homogeneous classes.
Class labels are fixed during alignment and RELION poses are used only for evaluation.

| Fixed-MRA measurement | 1.7.1 baseline |
| --- | ---: |
| assignment accuracy | 100% |
| angle error <= 2.8125 deg | 84.375% |
| angle error <= 5.625 deg | 97.1354% |
| angle error <= 11.25 deg | 97.9167% |
| shift error <= 1 px | 78.9063% |
| shift error <= 2 px | 96.3542% |
| mean reference/oracle correlation | 0.9078585 |
| mean aligned-particle NCC | 0.2364072 |
| median reliable stable FRC cutoff | 0.2187255 cycles/pixel |
| runtime, RTX 3090 | 68.4012 s |

Rescue reviewed 21/384 particles, rejected 20, accepted one, and left the final
iteration for local confirmation. Independent K=1 runs and the joint fixed-MRA run
produced identical poses, uncertainty arrays, and references. A deterministic repeat
produced zero pose/responsibility differences and unit reference correlations.

Known residuals are part of the frozen baseline, not immediate optimization targets:

- 8/384 particles have angle error above 11.25 degrees.
- 14/384 particles have shift error above 2 px.
- `max_adaptive_cells=32` caps every particle in this benchmark; selected coarse
  posterior mass rises from approximately 0.58 to 0.64 across the three iterations.

## Server closure commands

Run these commands against the frozen distributions and do not use `--resume` with an
older report:

```bash
python tools/server_validation.py \
  --suite quick \
  --backend cuda \
  --batch-size 512 \
  --output validation-results/alignimg-1.7.1-cuda-quick.json

python tools/server_validation.py \
  --suite quick \
  --backend cupy \
  --batch-size 512 \
  --output validation-results/alignimg-1.7.1-cupy-quick.json
```

Both reports identify AlignImg and alignimg-gpu 1.7.1, record 12/12 passing cases,
contain finite metrics, and satisfy every case's existing CPU/backend tolerance. CUDA
and CuPy both reproduce the CPU global result with identical poses and assignments,
reference correlation effectively equal to one, and responsibility MAE of
`1.49e-8`. Batch 512 is honored and the lowest recorded free VRAM is approximately
23.63 GB on the 24 GB RTX 3090 host. The 1.7.1 baseline is closed.

## Comparison contract for the next engine

The Fourier-native candidate engine must be compared with this baseline before it can
replace raster rotation. At minimum it must preserve:

- the integer-center pose convention;
- deterministic repeatability for deterministic modes;
- K=1 versus fixed-MRA equivalence;
- finite normalized responsibilities;
- the existing CPU/CUDA/CuPy scientific tolerances;
- VRAM-bounded batching;
- the 384-particle threshold counts without a broad regression.

The remaining few-percent pose outliers are not a reason to add recursive rescue or
additional constraints to the Fourier-native engine.
