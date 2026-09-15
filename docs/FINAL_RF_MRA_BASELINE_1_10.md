# AlignImg 1.10 final RF/MRA baseline

AlignImg 1.10 is the frozen pre-2.0 scientific baseline. The formal real-data
path was:

- `candidate_scoring="fourier"`
- `reference_update="fourier"`
- `score_model="fourier_ncc"`
- native CUDA, batch 512, memory fraction 0.8
- 5000 CTF-corrected 70S particles, K=10, 20 iterations
- 10 annealing iterations followed by 10 iterations at the final temperature
- seeds 0, 1, and 2

All three runs completed with finite arrays and references. Responsibility row
sums had a maximum error of `1.1920928955078125e-07`; every component met the
minimum half-set occupancy for stable FRC, and no component was reseeded.

| Metric | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Runtime (s) | 429.322 | 421.900 | 501.316 |
| Particle-iterations/s | 232.925 | 237.023 | 199.475 |
| Final occupancy entropy | 0.9765 | 0.9863 | 0.9847 |
| Mean maximum responsibility | 0.5583 | 0.5534 | 0.5270 |
| Mean expected Fourier NCC | 0.2588 | 0.2598 | 0.2573 |
| Median stable FRC resolution (Å) | 11.42 | 8.43 | 8.33 |
| RELION ARI | 0.1772 | 0.1483 | 0.1283 |
| RELION NMI | 0.2193 | 0.2271 | 0.1955 |
| Optimal RELION label agreement | 34.62% | 32.22% | 29.06% |

The output references were visually and rotationally consistent across seeds:
rotation-marginalized reference NCC averaged 0.944 for seed 0 versus seed 1 and
0.939 for seed 0 versus seed 2. Hard particle partitions were not seed-stable:
assignment ARI was 0.144 and 0.131, respectively. RF assignments therefore
remain alignment/view-component diagnostics, not biological classifications.

The legacy radial-profile matching diagnostic reported correlations near
0.9996. It discards angular information and is insufficiently discriminative
for these 70S references; it must not be interpreted as particle-partition
reproducibility.

## Frozen artifacts

| Artifact | SHA-256 |
|---|---|
| `re2dc-70s-final-n5000-k10-i20.json` | `371973ef4e7aec5c5c6449688b8e3e11a726c42fee2a1b63ad33627ed0dbcb78` |
| `re2dc-70s-final-n5000-k10-i20.relion.json` | `c4b29df2833df744e409ced91b6d67b081196d2fa0e87c098cf0ea16be6d1b5f` |
| `seed-0.report.json` | `f968263353f5eea1bcdfca251a6500d1b8905fefcf093e006ea078af1d992247` |
| `seed-1.report.json` | `7d249d04f4153e9052a8bd471be1ed945d6e4f95e88a39905b2c15c68c271d71` |
| `seed-2.report.json` | `bb0239067be09f54731d21e0931af1ec5f58d25d595887c683646fb6bc0da71f` |
| `seed-0.references.mrcs` | `5bd1c4feb880ea26126c3135293b5c65a7d5fb8a767964f1fd1d79e5079dcb77` |
| `seed-1.references.mrcs` | `d8bd07a743454a49a1eec44d882d0f8372d26b238f95ad4da6727e4fdb1dcfe2` |
| `seed-2.references.mrcs` | `71b4bda1fcd78c2b991e50bc922bf209b1a021d10a38a7e397ab117c662dda30` |
| `seed-0.result.npz` | `f7402a0bce2d9f7106ec5cc4150761c47498c69631dac6c923e317d1aa94e064` |
| `seed-1.result.npz` | `fa9203d05ce969b8fba77c264393c04a2c7341311362a0afd648a1eb9fb6aaff` |
| `seed-2.result.npz` | `38417a06a961d74911947e5980db695cd8be3a95ec6b5b60e9f23d46fe4abbfa` |

The corresponding 1.10 release artifacts are:

| Artifact | SHA-256 |
|---|---|
| `dist/alignimg-1.10.0-py3-none-any.whl` | `12332e6e75c5d35cc5112cf446a6e29c4db00f10780ac7884a6cf8d3af9d36db` |
| `dist/alignimg-1.10.0.tar.gz` | `6362fa980dce66294da0439c2c4e20d35cc8131e45457957ca56056141889b44` |
| `packages/alignimg-gpu/dist/alignimg_gpu-1.10.0.tar.gz` | `8947ab5e948ebd75996ef2be79f85f2e95aa066ca1b8dddc8a216f99b7c37d0a` |

