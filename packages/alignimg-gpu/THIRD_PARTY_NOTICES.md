# Third-party notices

AlignImg GPU was informed by the native MRA implementation preserved in the
RE2DC GPU reference sources. That implementation is a clean extraction and
refactoring of GPU multi-reference alignment work distributed with archived
ASCEP/Cryo-RALib sources. It retains ideas including polar-ring sampling,
weighted Fourier correlation, mirror search, angular interpolation, and
alignment decoding.

The current AlignImg GPU backend is a separate, later hybrid implementation. It
combines native CUDA/C++ transform, interpolation, reduction, and accumulation
kernels with CuPy FFT/scoring and Python controller stages; it does not
redistribute the complete historical RE2DC or ASCEP MRA search engine.

Original work acknowledged by the RE2DC GPU reference notice includes:

- Fabian Schoenfeld, Max Planck Institute of Molecular Physiology (2019)
- Szu-Chi Chung, Cheng-Yu Hung, Huei-Lun Siao, and Hung-Yi Wu,
  SABID Laboratory, Academia Sinica (2020)
- Additional SPHIRE MRA contributors named in the archived ASCEP sources

The archived CUDA source files are identified as
`scipion_sinica_v3.5/ascep/cuda/gpu_aln_common.{h,cu}` and
`scipion_sinica_v3.5/ascep/cuda/gpu_aln_noref.{h,cu}`. They are licensed under
GPL-3.0-or-later.

The associated Python reference preparation ports numerical behavior from
SPHIRE's `sp_statistics.py` and `sp_user_functions.py` and from
`libEM/sparx/emdata_sparx.cpp` in EMAN2. The upstream notices name Pawel A.
Penczek, Markus Stabrin, Fabian Schoenfeld, Thorsten Wagner, Tapu Shaikh, Adnan
Ali, Luca Lusnig, and Toshio Moriya, and identify copyright held by the
University of Texas - Houston Medical School and the Max Planck Institute of
Molecular Physiology. Those sources include a GNU GPL version 2-or-later grant;
this package is distributed under GPL-3.0-or-later.

The complete reference notices are available in the AlignImg source repository
under `third_party/`. This package preserves the contributor, source-path, and
license information needed to understand that lineage without distributing the
archived reference implementations.
