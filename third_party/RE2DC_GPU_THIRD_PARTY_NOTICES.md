# Third-party notices

The native MRA implementation is a clean extraction and refactoring of the GPU
multi-reference alignment method distributed with the archived ASCEP/Cryo-RAlib
sources in this repository. It retains the polar-ring sampling, weighted Fourier
correlation, mirror search, angular interpolation, and alignment decoding method.

Original work:

- Fabian Schoenfeld, Max Planck Institute of Molecular Physiology (2019)
- Szu-Chi Chung, Cheng-Yu Hung, Huei-Lun Siao, and Hung-Yi Wu,
  SABID Laboratory, Academia Sinica (2020)
- Additional SPHIRE MRA contributors named in
  `scipion_sinica_v3.5/ascep/test_mref_gpu_align.py`

The archived source files are
`scipion_sinica_v3.5/ascep/cuda/gpu_aln_common.{h,cu}` and
`scipion_sinica_v3.5/ascep/cuda/gpu_aln_noref.{h,cu}`. They are licensed under
GPL-3.0-or-later. The rewritten files in this package remain under the same terms.

The Python reference preparation ports the numerical sequence of SPHIRE's
`fsc`, `fsc_mask`, `fit_tanh`, `filt_tanl`, `amoeba`, and `ref_ali2d` routines,
plus the two-dimensional `EMData::phase_cog` method. The relevant upstream files
are `sphire/sphire/libpy/sp_statistics.py`,
`sphire/sphire/libpy/sp_user_functions.py`, and
`libEM/sparx/emdata_sparx.cpp` in the EMAN2 repository.

The SPHIRE user-function source names Pawel A. Penczek (2006), Markus Stabrin,
Fabian Schoenfeld, Thorsten Wagner, Tapu Shaikh, Adnan Ali, Luca Lusnig, and
Toshio Moriya (2019), with copyright held by the University of Texas - Houston
Medical School (2000-2006) and Max Planck Institute of Molecular Physiology
(2019). The `EMData::phase_cog` source names Pawel A. Penczek (2006), with
copyright held by the University of Texas - Houston Medical School (2000-2019).

Those upstream files are offered under a joint BSD/GNU notice and include a GNU
GPL version 2-or-later grant. This derivative optional package is distributed
under GPL-3.0-or-later. The upstream notices require existing author and copyright
attribution to be preserved; the names, dates, copyright holders, source paths,
and license choice above form part of this notice. The software is provided
without warranty, including without implied warranties of merchantability or
fitness for a particular purpose.
