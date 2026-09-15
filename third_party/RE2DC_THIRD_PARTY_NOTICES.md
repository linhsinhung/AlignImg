# Third-party notices

RE2DC revision 2 uses alignment controller and polar-search behavior ported from
the archived ASCEP and SPHIRE/EMAN2 sources. Original work includes contributions
by Pawel A. Penczek, Fabian Schoenfeld, Markus Stabrin, Thorsten Wagner, Tapu
Shaikh, Adnan Ali, Luca Lusnig, Toshio Moriya, Szu-Chi Chung, Cheng-Yu Hung,
Huei-Lun Siao, and Hung-Yi Wu.

Relevant archived and upstream sources include ASCEP
`test_reffree_gpu_align.py` and `test_mref_gpu_align.py`, SPHIRE
`sp_alignment.py`, `sp_statistics.py`, `sp_user_functions.py`, and EMAN2
`util_sparx.cpp` and `emdata_sparx.cpp`. The RE2DC ports preserve applicable
GPL-2.0-or-later and GPL-3.0-or-later attribution requirements and are
distributed under GPL-3.0-or-later, without warranty. Existing upstream notices
remain in the archived source tree.

The private SciPy-batched polar candidate in v0.4 is a vectorized execution of
that same `Numrinit`/`ringwe`/`Polar2Dm`/`ormq` behavior. Its technical source is
the attributed ASCEP/SPHIRE/EMAN2 path above; it does not import or redistribute
the separate manuscript-v4 runtime implementation.

The FIt-SNE executable is not included. Its redistribution and corresponding-
source audit must be completed separately; users select an audited executable
with `RE2DC_FITSNE_EXECUTABLE`.
