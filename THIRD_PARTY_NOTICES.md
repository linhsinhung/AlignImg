# Third-party notices

AlignImg 2.x incorporates algorithmic concepts and implementation work informed
by or derived from the attributed GPL-compatible RE2DC reference sources in the
source repository. Their lineage includes ASCEP/Cryo-RALib CUDA alignment work
and Python/C++ behavior from EMAN2, SPARX, and SPHIRE. The current AlignImg GPU
backend is a later hybrid implementation combining native CUDA/C++ kernels with
CuPy FFT/scoring and Python controller stages; it is not the complete historical
RE2DC or ASCEP MRA search engine.

Detailed upstream contributors, source paths, copyright holders, and licensing
terms are recorded in the RE2DC and RE2DC GPU notices distributed with this
package and, in the source repository, under `Reference/re2dc/` and
`Reference/re2dc-gpu/`.

RELION provided scientific inspiration for AlignImg's MAP/Bayesian pose-search
design. No RELION source code is distributed as part of AlignImg, and AlignImg
does not implement RELION's complete CTF, noise, likelihood, or regularized-
reconstruction model.

AlignImg uses NumPy, SciPy, and OpenCV. Their licenses are provided by their
respective distributions and are not changed by this notice.
