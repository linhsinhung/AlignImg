# AlignImg acknowledgements and source lineage

This document separates scientific inspiration, attributed implementation
lineage, and later AlignImg development. It is a provenance map, not a claim
that every upstream method is reproduced in full.

## Lineage map

| Project or source | Relationship to AlignImg | What is carried forward | What is not claimed | Repository evidence |
| --- | --- | --- | --- | --- |
| RELION | Scientific and methodological inspiration | Bayesian/MAP ideas for pose uncertainty, priors, iterative reference refinement, and low-SNR alignment | AlignImg does not contain RELION code or reproduce its complete CTF, noise, likelihood, or regularized-reconstruction model | [`README.md`](../README.md), [`docs/API.md`](API.md) |
| EMAN2, SPARX, and SPHIRE | Scientific inspiration and attributed implementation lineage through RE2DC | Polar alignment behavior and supporting Python/C++ numerical routines identified in the upstream notices | AlignImg is not presented as a complete EMAN2/SPHIRE implementation | [`Reference/re2dc/THIRD_PARTY_NOTICES.md`](../Reference/re2dc/THIRD_PARTY_NOTICES.md), [`Reference/re2dc-gpu/THIRD_PARTY_NOTICES.md`](../Reference/re2dc-gpu/THIRD_PARTY_NOTICES.md) |
| ASCEP, Cryo-RALib, and GPU ISAC lineage | Historical RF/MRA and CUDA technical lineage through RE2DC | Polar-ring search concepts and archived native CUDA alignment work | The current AlignImg GPU backend is not the complete historical ASCEP or RE2DC search engine | [`Reference/re2dc-gpu/THIRD_PARTY_NOTICES.md`](../Reference/re2dc-gpu/THIRD_PARTY_NOTICES.md), [Cryo-RALib paper](https://arxiv.org/abs/2011.05755) |
| RE2DC | Attributed reference implementation and accumulated validation experience | Standalone RF/MRA behavior, GPU implementation experience, and source-level provenance used while developing AlignImg | This document does not assign sole leadership or ownership of the collaborative RE2DC/ASCEP research lineage | [`Reference/re2dc/`](../Reference/re2dc/), [`Reference/re2dc-gpu/`](../Reference/re2dc-gpu/) |
| Earlier AlignImg | Direct predecessor maintained by Hsin-Hung Lin | Robust MAP-style single-reference iteration, pose priors, posterior search, and an API for arrays and MRC/MRCS stacks | Historical Git sources do not establish early STAR metadata support | Current Git history and [`docs/API.md`](API.md) |
| Current AlignImg | Present implementation developed and evaluated by Hsin-Hung Lin | Unified RF/MRA/refinement workflows, Fourier-domain scoring and reference updates, continuous quadratic refinement, class feedback, and bounded-memory CPU/CuPy/CUDA execution | MRA assignments are not claimed to be validated biological classifications | [`README.md`](../README.md), [`docs/RELEASE_FREEZE_2_2.md`](RELEASE_FREEZE_2_2.md) |

In compact form, the implementation history is:

```text
EMAN2 / SPARX / SPHIRE ─┐
ASCEP / Cryo-RALib CUDA ├─> attributed RE2DC reference work ─┐
GPU ISAC lineage ───────┘                                    ├─> current AlignImg
earlier AlignImg robust MAP-style API ───────────────────────┘

RELION Bayesian/MAP ideas ──> scientific influence, not code lineage
```

## Author and AI-assisted development

Hsin-Hung Lin directed the design and evaluation of AlignImg and is responsible
for the resulting software and its claims. AI tools assisted with source
analysis, comparison, porting, implementation, experimentation, documentation,
and validation. AI assistance does not replace attribution to the people and
projects whose work appears in the lineage above.

## Distribution records

The repository-level [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)
summarizes external provenance. More detailed contributor, source-path,
copyright, and licensing records are preserved in the notices under
[`Reference/re2dc/`](../Reference/re2dc/) and
[`Reference/re2dc-gpu/`](../Reference/re2dc-gpu/).

Core and GPU source distributions include the applicable notices as package
license files so that this information remains available outside a source
checkout.
