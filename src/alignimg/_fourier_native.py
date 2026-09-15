"""Experimental Fourier-native transforms and candidate scoring.

These primitives are kept out of the production workflow until their raster
conformance is validated on the Linux/CUDA reference host.
"""

from __future__ import annotations

import numpy as np

from ._fourier import _fourier_ncc, _shift_fourier
from ._geometry import validate_even_square
from ._profiling import count as profile_count, profile_stage
from ._frequency_tables import frequency_tables


def _periodic_bilinear_complex(
    values: np.ndarray,
    source_y: np.ndarray,
    source_x: np.ndarray,
) -> np.ndarray:
    size = values.shape[0]
    floor_y = np.floor(source_y)
    floor_x = np.floor(source_x)
    y0 = floor_y.astype(np.int64) % size
    x0 = floor_x.astype(np.int64) % size
    y1 = (y0 + 1) % size
    x1 = (x0 + 1) % size
    weight_y = source_y - floor_y
    weight_x = source_x - floor_x
    top = values[y0, x0] * (1.0 - weight_x) + values[y0, x1] * weight_x
    bottom = values[y1, x0] * (1.0 - weight_x) + values[y1, x1] * weight_x
    return top * (1.0 - weight_y) + bottom * weight_y


@profile_stage("fourier_transform_cpu")
def transform_fourier_cpu(
    fourier: np.ndarray,
    *,
    angle_deg: float = 0.0,
    shift_y_px: float = 0.0,
    shift_x_px: float = 0.0,
    mirror: bool = False,
) -> np.ndarray:
    """Transform one unshifted DFT about AlignImg's integer spatial origin."""
    source = np.asarray(fourier, dtype=np.complex64)
    if source.ndim != 2:
        raise ValueError("fourier must be two-dimensional")
    size = validate_even_square(tuple(source.shape))

    _, _, output_y, output_x, center_phase = frequency_tables(size)
    radians = np.deg2rad(float(angle_deg))
    cosine = np.cos(radians)
    sine = np.sin(radians)
    source_y = cosine * output_y + sine * output_x
    source_x = cosine * output_x - sine * output_y
    if mirror:
        source_x = -source_x

    centered_source = source * center_phase
    rotated_centered = _periodic_bilinear_complex(
        centered_source,
        source_y % size,
        source_x % size,
    )
    rotated = rotated_centered * center_phase
    shifted = _shift_fourier(rotated, float(shift_y_px), float(shift_x_px))
    return np.asarray(shifted, dtype=np.complex64)


def score_fourier_candidates_cpu(
    particle_fourier: np.ndarray,
    reference_fourier: np.ndarray,
    cells: np.ndarray,
    frequency_mask: np.ndarray,
    *, reference_norms: np.ndarray | None = None,
) -> np.ndarray:
    """Score flat pose/reference cells without returning to real space."""
    particle = np.asarray(particle_fourier, dtype=np.complex64)
    references = np.asarray(reference_fourier, dtype=np.complex64)
    values = np.asarray(cells)
    if particle.ndim != 2:
        raise ValueError("particle_fourier must be two-dimensional")
    if references.ndim != 3 or references.shape[1:] != particle.shape:
        raise ValueError("reference_fourier must have shape (K, H, H)")
    required = {
        "reference_index",
        "angle_deg",
        "shift_y_px",
        "shift_x_px",
        "mirror",
    }
    if (
        values.ndim != 1
        or values.dtype.names is None
        or not required.issubset(values.dtype.names)
    ):
        raise ValueError("cells must be a one-dimensional structured pose array")

    if reference_norms is None:
        reference_norms = np.asarray([
            float(np.sum(frequency_mask * np.abs(reference) ** 2))
            for reference in references
        ])
        profile_count("reference_norm_evaluations", len(references))
    scores = np.empty(len(values), dtype=np.float32)
    rotated_cache: dict[tuple[float, bool], np.ndarray] = {}
    for index, cell in enumerate(values):
        key = (round(float(cell["angle_deg"]), 7), bool(cell["mirror"]))
        rotated = rotated_cache.get(key)
        if rotated is None:
            rotated = transform_fourier_cpu(
                particle,
                angle_deg=float(cell["angle_deg"]),
                mirror=bool(cell["mirror"]),
            )
            rotated_cache[key] = rotated
        shifted = _shift_fourier(
            rotated,
            float(cell["shift_y_px"]),
            float(cell["shift_x_px"]),
        )
        scores[index] = _fourier_ncc(
            shifted,
            references[int(cell["reference_index"])],
            frequency_mask,
            reference_norms[int(cell["reference_index"])],
        )
    return scores
