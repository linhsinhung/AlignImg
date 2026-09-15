"""Canonical discrete image geometry for AlignImg workflows."""

from __future__ import annotations

import numpy as np


CENTER_CONVENTION = "integer-origin-floor-N-over-2"


def integer_center(size: int) -> float:
    """Return the RELION-compatible origin of a supported image axis."""
    value = int(size)
    if value < 2 or value % 2:
        raise ValueError("AlignImg alignment workflows require an even image size.")
    return float(value // 2)


def validate_even_square(shape: tuple[int, int]) -> int:
    """Validate a square even image shape and return its size."""
    if len(shape) != 2 or int(shape[0]) != int(shape[1]):
        raise ValueError("AlignImg alignment workflows require square images.")
    size = int(shape[0])
    integer_center(size)
    return size


def mirror_x_integer_origin(image: np.ndarray) -> np.ndarray:
    """Reflect x about the periodic integer origin ``N//2``."""
    source = np.asarray(image)
    validate_even_square(tuple(source.shape))
    return np.roll(source[:, ::-1], 1, axis=1)
