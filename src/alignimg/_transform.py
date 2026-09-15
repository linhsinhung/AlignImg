"""Canonical AlignImg 2.x image transforms."""

from __future__ import annotations

import cv2
import numpy as np

from ._geometry import integer_center, mirror_x_integer_origin, validate_even_square
from .models import PoseSet


def transform_image(
    image: np.ndarray,
    *,
    angle_deg: float = 0.0,
    shift_y_px: float = 0.0,
    shift_x_px: float = 0.0,
    mirror: bool = False,
) -> np.ndarray:
    """Apply integer-origin mirror, CCW rotation, and shift with wrap boundaries."""
    source = np.asarray(image, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("image must be two-dimensional.")
    height, width = source.shape
    size = validate_even_square((height, width))
    if mirror:
        source = mirror_x_integer_origin(source)
    center_value = integer_center(size)
    center = (center_value, center_value)
    rotation = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    rotation[0, 2] += float(shift_x_px)
    rotation[1, 2] += float(shift_y_px)
    return cv2.warpAffine(
        source,
        rotation,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    ).astype(np.float32, copy=False)


def transform_stack(images: np.ndarray, poses: PoseSet) -> np.ndarray:
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 3:
        raise ValueError("images must have shape (N, H, W).")
    if len(poses) != len(images):
        raise ValueError("poses and images must contain the same number of items.")
    output = np.empty_like(images)
    for index in range(len(images)):
        output[index] = transform_image(
            images[index],
            angle_deg=float(poses.angle_deg[index]),
            shift_y_px=float(poses.shift_y_px[index]),
            shift_x_px=float(poses.shift_x_px[index]),
            mirror=bool(poses.mirror[index]),
        )
    return output
