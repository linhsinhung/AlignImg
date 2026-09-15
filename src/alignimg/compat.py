"""Explicit adapters from historical pose formats to AlignImg 2.x contracts."""

from __future__ import annotations

import numpy as np

from ._geometry import integer_center
from .models import PoseSet


def poses_from_legacy_params(params: np.ndarray) -> PoseSet:
    """Convert legacy ``[angle, dy, dx, ...]`` columns to a named PoseSet.

    This converts parameter storage only. It cannot make the legacy zero-border
    transform numerically equivalent to the canonical integer-origin/wrap transform.
    """
    values = np.asarray(params, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("legacy params must have shape (N, 3+).")
    return PoseSet(
        values[:, 0],
        values[:, 1],
        values[:, 2],
        np.zeros(len(values), dtype=np.bool_),
    )


def poses_to_legacy_params(poses: PoseSet, score: np.ndarray | None = None) -> np.ndarray:
    """Convert PoseSet fields to legacy ``[angle, dy, dx, score]`` storage."""
    if np.any(poses.mirror):
        raise ValueError("legacy pose arrays cannot represent mirror transforms.")
    scores = (
        np.full(len(poses), np.nan, dtype=np.float32)
        if score is None
        else np.asarray(score, dtype=np.float32)
    )
    if scores.shape != (len(poses),) or not np.all(np.isfinite(scores) | np.isnan(scores)):
        raise ValueError("score must have shape (N,) and contain finite values or NaN.")
    return np.column_stack(
        (poses.angle_deg, poses.shift_y_px, poses.shift_x_px, scores)
    ).astype(np.float32)


def convert_v1_4_poses_to_integer_center(
    poses: PoseSet, image_size: int
) -> PoseSet:
    """Convert saved v1.4 geometric-center poses to the integer-origin contract."""
    new_center = integer_center(image_size)
    old_center = (int(image_size) - 1.0) / 2.0
    old_origin = np.asarray([old_center, old_center], dtype=np.float64)
    new_origin = np.asarray([new_center, new_center], dtype=np.float64)
    converted = np.empty((len(poses), 2), dtype=np.float64)
    for index, (angle, mirrored) in enumerate(
        zip(poses.angle_deg, poses.mirror)
    ):
        radians = np.deg2rad(float(angle))
        rotation = np.asarray(
            [
                [np.cos(radians), np.sin(radians)],
                [-np.sin(radians), np.cos(radians)],
            ],
            dtype=np.float64,
        )
        old_mirror_offset = (
            np.asarray([2.0 * old_center, 0.0], dtype=np.float64)
            if bool(mirrored)
            else np.zeros(2, dtype=np.float64)
        )
        new_mirror_offset = (
            np.asarray([2.0 * new_center, 0.0], dtype=np.float64)
            if bool(mirrored)
            else np.zeros(2, dtype=np.float64)
        )
        old_shift = np.asarray(
            [poses.shift_x_px[index], poses.shift_y_px[index]], dtype=np.float64
        )
        old_offset = (
            rotation @ old_mirror_offset
            + old_origin
            - rotation @ old_origin
            + old_shift
        )
        new_base_offset = (
            rotation @ new_mirror_offset
            + new_origin
            - rotation @ new_origin
        )
        # The linear rotation/mirror term is identical in both contracts, so
        # matching the affine offsets preserves the full transform.
        converted[index] = old_offset - new_base_offset
    return PoseSet(
        poses.angle_deg,
        converted[:, 1].astype(np.float32),
        converted[:, 0].astype(np.float32),
        poses.mirror,
    )
