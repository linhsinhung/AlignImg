"""Integer-origin transform contract tests."""

from __future__ import annotations

import numpy as np
import pytest

import alignimg as ai
from alignimg._engine import _center_reference
from alignimg._geometry import (
    CENTER_CONVENTION,
    integer_center,
    mirror_x_integer_origin,
)
from alignimg._transform import transform_image
from alignimg_gpu.backend import _TRANSFORM_KERNEL


def affine_for_pose(
    *, angle_deg: float, shift_y: float, shift_x: float, mirrored: bool, center: float
) -> np.ndarray:
    radians = np.deg2rad(angle_deg)
    rotation = np.asarray(
        [
            [np.cos(radians), np.sin(radians)],
            [-np.sin(radians), np.cos(radians)],
        ]
    )
    mirror = np.asarray([[-1.0, 0.0], [0.0, 1.0]]) if mirrored else np.eye(2)
    mirror_offset = np.asarray([2.0 * center, 0.0]) if mirrored else np.zeros(2)
    origin = np.asarray([center, center])
    matrix = np.eye(3)
    matrix[:2, :2] = rotation @ mirror
    matrix[:2, 2] = (
        rotation @ mirror_offset
        + origin
        - rotation @ origin
        + np.asarray([shift_x, shift_y])
    )
    return matrix


def test_integer_center_matches_even_shifted_dft_origin():
    assert integer_center(16) == 8.0
    shifted_dc = np.unravel_index(
        np.argmax(np.abs(np.fft.fftshift(np.fft.fft2(np.ones((16, 16)))))),
        (16, 16),
    )
    assert shifted_dc == (8, 8)
    with pytest.raises(ValueError, match="even image size"):
        integer_center(15)


def test_rotation_keeps_integer_origin_fixed_and_rotates_marker_about_it():
    image = np.zeros((16, 16), dtype=np.float32)
    image[8, 8] = 2.0
    image[8, 11] = 1.0

    rotated = transform_image(image, angle_deg=90.0)

    assert rotated[8, 8] == pytest.approx(2.0)
    assert np.unravel_index(np.argmax(rotated == 1.0), rotated.shape) == (5, 8)


def test_periodic_mirror_is_about_integer_origin_and_is_an_involution():
    image = np.zeros((16, 16), dtype=np.float32)
    image[4, 2] = 1.0

    mirrored = mirror_x_integer_origin(image)

    assert np.unravel_index(np.argmax(mirrored), mirrored.shape) == (4, 14)
    assert np.array_equal(mirror_x_integer_origin(mirrored), image)
    assert np.array_equal(transform_image(mirrored, mirror=True), image)


def test_reference_centering_targets_integer_origin():
    reference = np.zeros((16, 16), dtype=np.float32)
    reference[3, 5] = 1.0

    centered, shift_y, shift_x = _center_reference(reference)

    assert (shift_y, shift_x) == (5.0, 3.0)
    assert np.unravel_index(np.argmax(centered), centered.shape) == (8, 8)


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("angle", [0.0, 30.0, 90.0, 173.0])
def test_v1_4_pose_adapter_preserves_continuous_affine_transform(angle, mirrored):
    old = ai.PoseSet(
        np.asarray([angle]),
        np.asarray([2.25]),
        np.asarray([-1.5]),
        np.asarray([mirrored]),
    )
    new = ai.convert_v1_4_poses_to_integer_center(old, 16)

    old_matrix = affine_for_pose(
        angle_deg=angle,
        shift_y=2.25,
        shift_x=-1.5,
        mirrored=mirrored,
        center=7.5,
    )
    new_matrix = affine_for_pose(
        angle_deg=float(new.angle_deg[0]),
        shift_y=float(new.shift_y_px[0]),
        shift_x=float(new.shift_x_px[0]),
        mirrored=mirrored,
        center=8.0,
    )
    assert np.allclose(new_matrix, old_matrix, atol=1e-6)


def test_odd_boxes_are_rejected_by_transform_and_alignment():
    image = np.zeros((15, 15), dtype=np.float32)
    with pytest.raises(ValueError, match="even image size"):
        ai.transform_images(image[None], ai.PoseSet.identity(1))
    with pytest.raises(ValueError, match="even image size"):
        ai.align_to_references(image[None], image)


def test_result_metadata_records_integer_origin_contract():
    image = np.zeros((16, 16), dtype=np.float32)
    image[8, 8] = 1.0
    result = ai.align_to_references(
        image[None],
        image,
        config=ai.AlignmentConfig(
            max_iterations=1,
            top_l=1,
            angle_samples=12,
            proposal_angles_per_reference=1,
            translation_range=0,
            halfset_diagnostics=False,
            center_references=False,
        ),
    )
    assert result.metadata["center_convention"] == CENTER_CONVENTION
    assert "integer-origin" in result.metadata["transform_convention"]


def test_cupy_kernel_declares_same_integer_origin_and_mirror_axis():
    assert "const float center = size * 0.5f;" in _TRANSFORM_KERNEL
    assert "source_x = 2.0f * center - source_x;" in _TRANSFORM_KERNEL
    assert "(size - 1) * 0.5f" not in _TRANSFORM_KERNEL
