from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

import alignimg as ai
from alignimg._fourier import soft_circular_mask
from alignimg import workflows


def asymmetric_reference(size: int = 24, offset: int = 0) -> np.ndarray:
    y, x = np.indices((size, size), dtype=np.float32)
    image = np.exp(-((y - (7 + offset)) ** 2 + (x - 15) ** 2) / 6.0)
    image += 0.6 * np.exp(-((y - 17) ** 2 + (x - (8 + offset)) ** 2) / 3.0)
    return image.astype(np.float32)


def config(*, apply_final_pose_to_raw: bool) -> ai.AlignmentConfig:
    return ai.AlignmentConfig(
        max_iterations=1,
        top_l=2,
        angle_samples=24,
        proposal_angles_per_reference=2,
        translation_range=1.0,
        halfset_diagnostics=False,
        center_references=False,
        robust_weighting=False,
        batch_size=2,
        apply_final_pose_to_raw=apply_final_pose_to_raw,
    )


def expected_hard_average(
    images: np.ndarray, result: ai.AlignmentResult, reference_count: int
) -> np.ndarray:
    aligned = ai.transform_images(images, result.poses, backend="cpu")
    output = np.zeros((reference_count, *images.shape[1:]), dtype=np.float32)
    for reference_index in range(reference_count):
        selected = result.reference_assignments == reference_index
        weights = result.inlier_weights[selected].astype(np.float64)
        output[reference_index] = (
            np.tensordot(weights, aligned[selected].astype(np.float64), axes=(0, 0))
            / weights.sum()
        )
    return output * soft_circular_mask(images.shape[-1], None, 4.0)[None]


def test_default_class_averages_remain_soft_references():
    reference = asymmetric_reference()
    images = np.repeat(reference[None], 3, axis=0)
    result = ai.align_to_references(
        images,
        reference,
        config=config(apply_final_pose_to_raw=False),
        backend="cpu",
    )

    assert np.array_equal(result.class_averages, result.references)
    assert result.metadata["class_average_estimator"] == "soft_posterior_reference"


def test_raw_class_average_does_not_replace_soft_reference_or_pose_result():
    reference = asymmetric_reference()
    source = ai.PoseSet(
        angle_deg=np.asarray([0.0, 15.0, -20.0], dtype=np.float32),
        shift_y_px=np.asarray([0.0, 1.0, -1.0], dtype=np.float32),
        shift_x_px=np.asarray([0.0, -1.0, 1.0], dtype=np.float32),
        mirror=np.zeros(3, dtype=np.bool_),
    )
    images = ai.transform_images(np.repeat(reference[None], 3, axis=0), source)
    soft = ai.align_to_references(
        images,
        reference,
        config=config(apply_final_pose_to_raw=False),
        backend="cpu",
    )
    raw = ai.align_to_references(
        images,
        reference,
        config=config(apply_final_pose_to_raw=True),
        backend="cpu",
    )

    assert np.array_equal(raw.references, soft.references)
    assert np.array_equal(raw.poses.angle_deg, soft.poses.angle_deg)
    assert np.array_equal(raw.poses.shift_y_px, soft.poses.shift_y_px)
    assert np.array_equal(raw.poses.shift_x_px, soft.poses.shift_x_px)
    assert np.allclose(raw.class_averages, expected_hard_average(images, raw, 1))
    assert raw.metadata["class_average_estimator"] == (
        "final_map_pose_inlier_weighted_raw"
    )
    assert raw.metadata["class_average_batch_size"] == 2


def test_raw_class_average_uses_final_hard_class_assignments():
    references = np.stack(
        [asymmetric_reference(offset=0), asymmetric_reference(offset=2)]
    )
    images = np.stack([references[0], references[0], references[1], references[1]])
    assignments = np.asarray([0, 0, 1, 1], dtype=np.int32)
    priors = ai.make_class_priors(assignments=assignments, n_components=2, trust=1.0)
    result = ai.align_to_references(
        images,
        references,
        class_priors=priors,
        config=replace(config(apply_final_pose_to_raw=True), batch_size=3),
        backend="cpu",
    )

    assert np.array_equal(result.reference_assignments, assignments)
    assert np.allclose(result.class_averages, expected_hard_average(images, result, 2))


def test_gpu_raw_class_average_delegates_without_host_aligned_stack(monkeypatch):
    images = np.arange(4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
    references = np.zeros((2, 8, 8), dtype=np.float32)
    poses = ai.PoseSet.identity(4)
    assignments = np.asarray([0, 1, 0, 1], dtype=np.int32)
    weights = np.asarray([1.0, 0.8, 0.6, 0.4], dtype=np.float32)
    result = SimpleNamespace(
        references=references,
        poses=poses,
        reference_assignments=assignments,
        inlier_weights=weights,
        metadata={},
        _class_average_values=None,
    )
    calls = []

    def gpu_average(*args):
        calls.append(args)
        return np.full_like(references, 7.0), {
            "class_average_transform_backend": "cuda",
            "class_average_batch_size": 4,
            "class_average_empty_components": np.asarray([], dtype=np.int64),
            "class_average_gpu_accumulation": "fp64_device",
        }

    monkeypatch.setattr(workflows, "_load_gpu_function", lambda name: gpu_average)
    output = workflows._finalize_class_averages(
        result,
        images,
        replace(config(apply_final_pose_to_raw=True), batch_size=4),
        "cuda",
    )

    assert len(calls) == 1
    assert calls[0][0] is images
    assert calls[0][1] is poses
    np.testing.assert_array_equal(calls[0][2], assignments)
    np.testing.assert_array_equal(calls[0][3], weights)
    assert calls[0][4] is references
    assert np.all(output._class_average_values == 7.0)
    assert output.metadata["class_average_gpu_accumulation"] == "fp64_device"
    assert output.metadata["class_average_estimator"] == (
        "final_map_pose_inlier_weighted_raw"
    )
    assert result.metadata["class_average_empty_components"].size == 0
