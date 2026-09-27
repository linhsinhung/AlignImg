from __future__ import annotations

from dataclasses import replace

import numpy as np

import alignimg as ai
from alignimg._fourier import soft_circular_mask
from tools.fast_hard_validation import result_hashes


def _reference_stack(size: int = 20) -> np.ndarray:
    y, x = np.indices((size, size), dtype=np.float32)
    values = []
    centers = (
        ((5.0, 13.0, 1.0), (14.0, 7.0, 0.55)),
        ((6.0, 5.0, 0.85), (13.0, 14.0, 1.0)),
        ((4.0, 10.0, 0.65), (14.0, 11.0, 1.0)),
    )
    for blobs in centers:
        image = np.zeros((size, size), dtype=np.float32)
        for cy, cx, amplitude in blobs:
            image += amplitude * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / 5.0)
        values.append(image)
    return np.asarray(values, dtype=np.float32)


def _config(**changes) -> ai.AlignmentConfig:
    config = ai.AlignmentConfig(
        max_iterations=1,
        top_l=1,
        angle_samples=36,
        proposal_angles_per_reference=4,
        translation_range=2.0,
        translation_step=1.0,
        reference_update="spatial",
        robust_weighting=False,
        halfset_diagnostics=False,
        center_references=False,
        lowpass_sigma=0.0,
        mask_radius=8.0,
        mask_soft_edge=1.0,
        batch_size=8,
    )
    return replace(config, **changes)


def _assert_same_inference(
    first: ai.AlignmentResult, second: ai.AlignmentResult
) -> None:
    np.testing.assert_array_equal(
        first.reference_assignments, second.reference_assignments
    )
    np.testing.assert_array_equal(first.responsibilities, second.responsibilities)
    np.testing.assert_array_equal(first.poses.angle_deg, second.poses.angle_deg)
    np.testing.assert_array_equal(first.poses.shift_y_px, second.poses.shift_y_px)
    np.testing.assert_array_equal(first.poses.shift_x_px, second.poses.shift_x_px)
    np.testing.assert_array_equal(first.poses.mirror, second.poses.mirror)


def test_top_l_one_is_one_hot_and_spatial_mstep_is_direct_hard_average():
    references = _reference_stack()[:2]
    images = np.repeat(references, 2, axis=0)
    result = ai.align_to_references(
        images,
        references,
        config=_config(),
        backend="cpu",
    )

    expected_responsibilities = np.eye(2, dtype=np.float32)[
        result.reference_assignments
    ]
    np.testing.assert_array_equal(result.responsibilities, expected_responsibilities)
    np.testing.assert_array_equal(
        result.reference_assignments, np.argmax(result.responsibilities, axis=1)
    )
    np.testing.assert_array_equal(
        result.candidates.posterior,
        np.ones((len(images), 1), dtype=np.float32),
    )

    aligned = ai.transform_images(images, result.poses, backend="cpu")
    mask = soft_circular_mask(20, 8.0, 1.0)
    direct = np.zeros_like(references)
    for reference_index in range(2):
        selected = result.reference_assignments == reference_index
        weights = result.inlier_weights[selected].astype(np.float64)
        direct[reference_index] = (
            np.tensordot(weights, aligned[selected].astype(np.float64), axes=(0, 0))
            / weights.sum()
        ) * mask
    np.testing.assert_allclose(result.references, direct, atol=2e-6, rtol=0)


def test_fixed_one_hot_joint_k3_matches_three_independent_k1_runs():
    references = _reference_stack()
    component = np.repeat(np.arange(3, dtype=np.int32), 2)
    images = references[component]
    priors = ai.make_class_priors(assignments=component, n_components=3)
    joint = ai.align_to_references(
        images,
        references,
        class_priors=priors,
        config=_config(),
        backend="cpu",
    )

    np.testing.assert_array_equal(joint.reference_assignments, component)
    for reference_index in range(3):
        selected = component == reference_index
        single = ai.align_to_references(
            images[selected],
            references[reference_index],
            config=_config(),
            backend="cpu",
        )
        np.testing.assert_allclose(
            joint.references[reference_index], single.references[0], atol=2e-6, rtol=0
        )
        np.testing.assert_allclose(
            joint.poses.angle_deg[selected], single.poses.angle_deg, atol=1e-3, rtol=0
        )
        np.testing.assert_allclose(
            joint.poses.shift_y_px[selected], single.poses.shift_y_px, atol=1e-3, rtol=0
        )
        np.testing.assert_allclose(
            joint.poses.shift_x_px[selected], single.poses.shift_x_px, atol=1e-3, rtol=0
        )
        np.testing.assert_array_equal(joint.poses.mirror[selected], single.poses.mirror)


def test_uniform_priors_can_select_across_all_references():
    references = _reference_stack()
    expected = np.arange(3, dtype=np.int32)
    result = ai.align_to_references(
        references.copy(),
        references,
        class_priors=np.full((3, 3), 1.0 / 3.0, dtype=np.float32),
        config=_config(),
        backend="cpu",
    )

    np.testing.assert_array_equal(result.reference_assignments, expected)


def test_hard_candidate_is_deterministic_and_preserves_mirror_convention():
    reference = _reference_stack()[0]
    source = ai.PoseSet(
        np.asarray([0.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([True]),
    )
    image = ai.transform_images(reference[None], source, backend="cpu")
    config = _config(mirror_search=True)
    first = ai.align_to_references(image, reference, config=config, backend="cpu")
    second = ai.align_to_references(image, reference, config=config, backend="cpu")

    assert bool(first.poses.mirror[0])
    assert result_hashes(first) == result_hashes(second)

    mirror_off = ai.align_to_references(
        image, reference, config=replace(config, mirror_search=False), backend="cpu"
    )
    assert not bool(mirror_off.poses.mirror[0])


def test_output_and_diagnostic_switches_do_not_change_hard_inference():
    references = _reference_stack()[:2]
    images = np.repeat(references, 2, axis=0)
    base = ai.align_to_references(images, references, config=_config(), backend="cpu")
    raw = ai.align_to_references(
        images,
        references,
        config=_config(apply_final_pose_to_raw=True),
        backend="cpu",
    )
    halfsets = ai.align_to_references(
        images,
        references,
        config=_config(halfset_diagnostics=True),
        backend="cpu",
    )

    _assert_same_inference(base, raw)
    _assert_same_inference(base, halfsets)
    np.testing.assert_allclose(base.references, halfsets.references, atol=2e-6, rtol=0)
    assert raw.metadata["class_average_estimator"] == (
        "final_map_pose_inlier_weighted_raw"
    )
    assert raw.metadata["final_raw_average_seconds"] >= 0.0
    assert raw.diagnostics[-1]["raw_average_seconds"] >= 0.0
    assert "halfset_effective_weight" in halfsets.diagnostics[0]
