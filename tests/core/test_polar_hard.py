from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import alignimg as ai
from alignimg._engine import run_soft_alignment_cpu
from alignimg._fourier import prepare_stack
from alignimg._polar_hard import (
    angular_correlation,
    infer_polar_hard_candidates_cpu,
    polar_radius,
    quadratic_peak_offset,
    spatial_polar_rings,
    translation_center_grid,
)
from alignimg._transform import transform_image


def _config(**changes) -> ai.AlignmentConfig:
    config = ai.AlignmentConfig(
        search_strategy="polar_hard",
        candidate_scoring="polar",
        score_model="polar_ring_ccf",
        reference_update="fourier",
        top_l=1,
        max_iterations=1,
        angle_samples=256,
        translation_range=4.0,
        translation_step=1.0,
        robust_weighting=False,
        halfset_diagnostics=False,
        center_references=False,
    )
    return replace(config, **changes)


def _reference(size: int = 48, *, offset: int = 0) -> np.ndarray:
    y, x = np.indices((size, size), dtype=np.float32)
    value = np.exp(-((y - (0.31 * size + offset)) ** 2 + (x - 0.67 * size) ** 2) / 14.0)
    value += 0.65 * np.exp(
        -((y - 0.72 * size) ** 2 + (x - (0.42 * size - offset)) ** 2) / 9.0
    )
    return value.astype(np.float32)


def _inverse_pose(
    angle: float, shift_y: float, shift_x: float, mirror: bool
) -> tuple[float, float, float, bool]:
    radians = np.deg2rad(angle)
    rotation = np.asarray(
        [[np.cos(radians), np.sin(radians)], [-np.sin(radians), np.cos(radians)]],
        dtype=np.float64,
    )
    reflection = np.asarray([[-1.0, 0.0], [0.0, 1.0]])
    linear = rotation @ (reflection if mirror else np.eye(2))
    inverse_shift = -linear.T @ np.asarray([shift_x, shift_y], dtype=np.float64)
    return (
        angle if mirror else -angle,
        float(inverse_shift[1]),
        float(inverse_shift[0]),
        mirror,
    )


def _particle(
    reference: np.ndarray,
    *,
    angle: float,
    shift_y: float = 0.0,
    shift_x: float = 0.0,
    mirror: bool = False,
) -> np.ndarray:
    inverse = _inverse_pose(angle, shift_y, shift_x, mirror)
    return transform_image(
        reference,
        angle_deg=inverse[0],
        shift_y_px=inverse[1],
        shift_x_px=inverse[2],
        mirror=inverse[3],
    )


def _angle_delta(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return (actual - expected + 180.0) % 360.0 - 180.0


def test_polar_hard_config_contract_and_contradictory_options():
    normalized = _config().normalized(workflow="global")

    assert normalized.search_strategy == "polar_hard"
    assert normalized.candidate_scoring == "polar"
    assert normalized.score_model == "polar_ring_ccf"
    assert normalized.top_l == 1

    invalid = [
        (dict(candidate_scoring="fourier"), "candidate_scoring='polar'"),
        (dict(score_model="fourier_ncc"), "score_model='polar_ring_ccf'"),
        (dict(top_l=2), "top_l=1"),
        (dict(proposal_angles_per_reference=4), "proposal_angles_per_reference"),
        (dict(oversampling_order=2), "oversampling_order"),
        (dict(rescue_uncertain_particles=True), "rescue_uncertain_particles"),
    ]
    for changes, message in invalid:
        with pytest.raises(ValueError, match=message):
            replace(_config(), **changes).normalized(workflow="global")

    with pytest.raises(ValueError, match="require search_strategy='polar_hard'"):
        ai.AlignmentConfig(candidate_scoring="polar").normalized(workflow="global")


def test_spatial_polar_identity_known_angles_translation_sign_and_integer_origin():
    reference = _reference()
    truth_angle = np.asarray([0.0, 90.0, -37.5, 143.0], dtype=np.float32)
    truth_y = np.asarray([0.0, 2.0, -1.0, 0.0], dtype=np.float32)
    truth_x = np.asarray([0.0, -1.0, 2.0, 0.0], dtype=np.float32)
    images = np.stack(
        [
            _particle(reference, angle=a, shift_y=dy, shift_x=dx)
            for a, dy, dx in zip(truth_angle, truth_y, truth_x)
        ]
    )

    result = ai.align_to_references(images, reference, config=_config(), backend="cpu")

    angular_bin = 360.0 / 256
    assert (
        np.max(np.abs(_angle_delta(result.poses.angle_deg, truth_angle))) <= angular_bin
    )
    assert np.max(np.abs(result.poses.shift_y_px - truth_y)) <= 0.1
    assert np.max(np.abs(result.poses.shift_x_px - truth_x)) <= 0.1
    assert result.metadata["center_convention"] == "integer-origin-floor-N-over-2"


def test_periodic_angular_wrap_and_three_point_quadratic_fit():
    reference = _reference()
    config = _config(translation_range=0.0)
    prepared = prepare_stack(reference[None], config)
    radius = polar_radius(reference.shape[0], config)
    rings = spatial_polar_rings(
        prepared.spatial[0],
        center_y=0.0,
        center_x=0.0,
        radius=radius,
        angle_samples=config.angle_samples,
    )
    shifted = np.roll(rings, 255, axis=0)
    curve = angular_correlation(shifted, rings)

    assert int(np.argmax(curve)) == 255
    offset, accepted, reason = quadratic_peak_offset(0.8, 1.0, 0.9)
    assert accepted and reason == "accepted"
    assert -0.5 <= offset <= 0.5
    assert quadratic_peak_offset(1.0, 1.0, 1.0) == (
        0.0,
        False,
        "flat_or_convex",
    )
    assert quadratic_peak_offset(2.0, 1.0, 0.0) == (
        0.0,
        False,
        "flat_or_convex",
    )
    assert quadratic_peak_offset(np.nan, 1.0, 0.0) == (
        0.0,
        False,
        "non_finite",
    )
    assert quadratic_peak_offset(4.0, 3.0, 0.0) == (
        0.0,
        False,
        "out_of_range",
    )


def test_translation_grid_clips_boundaries_and_accumulates_centers():
    first, first_rejected = translation_center_grid(
        0.0,
        0.0,
        size=32,
        radius=11.0,
        translation_range=2.0,
        translation_step=1.0,
    )
    second, second_rejected = translation_center_grid(
        -2.0,
        0.0,
        size=32,
        radius=11.0,
        translation_range=2.0,
        translation_step=1.0,
    )

    assert first[0] == (-2.0, -2.0)
    assert (-4.0, 0.0) in second
    assert first_rejected == 0
    assert second_rejected == 0

    clipped, rejected = translation_center_grid(
        -4.0,
        0.0,
        size=32,
        radius=11.0,
        translation_range=2.0,
        translation_step=1.0,
    )
    assert all(center_y >= -5.0 for center_y, _ in clipped)
    assert rejected > 0


def test_later_iteration_uses_accumulated_translation_center_and_global_angle():
    reference = _reference(32)
    image = _particle(reference, angle=90.0, shift_y=4.0)
    config = _config(
        angle_samples=128,
        translation_range=2.0,
        max_iterations=3,
    ).normalized(workflow="global")
    particles = prepare_stack(image[None], config)
    references = prepare_stack(reference[None], config)
    priors = np.ones((1, 1), dtype=np.float32)

    first = infer_polar_hard_candidates_cpu(
        particles, references, config, priors, 0.08, None
    )
    first_center = np.asarray(
        [[first["_polar_center_y_px"][0, 0], first["_polar_center_x_px"][0, 0]]]
    )
    second = infer_polar_hard_candidates_cpu(
        particles,
        references,
        config,
        priors,
        0.08,
        None,
        translation_centers=first_center,
    )

    first_error = abs(float(first["shift_y_px"][0, 0]) - 4.0)
    second_error = abs(float(second["shift_y_px"][0, 0]) - 4.0)
    assert second_error < first_error
    assert not np.array_equal(first["_polar_center_x_px"], second["_polar_center_x_px"])
    assert abs(float(_angle_delta(second["angle_deg"][0, 0], 90.0))) <= 360 / 128


def test_mirror_convention_and_hard_one_hot_contract():
    reference = _reference()
    image = _particle(
        reference,
        angle=-37.5,
        shift_y=-1.0,
        shift_x=2.0,
        mirror=True,
    )
    result = ai.align_to_references(
        image[None],
        reference,
        config=_config(mirror_search=True),
        backend="cpu",
    )

    assert bool(result.poses.mirror[0]) is True
    assert abs(float(_angle_delta(result.poses.angle_deg[0], -37.5))) <= 360 / 256
    assert np.array_equal(result.responsibilities, np.ones((1, 1), dtype=np.float32))
    assert result.reference_assignments[0] == 0


def test_zero_priors_tie_order_and_corrective_priors():
    reference = _reference()
    references = np.stack([reference, reference])
    image = reference[None]
    config = _config(translation_range=0.0, angle_samples=64)

    zero_excludes_first = ai.align_to_references(
        image,
        references,
        class_priors=np.asarray([[0.0, 1.0]], dtype=np.float32),
        config=config,
        backend="cpu",
    )
    uniform_tie = ai.align_to_references(
        image,
        references,
        class_priors=np.asarray([[0.5, 0.5]], dtype=np.float32),
        config=config,
        backend="cpu",
    )
    corrective = ai.align_to_references(
        image,
        references,
        class_priors=np.asarray([[0.1, 0.9]], dtype=np.float32),
        config=config,
        backend="cpu",
    )

    assert zero_excludes_first.reference_assignments[0] == 1
    assert uniform_tie.reference_assignments[0] == 0
    assert corrective.reference_assignments[0] == 1


def test_fixed_k3_matches_independent_k1_and_preserves_mstep_contracts():
    references = np.stack(
        [_reference(offset=-2), _reference(offset=0), _reference(offset=2)]
    )
    component = np.repeat(np.arange(3, dtype=np.int32), 2)
    angles = np.asarray([0.0, 90.0, -37.5, 45.0, 143.0, -90.0])
    images = np.stack(
        [
            _particle(references[k], angle=float(angle))
            for k, angle in zip(component, angles)
        ]
    )
    priors = np.eye(3, dtype=np.float32)[component]

    for reference_update in ("spatial", "fourier"):
        config = _config(translation_range=0.0, reference_update=reference_update)
        joint = ai.align_to_references(
            images,
            references,
            class_priors=priors,
            config=config,
            backend="cpu",
        )
        assert np.array_equal(joint.reference_assignments, component)
        assert np.all(np.isfinite(joint.references))
        assert np.allclose(joint.responsibilities.sum(axis=1), 1.0)
        for reference_index in range(3):
            selected = component == reference_index
            single = ai.align_to_references(
                images[selected],
                references[reference_index],
                config=config,
                backend="cpu",
            )
            assert (
                np.max(
                    np.abs(
                        _angle_delta(
                            joint.poses.angle_deg[selected], single.poses.angle_deg
                        )
                    )
                )
                <= 1e-6
            )
            assert np.allclose(
                joint.poses.shift_y_px[selected], single.poses.shift_y_px, atol=1e-6
            )
            assert np.allclose(
                joint.poses.shift_x_px[selected], single.poses.shift_x_px, atol=1e-6
            )
            assert np.allclose(
                joint.references[reference_index], single.references[0], atol=2e-5
            )


def test_polar_hard_injected_inference_receives_accumulated_translation_centers():
    reference = _reference(32)
    config = _config(
        translation_range=0.0,
        max_iterations=2,
        center_references=False,
    )
    particles = prepare_stack(reference[None], config)
    observed_centers = []

    def candidate_inference(
        _particles,
        prepared_references,
        _config_value,
        priors,
        temperature,
        initial_poses,
        *,
        translation_centers=None,
        rescue_mask=None,
    ):
        del _particles, _config_value, rescue_mask
        observed_centers.append(
            None
            if translation_centers is None
            else np.asarray(translation_centers).copy()
        )
        return infer_polar_hard_candidates_cpu(
            particles,
            prepared_references,
            config,
            priors,
            temperature,
            initial_poses,
            translation_centers=translation_centers,
        )

    result = run_soft_alignment_cpu(
        reference[None],
        reference[None],
        config=config,
        class_priors=None,
        initial_poses=None,
        workflow="global",
        _candidate_inference=candidate_inference,
        _backend_name="cuda",
    )
    assert observed_centers[0] is None
    assert observed_centers[1].shape == (1, 2)
    assert np.all(np.isfinite(observed_centers[1]))
    first, second = result.diagnostics
    assert first["effective_search_strategy"] == "polar_hard"
    assert first["effective_candidate_scoring"] == "polar"
    assert first["effective_score_model"] == "polar_ring_ccf"
    assert np.array_equal(first["active_reference_count"], [1])
    assert first["polar_angular_bin_count"] == config.angle_samples
    assert np.array_equal(first["polar_translation_center_count"], [1])
    assert first["polar_quadratic_fit_attempt_count"] == 1
    assert (
        first["polar_quadratic_fit_accept_count"]
        + first["polar_quadratic_fit_fallback_count"]
        == 1
    )
    assert first["polar_boundary_hit_count"] == 0
    assert np.array_equal(first["hard_class_occupancy"], [1])
    assert first["hard_reassignment_fraction"] is None
    assert np.array_equal(first["hard_assignment_transition_matrix"], [[0]])
    assert second["hard_reassignment_fraction"] == 0.0
    assert np.array_equal(second["hard_assignment_transition_matrix"], [[1]])
    for name in (
        "candidate_inference_seconds",
        "reference_update_seconds",
        "centering_seconds",
        "frc_diagnostics_seconds",
    ):
        assert first[name] >= 0.0


def test_empty_component_behavior_remains_observational():
    reference = _reference(32)
    result = ai.align_to_references(
        reference[None],
        np.stack([reference, reference]),
        class_priors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        config=_config(translation_range=0.0, angle_samples=64),
        backend="cpu",
    )

    assert result.reference_assignments[0] == 0
    assert np.array_equal(result.diagnostics[0]["reseeded_components"], [1])
    assert np.all(np.isfinite(result.references))
