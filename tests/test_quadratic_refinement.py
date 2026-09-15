from __future__ import annotations

import numpy as np
import pytest

import alignimg as ai
from alignimg._engine import _inlier_weights
from alignimg._quadratic import (
    TranslationProfiles,
    angular_local_maxima,
    infer_quadratic_candidates,
    quadratic_vertex_offset,
)
from alignimg._fourier import prepare_stack
from alignimg._transform import transform_image


def asymmetric_reference(size: int = 32) -> np.ndarray:
    y, x = np.indices((size, size), dtype=np.float32)
    result = np.exp(-((y - 10) ** 2 + (x - 18) ** 2) / 10.0)
    result += 0.7 * np.exp(-((y - 22) ** 2 + (x - 11) ** 2) / 5.0)
    result += 0.3 * np.exp(-((y - 7) ** 2 + (x - 6) ** 2) / 3.0)
    return result.astype(np.float32)


def quadratic_config(**changes) -> ai.AlignmentConfig:
    values = dict(
        max_iterations=1,
        top_l=8,
        search_strategy="quadratic_refine",
        local_angle_range=7.0,
        coarse_angle_step=1.0,
        local_shift_range=3.0,
        halfset_diagnostics=False,
        center_references=False,
        robust_weighting=False,
        mask_soft_edge=2.0,
    )
    values.update(changes)
    return ai.AlignmentConfig(**values)


def test_quadratic_vertex_and_fallback_contracts():
    offset, reason = quadratic_vertex_offset(-2.25, -0.25, -0.25)
    assert offset == pytest.approx(0.5)
    assert reason == "accepted"
    assert quadratic_vertex_offset(1.0, 1.0, 1.0)[1] == "flat_or_convex"
    assert quadratic_vertex_offset(0.0, 1.0, 4.0)[1] == "flat_or_convex"
    assert quadratic_vertex_offset(0.0, 1.0, 1.9)[1] == "out_of_bounds"
    assert quadratic_vertex_offset(np.nan, 1.0, 0.0)[1] == "nonfinite"


def test_fixed_reference_robust_weights_match_independent_groups():
    scores = np.asarray([0.1, 0.9, 0.2, 0.7, 0.4, 0.8], dtype=np.float32)
    groups = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int32)
    config = quadratic_config(robust_weighting=True)
    grouped = _inlier_weights(scores, config, groups=groups)
    for group in (0, 1):
        selected = groups == group
        assert np.array_equal(
            grouped[selected], _inlier_weights(scores[selected], config)
        )


def test_angular_local_maxima_are_stable_and_collapse_plateaus():
    assert np.array_equal(
        angular_local_maxima(np.asarray([0.0, 2.0, 2.0, 1.0, 3.0, 0.0])),
        np.asarray([1, 4]),
    )
    assert np.array_equal(
        angular_local_maxima(np.asarray([3.0, 2.0, 1.0])), np.asarray([0])
    )


def test_cpu_quadratic_refine_recovers_fractional_pose():
    reference = asymmetric_reference()
    source_angle = 3.4
    source_y = 1.3
    source_x = -0.7
    particle = transform_image(
        reference,
        angle_deg=source_angle,
        shift_y_px=source_y,
        shift_x_px=source_x,
    )
    inverse_radians = np.deg2rad(-source_angle)
    inverse_x = -(
        np.cos(inverse_radians) * source_x
        - np.sin(inverse_radians) * source_y
    )
    inverse_y = -(
        np.sin(inverse_radians) * source_x
        + np.cos(inverse_radians) * source_y
    )
    expected = np.asarray([-source_angle, inverse_y, inverse_x])
    initial = ai.PoseSet(
        angle_deg=np.asarray([-3.0], dtype=np.float32),
        shift_y_px=np.asarray([inverse_y + 0.2], dtype=np.float32),
        shift_x_px=np.asarray([inverse_x - 0.2], dtype=np.float32),
        mirror=np.asarray([False]),
    )

    result = ai.refine_alignment(
        particle[None],
        reference,
        initial,
        config=quadratic_config(),
        backend="cpu",
    )
    actual = np.asarray(
        [
            result.poses.angle_deg[0],
            result.poses.shift_y_px[0],
            result.poses.shift_x_px[0],
        ]
    )

    assert abs((actual[0] - expected[0] + 180.0) % 360.0 - 180.0) <= 0.5
    assert np.linalg.norm(actual[1:] - expected[1:]) <= 0.2
    assert result.diagnostics[0]["angle_fit_accept_count"] >= 1
    assert result.diagnostics[0]["translation_fit_accept_count"] >= 1
    assert result.metadata["search_strategy"] == "quadratic_refine"


def test_whitened_quadratic_refine_produces_finite_normalized_results():
    reference = asymmetric_reference()
    particle = transform_image(reference, angle_deg=2.5, shift_y_px=0.4)
    initial = ai.PoseSet(
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([-0.5], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([False]),
    )
    result = ai.refine_alignment(
        particle[None],
        reference,
        initial,
        config=quadratic_config(score_model="whitened_fourier_ncc"),
        backend="cpu",
    )
    assert np.all(np.isfinite(result.references))
    assert np.all(np.isfinite(result.responsibilities))
    assert np.allclose(result.responsibilities.sum(axis=1), 1.0)


def test_quadratic_refine_preserves_initial_mirror_when_search_is_off():
    reference = asymmetric_reference()
    particle = transform_image(reference, mirror=True)
    initial = ai.PoseSet(
        np.asarray([0.2], dtype=np.float32),
        np.asarray([0.1], dtype=np.float32),
        np.asarray([-0.1], dtype=np.float32),
        np.asarray([True]),
    )
    result = ai.refine_alignment(
        particle[None], reference, initial, config=quadratic_config(), backend="cpu"
    )
    assert bool(result.poses.mirror[0])
    assert result.candidates.score[0, 0] > 0.98


def test_quadratic_full_modes_drive_responsibilities_and_public_top_l():
    config = quadratic_config(
        top_l=1,
        local_angle_range=2.0,
        coarse_angle_step=1.0,
        pose_angle_sigma=1e6,
    ).normalized(workflow="refine")
    particles = prepare_stack(np.ones((1, 16, 16), dtype=np.float32), config)
    references = prepare_stack(np.ones((2, 16, 16), dtype=np.float32), config)
    initial = ai.PoseSet(
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=np.bool_),
    )

    def profiles(*args):
        reference_index = int(args[3])
        angles = np.asarray(args[4], dtype=np.float64)
        center = -1.0 if reference_index == 0 else 1.0
        objective = -0.1 * (angles - center) ** 2 - 0.05 * reference_index
        return TranslationProfiles(
            angle_deg=angles,
            shift_y_px=np.zeros(len(angles)),
            shift_x_px=np.zeros(len(angles)),
            score=objective,
            objective=objective,
            fit_attempt_count=0,
            fit_accept_count=0,
            boundary_hit_count=0,
            flat_or_convex_count=0,
            out_of_bounds_count=0,
            exact_reject_count=0,
            objective_gain=0.0,
            ifft_count=len(angles),
        )

    values = infer_quadratic_candidates(
        particles,
        references,
        config,
        np.asarray([[0.6, 0.4]], dtype=np.float32),
        1.0,
        initial,
        translation_profiler=profiles,
    )
    full = values["_mstep_posterior"]
    assert len(full) == 10
    assert np.sum(full) == pytest.approx(1.0)
    assert values["posterior"].shape == (1, 1)
    assert values["posterior"][0, 0] == pytest.approx(1.0)
    assert values["_quadratic_angular_mode_count"][0] == 2
    assert values["_quadratic_posterior_support_count"][0] == 10
    assert values["_full_map_posterior"][0] < 1.0


def test_angle_pose_prior_changes_the_selected_quadratic_mode():
    initial = ai.PoseSet.identity(1)

    def profiles(*args):
        angles = np.asarray(args[4], dtype=np.float64)
        objective = np.maximum(
            2.0 - angles**2,
            2.3 - (angles - 3.0) ** 2,
        )
        return TranslationProfiles(
            angle_deg=angles,
            shift_y_px=np.zeros(len(angles)),
            shift_x_px=np.zeros(len(angles)),
            score=objective,
            objective=objective,
            fit_attempt_count=0,
            fit_accept_count=0,
            boundary_hit_count=0,
            flat_or_convex_count=0,
            out_of_bounds_count=0,
            exact_reject_count=0,
            objective_gain=0.0,
            ifft_count=len(angles),
        )

    def infer(sigma: float) -> dict[str, np.ndarray]:
        config = quadratic_config(
            local_angle_range=3.0,
            coarse_angle_step=1.0,
            pose_angle_sigma=sigma,
        ).normalized(workflow="refine")
        particles = prepare_stack(
            np.ones((1, 16, 16), dtype=np.float32), config
        )
        references = prepare_stack(
            np.ones((1, 16, 16), dtype=np.float32), config
        )
        return infer_quadratic_candidates(
            particles,
            references,
            config,
            np.ones((1, 1), dtype=np.float32),
            1.0,
            initial,
            translation_profiler=profiles,
        )

    tight = infer(1.0)
    broad = infer(1e6)
    assert tight["angle_deg"][0, 0] == pytest.approx(0.0)
    assert broad["angle_deg"][0, 0] == pytest.approx(3.0)


def test_fixed_mra_matches_independent_k1_quadratic_refinement():
    first = asymmetric_reference()
    second = np.rot90(first).copy()
    references = np.stack((first, second))
    labels = np.asarray([0, 1, 0, 1], dtype=np.int32)
    source_angles = np.asarray([2.4, -3.2, 1.7, -2.1], dtype=np.float32)
    particles = np.stack(
        [
            transform_image(
                references[label],
                angle_deg=float(angle),
                shift_y_px=0.4,
                shift_x_px=-0.3,
            )
            for label, angle in zip(labels, source_angles, strict=True)
        ]
    )
    initial = ai.PoseSet(
        -source_angles + 0.4,
        np.full(4, -0.3, dtype=np.float32),
        np.full(4, 0.2, dtype=np.float32),
        np.zeros(4, dtype=np.bool_),
    )
    config = quadratic_config(max_iterations=3, robust_weighting=True)
    priors = ai.make_class_priors(assignments=labels, n_components=2)
    joint = ai.refine_alignment(
        particles,
        references,
        initial,
        class_priors=priors,
        config=config,
        backend="cpu",
    )
    assert np.array_equal(joint.reference_assignments, labels)
    assert joint.metadata["robust_weighting_scope"] == "fixed_reference"
    for component in range(2):
        selected = labels == component
        local_initial = ai.PoseSet(
            initial.angle_deg[selected],
            initial.shift_y_px[selected],
            initial.shift_x_px[selected],
            initial.mirror[selected],
        )
        single = ai.refine_alignment(
            particles[selected],
            references[component],
            local_initial,
            config=config,
            backend="cpu",
        )
        assert np.allclose(
            joint.poses.angle_deg[selected], single.poses.angle_deg, atol=1e-5
        )
        assert np.allclose(
            joint.poses.shift_y_px[selected], single.poses.shift_y_px, atol=1e-5
        )
        assert np.allclose(
            joint.poses.shift_x_px[selected], single.poses.shift_x_px, atol=1e-5
        )
        assert np.corrcoef(
            joint.references[component].ravel(), single.references[0].ravel()
        )[0, 1] >= 0.9999


def test_corrective_quadratic_priors_remain_soft_and_normalized():
    reference = asymmetric_reference()
    references = np.stack((reference, np.rot90(reference).copy()))
    particles = references.copy()
    initial = ai.PoseSet(
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.bool_),
    )
    priors = ai.make_class_priors(
        assignments=np.asarray([0, 1]), n_components=2, trust=0.9
    )
    result = ai.refine_alignment(
        particles,
        references,
        initial,
        class_priors=priors,
        config=quadratic_config(),
        backend="cpu",
    )
    assert np.all(np.isfinite(result.responsibilities))
    assert np.allclose(result.responsibilities.sum(axis=1), 1.0)
    assert np.all(result.diagnostics[0]["effective_component_weight"] > 0.0)


@pytest.mark.parametrize("backend", ["cupy", "cuda"])
def test_quadratic_backend_pose_parity_when_gpu_is_available(backend: str):
    if not ai.available_alignment_backends()[backend]["available"]:
        pytest.skip(f"{backend} backend is unavailable")
    reference = asymmetric_reference()
    particles = np.stack(
        (
            transform_image(reference, angle_deg=2.4, shift_y_px=0.6),
            transform_image(reference, angle_deg=-3.1, shift_x_px=-0.7),
        )
    )
    initial = ai.PoseSet(
        np.asarray([-2.0, 3.0], dtype=np.float32),
        np.asarray([-0.5, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.6], dtype=np.float32),
        np.zeros(2, dtype=np.bool_),
    )
    config = quadratic_config()
    cpu = ai.refine_alignment(
        particles, reference, initial, config=config, backend="cpu"
    )
    gpu = ai.refine_alignment(
        particles, reference, initial, config=config, backend=backend
    )
    assert gpu.metadata["backend"] == backend
    expected_peak_backend = "native_cuda" if backend == "cuda" else "cupy"
    assert gpu.metadata["quadratic_peak_backend"] == expected_peak_backend
    assert np.allclose(gpu.poses.angle_deg, cpu.poses.angle_deg, atol=1e-3)
    assert np.allclose(gpu.poses.shift_y_px, cpu.poses.shift_y_px, atol=1e-3)
    assert np.allclose(gpu.poses.shift_x_px, cpu.poses.shift_x_px, atol=1e-3)
    assert np.allclose(gpu.responsibilities, cpu.responsibilities, atol=2e-4)


def test_quadratic_config_and_refine_preset_contract():
    assert (
        ai.AlignmentConfig(search_strategy=" QUADRATIC_REFINE ")
        .normalized(workflow="refine")
        .search_strategy
        == "quadratic_refine"
    )
    preset = ai.AlignmentConfig.preset("refine")
    assert preset.search_strategy == "quadratic_refine"
    assert preset.local_angle_range == 7.0
    assert preset.coarse_angle_step == 1.0
    assert preset.local_shift_range == 3.0
    with pytest.raises(ValueError, match="candidate_scoring"):
        quadratic_config(candidate_scoring="raster").normalized(workflow="refine")
    with pytest.raises(ValueError, match="reference_update"):
        quadratic_config(reference_update="spatial").normalized(workflow="refine")
    with pytest.raises(ValueError, match="rescue"):
        quadratic_config(rescue_uncertain_particles=True).normalized(
            workflow="refine"
        )
    with pytest.raises(ValueError, match="requires initial_poses"):
        ai.align_to_references(
            np.zeros((1, 16, 16), dtype=np.float32),
            np.zeros((16, 16), dtype=np.float32),
            config=quadratic_config(),
        )
