from __future__ import annotations

import numpy as np
import pytest

import alignimg as ai
from alignimg._adaptive import (
    fine_offsets,
    infer_adaptive_candidates,
    oversample_coarse_cells,
    schedule_uncertain_rescue,
    score_candidates_fourier_cpu,
    select_posterior_mass,
    symmetric_grid_offsets,
)
from alignimg._fourier import prepare_stack
from alignimg._transform import transform_image


def asymmetric_reference(size: int = 32) -> np.ndarray:
    y, x = np.indices((size, size), dtype=np.float32)
    result = np.exp(-((y - 9) ** 2 + (x - 20) ** 2) / 8.0)
    result += 0.65 * np.exp(-((y - 23) ** 2 + (x - 11) ** 2) / 4.0)
    result += 0.3 * np.exp(-((y - 17) ** 2 + (x - 25) ** 2) / 2.5)
    return result.astype(np.float32)


def adaptive_config(**changes) -> ai.AlignmentConfig:
    values = dict(
        search_strategy="adaptive_posterior",
        max_iterations=1,
        top_l=16,
        angle_samples=72,
        coarse_angle_step=6.0,
        coarse_shift_step=1.0,
        local_angle_range=12.0,
        local_shift_range=2.0,
        adaptive_fraction=0.999,
        oversampling_order=1,
        max_adaptive_cells=None,
        temperature_start=0.04,
        temperature_end=0.04,
        pose_angle_sigma=15.0,
        pose_shift_sigma=3.0,
        halfset_diagnostics=False,
        center_references=False,
        batch_size=3,
    )
    values.update(changes)
    return ai.AlignmentConfig(**values)


def angular_error(value: float, truth: float) -> float:
    return abs((float(value) - float(truth) + 180.0) % 360.0 - 180.0)


def test_posterior_mass_selection_reaches_fraction_and_is_stable():
    posterior = np.asarray([0.08, 0.46, 0.44, 0.02], dtype=np.float64)

    selected, mass, capped = select_posterior_mass(
        posterior, fraction=0.85, max_cells=None
    )

    assert selected.tolist() == [1, 2]
    assert mass >= 0.85
    assert capped is False
    selected, mass, capped = select_posterior_mass(
        posterior, fraction=0.999, max_cells=1
    )
    assert selected.tolist() == [1]
    assert mass == pytest.approx(0.46)
    assert capped is True


def test_posterior_mass_selection_keeps_at_least_one_cell():
    selected, mass, _ = select_posterior_mass(
        np.asarray([0.0, 0.0, 1.0]), fraction=0.999, max_cells=None
    )
    assert selected.tolist() == [2]
    assert mass == pytest.approx(1.0)


def test_posterior_mass_selection_retains_two_modes_when_needed_for_mass():
    selected, mass, _ = select_posterior_mass(
        np.asarray([0.48, 0.01, 0.47, 0.04]),
        fraction=0.90,
        max_cells=None,
    )

    assert selected.tolist() == [0, 2]
    assert mass == pytest.approx(0.95)


def test_uncertainty_rescue_is_bounded_and_prioritizes_boundary_then_entropy():
    config = adaptive_config(
        rescue_uncertain_particles=True,
        rescue_normalized_entropy_threshold=0.80,
        rescue_map_posterior_threshold=0.05,
        rescue_max_fraction=0.40,
    ).normalized(workflow="refine")

    scheduled, entropy_trigger, map_trigger = schedule_uncertain_rescue(
        boundary_hit=np.asarray([False, True, False, False, False]),
        normalized_entropy=np.asarray([0.90, 0.70, 0.99, 0.95, 0.20]),
        map_posterior=np.asarray([0.20, 0.20, 0.20, 0.01, 0.01]),
        eligible=np.asarray([True, True, True, True, False]),
        group_index=np.zeros(5, dtype=np.int32),
        config=config,
    )

    assert entropy_trigger.tolist() == [True, False, True, True, False]
    assert map_trigger.tolist() == [False, False, False, True, False]
    assert scheduled.tolist() == [False, True, True, False, False]


def test_uncertainty_rescue_cap_is_applied_per_component():
    config = adaptive_config(
        rescue_uncertain_particles=True,
        rescue_normalized_entropy_threshold=0.0,
        rescue_max_fraction=0.25,
    ).normalized(workflow="refine")

    scheduled, _, _ = schedule_uncertain_rescue(
        boundary_hit=np.zeros(8, dtype=bool),
        normalized_entropy=np.arange(8, dtype=np.float64) / 8.0,
        map_posterior=np.full(8, 0.5),
        eligible=np.ones(8, dtype=bool),
        group_index=np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
        config=config,
    )

    assert scheduled.tolist() == [False, False, False, True, False, False, False, True]


def test_centered_coarse_grid_and_oversampling_contract():
    assert np.array_equal(
        symmetric_grid_offsets(15.0, 6.0),
        np.asarray([-12.0, -6.0, 0.0, 6.0, 12.0]),
    )
    assert np.array_equal(fine_offsets(6.0, 1), [-3.0, 0.0, 3.0])
    cells = np.asarray(
        [(0, 179.0, 0.0, 0.0, False)],
        dtype=[
            ("reference_index", np.int32),
            ("angle_deg", np.float64),
            ("shift_y_px", np.float64),
            ("shift_x_px", np.float64),
            ("mirror", np.bool_),
        ],
    )

    fine = oversample_coarse_cells(
        cells,
        center_angle_deg=179.0,
        center_shift_y_px=0.0,
        center_shift_x_px=0.0,
        coarse_angle_step=6.0,
        coarse_shift_step=1.0,
        local_angle_range=6.0,
        local_shift_range=1.0,
        oversampling_order=1,
    )

    assert np.any(np.isclose(fine["angle_deg"], 179.0))
    assert np.any(np.isclose(fine["angle_deg"], -178.0))
    assert np.max(np.abs(fine["shift_y_px"])) <= 1.0
    assert np.max(np.abs(fine["shift_x_px"])) <= 1.0
    keys = np.column_stack(
        (
            fine["reference_index"],
            np.round(fine["angle_deg"], 6),
            np.round(fine["shift_y_px"], 6),
            np.round(fine["shift_x_px"], 6),
            fine["mirror"],
        )
    )
    assert len(np.unique(keys, axis=0)) == len(fine)


def test_adaptive_refinement_improves_known_pose_and_reference():
    rng = np.random.default_rng(7)
    reference = asymmetric_reference()
    source_angle = 12.0
    clean = transform_image(reference, angle_deg=source_angle)
    particles = np.stack(
        [clean + 0.08 * rng.normal(size=clean.shape) for _ in range(8)]
    ).astype(np.float32)
    initial = ai.PoseSet(
        np.full(8, -6.0, dtype=np.float32),
        np.zeros(8, dtype=np.float32),
        np.zeros(8, dtype=np.float32),
        np.zeros(8, dtype=bool),
    )
    initial_aligned = ai.transform_images(particles, initial).mean(axis=0)
    initial_correlation = np.corrcoef(initial_aligned.ravel(), reference.ravel())[0, 1]

    result = ai.refine_alignment(
        particles,
        reference,
        initial,
        config=adaptive_config(),
        backend="cpu",
    )

    assert np.median(
        [angular_error(value, -source_angle) for value in result.poses.angle_deg]
    ) < angular_error(-6.0, -source_angle)
    final_correlation = np.corrcoef(result.references[0].ravel(), reference.ravel())[
        0, 1
    ]
    assert final_correlation >= initial_correlation - 0.01
    assert np.allclose(result.candidates.posterior.sum(axis=1), 1.0)
    assert np.all(np.isfinite(result.pose_entropy))
    assert np.all(result.map_posterior > 0.0)


def test_adaptive_translation_oversampling_recovers_subpixel_shift():
    reference = asymmetric_reference()
    particle = transform_image(reference, shift_y_px=1.5, shift_x_px=-1.0)
    initial = ai.PoseSet.identity(1)

    result = ai.refine_alignment(
        particle[None],
        reference,
        initial,
        config=adaptive_config(
            local_angle_range=0.0,
            coarse_angle_step=6.0,
            local_shift_range=2.0,
            coarse_shift_step=1.0,
        ),
        backend="cpu",
    )

    assert float(result.poses.shift_y_px[0]) == pytest.approx(-1.5, abs=0.26)
    assert float(result.poses.shift_x_px[0]) == pytest.approx(1.0, abs=0.26)


def test_adaptive_results_do_not_depend_on_particle_batch_size():
    reference = asymmetric_reference(24)
    particles = np.stack(
        [transform_image(reference, angle_deg=value) for value in (6, 9, 12, 15)]
    )
    initial = ai.PoseSet(
        np.asarray([-3.0, -6.0, -9.0, -12.0]),
        np.zeros(4),
        np.zeros(4),
        np.zeros(4, dtype=bool),
    )
    first = ai.refine_alignment(
        particles,
        reference,
        initial,
        config=adaptive_config(batch_size=1),
        backend="cpu",
    )
    second = ai.refine_alignment(
        particles,
        reference,
        initial,
        config=adaptive_config(batch_size=4),
        backend="cpu",
    )

    assert np.allclose(first.poses.angle_deg, second.poses.angle_deg)
    assert np.allclose(first.poses.shift_y_px, second.poses.shift_y_px)
    assert np.allclose(first.poses.shift_x_px, second.poses.shift_x_px)
    assert np.allclose(first.candidates.posterior, second.candidates.posterior)
    assert np.allclose(first.references, second.references)


def test_batched_adaptive_scorer_preserves_serial_candidate_contract():
    reference = asymmetric_reference(24)
    particles = np.stack(
        [transform_image(reference, angle_deg=value) for value in (6, 9, 12, 15)]
    )
    initial = ai.PoseSet(
        np.asarray([-3.0, -6.0, -9.0, -12.0]),
        np.zeros(4),
        np.zeros(4),
        np.zeros(4, dtype=bool),
    )
    config = adaptive_config(batch_size=16).normalized(workflow="refine")
    priors = np.ones((len(particles), 1), dtype=np.float32)

    serial = infer_adaptive_candidates(
        prepare_stack(particles, config),
        prepare_stack(reference[None], config),
        config,
        priors,
        0.04,
        initial,
        score_candidates=score_candidates_fourier_cpu,
    )
    calls = []

    def score_batches(particle_stack, reference_stack, indices, cell_batches):
        calls.append(np.asarray(indices).copy())
        return [
            score_candidates_fourier_cpu(
                particle_stack, reference_stack, int(index), cells
            )
            for index, cells in zip(indices, cell_batches, strict=True)
        ]

    batched = infer_adaptive_candidates(
        prepare_stack(particles, config),
        prepare_stack(reference[None], config),
        config,
        priors,
        0.04,
        initial,
        score_candidates=score_candidates_fourier_cpu,
        score_candidate_batches=score_batches,
    )

    assert len(calls) == 2
    assert all(call.tolist() == [0, 1, 2, 3] for call in calls)
    assert serial.keys() == batched.keys()
    for name in serial:
        np.testing.assert_equal(batched[name], serial[name])


def test_adaptive_top_l_only_limits_public_hypotheses_not_marginalization():
    reference = asymmetric_reference(24)
    particles = np.stack(
        [transform_image(reference, angle_deg=value) for value in (6, 9, 12, 15)]
    )
    initial = ai.PoseSet(
        np.asarray([-3.0, -6.0, -9.0, -12.0]),
        np.zeros(4),
        np.zeros(4),
        np.zeros(4, dtype=bool),
    )
    narrow = ai.refine_alignment(
        particles,
        reference,
        initial,
        config=adaptive_config(top_l=1, adaptive_fraction=1.0),
        backend="cpu",
    )
    wide = ai.refine_alignment(
        particles,
        reference,
        initial,
        config=adaptive_config(top_l=16, adaptive_fraction=1.0),
        backend="cpu",
    )

    assert narrow.candidates.posterior.shape == (4, 1)
    assert wide.candidates.posterior.shape == (4, 16)
    assert np.allclose(narrow.poses.angle_deg, wide.poses.angle_deg)
    assert np.allclose(narrow.references, wide.references, atol=1e-7)
    assert np.allclose(narrow.responsibilities, wide.responsibilities, atol=1e-7)
    assert np.allclose(narrow.pose_entropy, wide.pose_entropy, atol=1e-7)
    assert np.allclose(narrow.map_posterior, wide.map_posterior, atol=1e-7)
    assert narrow.diagnostics[0]["mean_retained_fine_mass"] < 1.0


def test_boundary_rescue_returns_bad_initial_pose_to_global_basin():
    reference = asymmetric_reference()
    source_angle = 60.0
    particle = transform_image(reference, angle_deg=source_angle)
    particles = np.repeat(particle[None], 10, axis=0)
    initial_angles = np.full(10, -source_angle, dtype=np.float32)
    initial_angles[-1] = 0.0
    initial = ai.PoseSet(
        initial_angles,
        np.zeros(10),
        np.zeros(10),
        np.zeros(10, dtype=bool),
    )
    without_rescue = ai.refine_alignment(
        particles,
        reference,
        initial,
        config=adaptive_config(
            max_iterations=3,
            local_angle_range=12.0,
            rescue_uncertain_particles=False,
        ),
        backend="cpu",
    )
    with_rescue = ai.refine_alignment(
        particles,
        reference,
        initial,
        config=adaptive_config(
            max_iterations=3,
            local_angle_range=12.0,
            rescue_uncertain_particles=True,
            proposal_angles_per_reference=8,
        ),
        backend="cpu",
    )

    failed_error = angular_error(without_rescue.poses.angle_deg[-1], -source_angle)
    rescued_error = angular_error(with_rescue.poses.angle_deg[-1], -source_angle)
    assert failed_error > 12.0
    assert rescued_error <= 5.0
    assert any(item["rescue_particle_count"] > 0 for item in with_rescue.diagnostics)


def test_entropy_rescue_uses_global_fallback_without_a_boundary_hit():
    reference = asymmetric_reference()
    source_angle = 60.0
    particles = np.repeat(
        transform_image(reference, angle_deg=source_angle)[None], 10, axis=0
    )
    initial_angles = np.full(10, -source_angle, dtype=np.float32)
    initial_angles[0] = 0.0
    initial = ai.PoseSet(
        initial_angles,
        np.zeros(10),
        np.zeros(10),
        np.zeros(10, dtype=bool),
    )
    result = ai.refine_alignment(
        particles,
        reference,
        initial,
        config=adaptive_config(
            max_iterations=3,
            local_angle_range=0.0,
            local_shift_range=0.0,
            rescue_uncertain_particles=True,
            rescue_normalized_entropy_threshold=0.0,
            rescue_max_fraction=0.1,
            proposal_angles_per_reference=8,
        ),
        backend="cpu",
    )

    assert result.diagnostics[0]["boundary_hit_count"] == 0
    assert result.diagnostics[0]["rescue_entropy_trigger_count"] == 10
    assert result.diagnostics[0]["rescue_scheduled_count"] == 1
    assert result.diagnostics[1]["rescue_particle_count"] == 1
    assert result.diagnostics[1]["rescue_accepted_count"] == 1
    assert result.diagnostics[1]["rescue_scheduled_count"] == 0
    assert result.diagnostics[2]["rescue_particle_count"] == 0
    assert angular_error(result.poses.angle_deg[0], -source_angle) <= 5.0


def test_rescue_rejects_a_global_proposal_without_required_score_gain():
    reference = asymmetric_reference()
    source_angle = 60.0
    particles = np.repeat(
        transform_image(reference, angle_deg=source_angle)[None], 10, axis=0
    )
    initial = ai.PoseSet(
        np.full(10, -source_angle, dtype=np.float32),
        np.zeros(10),
        np.zeros(10),
        np.zeros(10, dtype=bool),
    )
    result = ai.refine_alignment(
        particles,
        reference,
        initial,
        config=adaptive_config(
            max_iterations=3,
            local_angle_range=0.0,
            local_shift_range=0.0,
            rescue_uncertain_particles=True,
            rescue_normalized_entropy_threshold=0.0,
            rescue_max_fraction=0.1,
            rescue_min_score_improvement=10.0,
            proposal_angles_per_reference=8,
        ),
        backend="cpu",
    )

    assert result.diagnostics[1]["rescue_particle_count"] == 1
    assert result.diagnostics[1]["rescue_accepted_count"] == 0
    assert result.diagnostics[1]["rescue_rejected_count"] == 1
    assert (
        max(angular_error(value, -source_angle) for value in result.poses.angle_deg)
        <= 1.0
    )


def test_workflow_strategy_defaults_are_explicit():
    assert ai.AlignmentConfig.preset("reference_free").search_strategy == "proposal"
    assert ai.AlignmentConfig.preset("global_balanced").search_strategy == "proposal"
    assert ai.AlignmentConfig.preset("refine").search_strategy == "quadratic_refine"
    with pytest.raises(ValueError, match="requires initial_poses"):
        ai.align_to_references(
            np.zeros((1, 16, 16), dtype=np.float32),
            np.zeros((16, 16), dtype=np.float32),
            config=adaptive_config(),
        )


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"search_strategy": "unknown"}, "search_strategy"),
        ({"coarse_angle_step": 0.0}, "coarse_angle_step"),
        ({"coarse_shift_step": 0.0}, "coarse_shift_step"),
        ({"local_angle_range": -1.0}, "local_angle_range"),
        ({"local_shift_range": -1.0}, "local_shift_range"),
        ({"adaptive_fraction": 0.0}, "adaptive_fraction"),
        ({"oversampling_order": -1}, "oversampling_order"),
        ({"max_adaptive_cells": 0}, "max_adaptive_cells"),
        (
            {"rescue_normalized_entropy_threshold": -0.1},
            "rescue_normalized_entropy_threshold",
        ),
        (
            {"rescue_map_posterior_threshold": 1.1},
            "rescue_map_posterior_threshold",
        ),
        ({"rescue_max_fraction": 0.0}, "rescue_max_fraction"),
        ({"rescue_min_score_improvement": -0.1}, "rescue_min_score_improvement"),
        ({"candidate_scoring": "unknown"}, "candidate_scoring"),
    ],
)
def test_adaptive_config_rejects_invalid_values(changes, message):
    with pytest.raises(ValueError, match=message):
        adaptive_config(**changes).normalized(workflow="refine")


@pytest.mark.parametrize("backend", ["cupy", "cuda"])
def test_adaptive_backend_parity_when_gpu_is_available(backend: str):
    if not ai.available_alignment_backends()[backend]["available"]:
        pytest.skip(f"{backend} backend is unavailable")
    reference = asymmetric_reference(24)
    particles = np.stack(
        [transform_image(reference, angle_deg=value) for value in (6.0, 12.0)]
    )
    initial = ai.PoseSet(
        np.asarray([-3.0, -9.0]),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2, dtype=bool),
    )
    config = adaptive_config(batch_size=2)
    cpu = ai.refine_alignment(
        particles, reference, initial, config=config, backend="cpu"
    )
    gpu = ai.refine_alignment(
        particles, reference, initial, config=config, backend=backend
    )

    assert np.allclose(cpu.poses.angle_deg, gpu.poses.angle_deg, atol=1e-3)
    assert np.allclose(cpu.poses.shift_y_px, gpu.poses.shift_y_px, atol=1e-3)
    assert np.allclose(cpu.poses.shift_x_px, gpu.poses.shift_x_px, atol=1e-3)
    assert np.allclose(cpu.candidates.posterior, gpu.candidates.posterior, atol=2e-4)
    assert np.allclose(cpu.references, gpu.references, atol=2e-4)
