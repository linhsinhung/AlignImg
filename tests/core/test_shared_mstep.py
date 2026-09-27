from __future__ import annotations

import copy

import numpy as np
import pytest

import alignimg as ai
from alignimg._engine import (
    _apply_reference_center_shifts,
    _update_references,
    _update_references_fourier,
    _update_references_fourier_shared,
    _update_references_shared,
    run_soft_alignment_cpu,
)


def images_and_candidates():
    size = 32
    y, x = np.indices((size, size), dtype=np.float32)
    base = np.exp(-((y - 9.0) ** 2 + (x - 19.0) ** 2) / 14.0)
    base += 0.55 * np.exp(-((y - 22.0) ** 2 + (x - 11.0) ** 2) / 8.0)
    images = np.stack(
        [np.roll(base, (index - 2, 2 - index), axis=(0, 1)) for index in range(6)]
    ).astype(np.float32)
    candidates = {
        "reference_index": np.asarray(
            [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int32
        ),
        "angle_deg": np.asarray(
            [[0, 12], [-7, 18], [23, -11], [31, 4], [-19, 8], [15, -27]],
            dtype=np.float32,
        ),
        "shift_y_px": np.asarray(
            [[1, 0], [0, -1], [1, -1], [0, 1], [-1, 0], [1, -1]],
            dtype=np.float32,
        ),
        "shift_x_px": np.asarray(
            [[0, -1], [1, 0], [-1, 1], [1, 0], [0, 1], [-1, 0]],
            dtype=np.float32,
        ),
        "mirror": np.asarray(
            [[False, True], [False, False], [True, False], [False, True],
             [False, False], [True, False]],
            dtype=np.bool_,
        ),
        "posterior": np.asarray(
            [[0.7, 0.3], [0.6, 0.4], [0.8, 0.2], [0.55, 0.45],
             [0.75, 0.25], [0.65, 0.35]],
            dtype=np.float32,
        ),
    }
    inlier = np.asarray([1.0, 0.8, 0.65, 0.9, 0.5, 0.7], dtype=np.float32)
    first_half = np.asarray([True, False, True, False, False, True])
    return images, candidates, inlier, first_half


@pytest.mark.parametrize(
    ("legacy_updater", "shared_updater", "atol"),
    [
        (_update_references, _update_references_shared, 2e-6),
        (_update_references_fourier, _update_references_fourier_shared, 3e-6),
    ],
)
def test_shared_mstep_matches_three_legacy_updates_with_centering(
    legacy_updater, shared_updater, atol
):
    images, candidates, inlier, first_half = images_and_candidates()
    config = ai.AlignmentConfig(
        center_references=True,
        mask_radius=12.0,
        mask_soft_edge=3.0,
        lowpass_sigma=0.7,
    )
    fourier = np.fft.fft2(images, axes=(-2, -1)).astype(np.complex64)
    legacy_candidates = copy.deepcopy(candidates)
    full, full_weights, center_shifts = legacy_updater(
        images,
        legacy_candidates,
        inlier,
        3,
        config,
        particle_fourier=fourier,
    )
    assert any(shift != (0.0, 0.0) for shift in center_shifts)
    _apply_reference_center_shifts(legacy_candidates, center_shifts)
    first, first_weights, _ = legacy_updater(
        images,
        legacy_candidates,
        inlier,
        3,
        config,
        subset=first_half,
        particle_fourier=fourier,
    )
    second, second_weights, _ = legacy_updater(
        images,
        legacy_candidates,
        inlier,
        3,
        config,
        subset=~first_half,
        particle_fourier=fourier,
    )

    shared = shared_updater(
        images,
        candidates,
        inlier,
        3,
        config,
        first_half=first_half,
        particle_fourier=fourier,
    )

    assert np.allclose(shared.references, full, atol=atol)
    assert np.allclose(shared.effective_weights, full_weights, atol=1e-6)
    assert shared.center_shifts == center_shifts
    assert np.allclose(shared.half_references[0], first, atol=atol)
    assert np.allclose(shared.half_references[1], second, atol=atol)
    assert np.allclose(shared.half_weights[0], first_weights, atol=1e-6)
    assert np.allclose(shared.half_weights[1], second_weights, atol=1e-6)
    assert np.all(shared.references[2] == 0.0)
    assert np.all(shared.half_references[:, 2] == 0.0)


def test_cpu_shared_fourier_workflow_matches_legacy_halfset_updates():
    images, _, _, _ = images_and_candidates()
    references = np.stack((images[0], images[3]))
    config = ai.AlignmentConfig(
        max_iterations=2,
        angle_samples=24,
        proposal_angles_per_reference=4,
        translation_range=2,
        top_l=4,
        robust_weighting=True,
        center_references=True,
        mask_radius=12.0,
        mask_soft_edge=3.0,
        lowpass_sigma=0.5,
        halfset_diagnostics=True,
        random_seed=5,
    ).normalized(workflow="global")
    shared = run_soft_alignment_cpu(
        images,
        references,
        config=config,
        class_priors=None,
        initial_poses=None,
        workflow="global",
    )
    legacy = run_soft_alignment_cpu(
        images,
        references,
        config=config,
        class_priors=None,
        initial_poses=None,
        workflow="global",
        _reference_updater=_update_references_fourier,
    )

    assert shared.metadata["halfset_update_policy"] == "shared_unnormalized_accumulation"
    assert legacy.metadata["halfset_update_policy"] == "separate_reference_updates"
    assert np.array_equal(shared.reference_assignments, legacy.reference_assignments)
    assert np.allclose(shared.responsibilities, legacy.responsibilities, atol=2e-6)
    assert np.allclose(shared.references, legacy.references, atol=3e-6)
    assert np.allclose(shared.reference_history, legacy.reference_history, atol=3e-6)
    for current, baseline in zip(shared.diagnostics, legacy.diagnostics):
        assert np.allclose(
            current["effective_component_weight"],
            baseline["effective_component_weight"],
            atol=2e-6,
        )
        assert np.allclose(
            current["halfset_effective_weight"],
            baseline["halfset_effective_weight"],
            atol=2e-6,
        )
        assert np.allclose(current["frc"], baseline["frc"], atol=2e-6)
        assert np.allclose(
            current["frc_0143_stable_cutoff_cyc_per_px"],
            baseline["frc_0143_stable_cutoff_cyc_per_px"],
            atol=2e-6,
        )


def test_cpu_shared_fourier_mstep_matches_legacy_adaptive_ragged_posterior():
    images, _, _, _ = images_and_candidates()
    references = np.stack((images[0], images[3]))
    initial_poses = ai.PoseSet(
        angle_deg=np.zeros(len(images), dtype=np.float32),
        shift_y_px=np.zeros(len(images), dtype=np.float32),
        shift_x_px=np.zeros(len(images), dtype=np.float32),
        mirror=np.zeros(len(images), dtype=np.bool_),
    )
    config = ai.AlignmentConfig(
        max_iterations=1,
        search_strategy="adaptive_posterior",
        coarse_angle_step=12.0,
        coarse_shift_step=1.0,
        local_angle_range=6.0,
        local_shift_range=1.0,
        adaptive_fraction=0.95,
        oversampling_order=1,
        max_adaptive_cells=4,
        top_l=4,
        robust_weighting=True,
        center_references=True,
        halfset_diagnostics=True,
        random_seed=3,
    ).normalized(workflow="refine")
    shared = run_soft_alignment_cpu(
        images,
        references,
        config=config,
        class_priors=None,
        initial_poses=initial_poses,
        workflow="refine",
    )
    legacy = run_soft_alignment_cpu(
        images,
        references,
        config=config,
        class_priors=None,
        initial_poses=initial_poses,
        workflow="refine",
        _reference_updater=_update_references_fourier,
    )

    assert np.array_equal(shared.reference_assignments, legacy.reference_assignments)
    assert np.allclose(shared.candidates.posterior, legacy.candidates.posterior, atol=2e-6)
    assert np.allclose(shared.references, legacy.references, atol=3e-6)
    assert np.allclose(
        shared.diagnostics[0]["halfset_effective_weight"],
        legacy.diagnostics[0]["halfset_effective_weight"],
        atol=2e-6,
    )
    assert np.allclose(
        shared.diagnostics[0]["frc"], legacy.diagnostics[0]["frc"], atol=2e-6
    )
