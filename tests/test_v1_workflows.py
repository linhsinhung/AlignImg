"""Contract and scientific smoke tests for the AlignImg 2.x engine."""

from __future__ import annotations

import numpy as np
import pytest

import alignimg as ai
from alignimg._engine import _stable_frc_cutoff
from alignimg._geometry import mirror_x_integer_origin
from alignimg._transform import transform_image
from alignimg._fourier import _candidate_angles, prepare_stack


def asymmetric_reference(size: int = 32) -> np.ndarray:
    y, x = np.indices((size, size), dtype=np.float32)
    result = np.exp(-((y - 10) ** 2 + (x - 18) ** 2) / 10.0)
    result += 0.7 * np.exp(-((y - 22) ** 2 + (x - 11) ** 2) / 5.0)
    return result.astype(np.float32)


def fast_config(**changes) -> ai.AlignmentConfig:
    values = dict(
        max_iterations=1,
        top_l=4,
        angle_samples=72,
        translation_range=5,
        halfset_diagnostics=False,
        center_references=False,
        mask_soft_edge=2,
    )
    values.update(changes)
    return ai.AlignmentConfig(**values)


def test_proposal_angle_budget_is_independent_and_validated():
    config = ai.AlignmentConfig(
        top_l=8, angle_samples=72, proposal_angles_per_reference=6
    ).normalized()
    assert config.proposal_angles_per_reference == 6
    with pytest.raises(ValueError, match="proposal_angles_per_reference"):
        ai.AlignmentConfig(
            angle_samples=72, proposal_angles_per_reference=73
        ).normalized()


def test_workflow_presets_are_explicit_and_distinct():
    balanced = ai.AlignmentConfig.preset("global_balanced")
    accurate = ai.AlignmentConfig.preset("global_accurate")
    reference_free = ai.AlignmentConfig.preset("reference_free")
    refine = ai.AlignmentConfig.preset("refine")
    assert balanced.proposal_angles_per_reference == 6
    assert accurate.proposal_angles_per_reference == 8
    assert balanced.temperature_end == accurate.temperature_end == 0.05
    assert reference_free.top_l == 4
    assert reference_free.temperature_end == 0.02
    assert reference_free.temperature_anneal_iterations == 10
    assert refine.robust_weighting is True
    with pytest.raises(ValueError, match="unknown alignment preset"):
        ai.AlignmentConfig.preset("unknown")


def test_canonical_transform_is_mirror_rotate_shift_with_wrap():
    image = np.zeros((16, 16), dtype=np.float32)
    image[4, 2] = 1.0
    poses = ai.PoseSet(
        angle_deg=np.array([0], dtype=np.float32),
        shift_y_px=np.array([0], dtype=np.float32),
        shift_x_px=np.array([3], dtype=np.float32),
        mirror=np.array([True]),
    )
    transformed = ai.transform_images(image[None], poses)[0]
    assert np.unravel_index(np.argmax(transformed), transformed.shape) == (4, 1)


def test_global_fourier_alignment_recovers_high_snr_pose():
    reference = asymmetric_reference()
    particle = transform_image(reference, angle_deg=30, shift_y_px=2, shift_x_px=-1)
    result = ai.align_to_references(particle[None], reference, config=fast_config())
    aligned = ai.transform_images(particle[None], result.poses)[0]
    correlation = np.corrcoef(aligned.ravel(), reference.ravel())[0, 1]

    assert result.references.shape == (1, 32, 32)
    assert result.responsibilities.shape == (1, 1)
    assert result.candidates.score.shape == (1, 4)
    assert np.allclose(result.responsibilities.sum(axis=1), 1.0)
    assert abs(float(result.poses.angle_deg[0]) + 30.0) <= 5.0
    assert correlation > 0.95


def test_polar_proposals_use_inverse_angle_and_keep_180_degree_pair():
    reference = asymmetric_reference(48)
    config = fast_config(angle_samples=72)
    prepared_reference = prepare_stack(reference[None], config)
    for source_angle in (-120.0, -35.0, 65.0, 160.0):
        particle = transform_image(
            reference, angle_deg=source_angle, shift_y_px=2, shift_x_px=-1
        )
        prepared_particle = prepare_stack(particle[None], config)
        proposals = _candidate_angles(
            prepared_particle.polar[0], prepared_reference.polar[0], 4
        )
        errors = [
            abs((angle + source_angle + 180.0) % 360.0 - 180.0)
            for angle in proposals
        ]
        assert min(errors) <= 5.0
        pair_delta = (proposals[1] - proposals[0]) % 360.0
        assert pair_delta == pytest.approx(180.0)


def test_class_priors_can_fix_multi_reference_assignments():
    first = asymmetric_reference(24)
    second = np.rot90(first).copy()
    images = np.stack((first, second)).astype(np.float32)
    priors = np.eye(2, dtype=np.float32)
    result = ai.align_to_references(
        images,
        np.stack((first, second)),
        class_priors=priors,
        config=fast_config(top_l=2, angle_samples=36, translation_range=2),
    )
    assert np.array_equal(result.reference_assignments, np.array([0, 1]))
    assert np.array_equal(result.responsibilities > 0, priors.astype(bool))


def test_make_class_priors_supports_fixed_and_corrective_feedback():
    assignments = np.array([0, 2, 1], dtype=np.int32)
    fixed = ai.make_class_priors(assignments=assignments, n_components=3)
    corrective = ai.make_class_priors(
        assignments=assignments, n_components=3, trust=0.9
    )
    soft = ai.make_class_priors(
        responsibilities=np.array([[2.0, 1.0], [0.0, 4.0]], dtype=np.float32),
        trust=0.5,
    )

    assert np.array_equal(np.argmax(fixed, axis=1), assignments)
    assert np.array_equal(fixed > 0.0, np.eye(3, dtype=bool)[assignments])
    assert np.allclose(corrective[np.arange(3), assignments], 0.9 + 0.1 / 3)
    assert np.all(corrective > 0.0)
    assert np.allclose(corrective.sum(axis=1), 1.0)
    assert np.allclose(soft, [[7 / 12, 5 / 12], [1 / 4, 3 / 4]])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({}, "exactly one"),
        (
            {
                "assignments": np.array([0]),
                "responsibilities": np.ones((1, 1)),
            },
            "exactly one",
        ),
        ({"assignments": np.array([0.0]), "n_components": 1}, "integer"),
        ({"assignments": np.array([1]), "n_components": 1}, "range"),
        ({"responsibilities": np.zeros((1, 2))}, "positive sum"),
        ({"responsibilities": np.ones((1, 2)), "n_components": 3}, "match"),
        ({"assignments": np.array([0]), "n_components": 1, "trust": 1.1}, "trust"),
    ],
)
def test_make_class_priors_rejects_invalid_feedback(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ai.make_class_priors(**kwargs)


def test_corrective_feedback_can_recover_deliberately_wrong_assignment():
    first = asymmetric_reference(24)
    y, x = np.indices((24, 24), dtype=np.float32)
    second = (
        np.exp(-((y - 7) ** 2 + (x - 7) ** 2) / 5.0)
        + 0.5 * np.exp(-((y - 17) ** 2 + (x - 18) ** 2) / 3.0)
    ).astype(np.float32)
    references = np.stack((first, second))
    truth = np.array([0, 0, 1, 1], dtype=np.int32)
    images = references[truth]
    feedback = np.array([1, 0, 1, 1], dtype=np.int32)
    config = fast_config(
        top_l=4,
        angle_samples=72,
        proposal_angles_per_reference=4,
        translation_range=1,
        temperature_start=0.02,
        temperature_end=0.02,
    )
    fixed = ai.refine_alignment(
        images,
        references,
        ai.PoseSet.identity(4),
        class_priors=ai.make_class_priors(
            assignments=feedback, n_components=2, trust=1.0
        ),
        config=config,
    )
    corrective = ai.refine_alignment(
        images,
        references,
        ai.PoseSet.identity(4),
        class_priors=ai.make_class_priors(
            assignments=feedback, n_components=2, trust=0.5
        ),
        config=config,
    )

    assert np.array_equal(fixed.reference_assignments, feedback)
    assert np.mean(corrective.reference_assignments == truth) > np.mean(
        fixed.reference_assignments == truth
    )


def test_reference_free_bootstrap_is_seeded_and_records_components():
    first = asymmetric_reference(24)
    second = np.flipud(first).copy()
    images = np.stack((first, np.roll(first, 1, 0), second, np.roll(second, -1, 1)))
    config = fast_config(top_l=2, angle_samples=36, translation_range=2, random_seed=7)
    one = ai.reference_free_align(images, n_components=2, config=config)
    two = ai.reference_free_align(images, n_components=2, config=config)

    assert one.references.shape == (2, 24, 24)
    assert np.array_equal(one.metadata["bootstrap_labels"], two.metadata["bootstrap_labels"])
    assert np.allclose(one.references, two.references)
    assert "component_weight_cv" in one.diagnostics[-1]
    assert "maximum_offdiagonal_reference_correlation" in one.diagnostics[-1]


def test_refine_uses_robust_weighting():
    reference = asymmetric_reference(24)
    images = np.stack((reference, reference, np.zeros_like(reference)))
    result = ai.refine_alignment(
        images,
        reference,
        ai.PoseSet.identity(3),
        config=fast_config(robust_weighting=True, top_l=2, angle_samples=36),
    )
    assert result.inlier_weights.shape == (3,)
    assert result.inlier_weights[-1] < result.inlier_weights[0]


def test_fixed_class_padding_does_not_create_nan_diagnostics():
    first = asymmetric_reference(24)
    second = np.flipud(first).copy()
    images = np.stack((first, second))
    priors = ai.make_class_priors(
        assignments=np.array([0, 1], dtype=np.int32),
        n_components=2,
        trust=1.0,
    )
    result = ai.refine_alignment(
        images,
        images,
        ai.PoseSet.identity(2),
        class_priors=priors,
        config=fast_config(
            top_l=8,
            angle_samples=36,
            proposal_angles_per_reference=6,
            translation_range=2,
        ),
    )
    assert np.all(np.isfinite(result.candidates.posterior))
    assert np.isfinite(result.diagnostics[-1]["mean_expected_fourier_ncc"])


def test_mirror_search_is_explicit_and_recoverable():
    reference = asymmetric_reference(24)
    particle = mirror_x_integer_origin(reference)
    result = ai.align_to_references(
        particle[None],
        reference,
        config=fast_config(
            top_l=4, angle_samples=36, translation_range=2, mirror_search=True
        ),
    )
    aligned = ai.transform_images(particle[None], result.poses)[0]
    assert bool(result.poses.mirror[0])
    assert np.corrcoef(aligned.ravel(), reference.ravel())[0, 1] > 0.99


def test_temperature_history_and_halfset_diagnostics_are_recorded():
    reference = asymmetric_reference(20)
    images = np.stack((reference, np.roll(reference, 1, 0)))
    config = fast_config(
        max_iterations=2,
        top_l=2,
        angle_samples=24,
        translation_range=1,
        temperature_start=0.1,
        temperature_end=0.025,
        halfset_diagnostics=True,
    )
    result = ai.align_to_references(images, reference, config=config)
    assert len(result.reference_history) == 3
    assert [item["temperature"] for item in result.diagnostics] == [0.1, 0.025]
    assert result.diagnostics[0]["frc"].shape == (1, 10)
    assert result.diagnostics[0]["frc_0143_stable_cutoff_cyc_per_px"].shape == (1,)
    assert result.metadata["halfset_membership"].shape == (2,)


def test_temperature_schedule_can_anneal_then_hold():
    reference = asymmetric_reference(20)
    config = fast_config(
        max_iterations=5,
        top_l=2,
        angle_samples=24,
        translation_range=1,
        temperature_start=0.08,
        temperature_end=0.02,
        temperature_anneal_iterations=3,
    )
    result = ai.align_to_references(reference[None], reference, config=config)
    assert [item["temperature"] for item in result.diagnostics] == pytest.approx(
        [0.08, 0.04, 0.02, 0.02, 0.02]
    )
    with pytest.raises(ValueError, match="temperature_anneal_iterations"):
        ai.AlignmentConfig(
            max_iterations=3, temperature_anneal_iterations=4
        ).normalized()


def test_stable_frc_cutoff_ignores_isolated_late_threshold_crossing():
    curve = np.array(
        [1.0, 0.8, 0.5, 0.2, 0.08, 0.04, 0.02, 0.3, 0.01, 0.0],
        dtype=np.float32,
    )
    stable = _stable_frc_cutoff(curve, size=20)
    legacy = np.flatnonzero(curve >= 0.143)[-1] / 20
    assert 0.15 < stable < 0.3
    assert stable < legacy


def test_refine_improves_pose_alignment_from_nearby_initial_pose():
    reference = asymmetric_reference(24)
    particle = transform_image(reference, angle_deg=5, shift_y_px=2, shift_x_px=-2)
    config = fast_config(
        top_l=4,
        angle_samples=72,
        proposal_angles_per_reference=5,
        translation_range=3,
        pose_angle_sigma=15,
        pose_shift_sigma=3,
    )
    before = np.corrcoef(particle.ravel(), reference.ravel())[0, 1]
    result = ai.refine_alignment(
        particle[None], reference, ai.PoseSet.identity(1), config=config
    )
    aligned = ai.transform_images(particle[None], result.poses)[0]
    after = np.corrcoef(aligned.ravel(), reference.ravel())[0, 1]
    assert after > before + 0.2
    assert after > 0.85


def test_known_reference_alignment_improves_average_at_snr_point_two():
    reference = asymmetric_reference()
    rng = np.random.default_rng(2)
    particles = []
    for angle in (-30, -20, -10, 0, 10, 20, 30, 40):
        signal = transform_image(reference, angle_deg=angle)
        noise_sigma = np.sqrt(np.var(signal) / 0.2)
        particles.append(signal + rng.normal(0, noise_sigma, signal.shape))
    particles = np.asarray(particles, dtype=np.float32)
    result = ai.align_to_references(
        particles,
        reference,
        config=fast_config(translation_range=2),
    )
    aligned = ai.transform_images(particles, result.poses)
    before = np.corrcoef(particles.mean(axis=0).ravel(), reference.ravel())[0, 1]
    after = np.corrcoef(aligned.mean(axis=0).ravel(), reference.ravel())[0, 1]
    assert after > before + 0.15


def test_v1_input_validation_and_explicit_gpu_failure():
    reference = asymmetric_reference(16)
    with pytest.raises(ValueError, match="square"):
        ai.align_to_references(
            np.zeros((2, 16, 12)), np.zeros((1, 16, 12)), config=fast_config()
        )
    with pytest.raises(ValueError, match="class_priors"):
        ai.align_to_references(
            reference[None], reference, class_priors=np.zeros((1, 1)), config=fast_config()
        )
    if not ai.available_alignment_backends()["gpu"]["available"]:
        with pytest.raises(RuntimeError, match="alignimg-gpu"):
            ai.align_to_references(
                reference[None], reference, config=fast_config(), backend="gpu"
            )


def test_legacy_pose_adaptor_is_explicit_and_rejects_mirror():
    legacy = np.array([[10.0, 2.0, -1.0, 0.9]], dtype=np.float32)
    poses = ai.poses_from_legacy_params(legacy)
    restored = ai.poses_to_legacy_params(poses, legacy[:, 3])
    assert np.allclose(restored, legacy)
    with pytest.raises(ValueError, match="mirror"):
        ai.poses_to_legacy_params(
            ai.PoseSet(poses.angle_deg, poses.shift_y_px, poses.shift_x_px, np.ones(1, dtype=bool))
        )
