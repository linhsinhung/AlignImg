"""Fixed MRA remains equivalent to independent single-reference alignment."""

import numpy as np

import alignimg as ai


def test_fixed_mra_candidates_match_independent_k1_searches():
    size = 24
    y, x = np.indices((size, size), dtype=np.float32)
    references = np.stack(
        (
            np.exp(-((y - 7) ** 2 + (x - 15) ** 2) / 5.0)
            + 0.6 * np.exp(-((y - 17) ** 2 + (x - 9) ** 2) / 3.0),
            np.exp(-((y - 16) ** 2 + (x - 16) ** 2) / 4.0)
            + 0.7 * np.exp(-((y - 6) ** 2 + (x - 8) ** 2) / 5.0),
        )
    ).astype(np.float32)
    assignments = np.repeat(np.arange(2, dtype=np.int32), 4)
    particles = references[assignments]
    config = ai.AlignmentConfig(
        max_iterations=1,
        top_l=8,
        angle_samples=36,
        proposal_angles_per_reference=8,
        translation_range=3.0,
        halfset_diagnostics=False,
        center_references=False,
        batch_size=8,
    )
    priors = ai.make_class_priors(
        assignments=assignments,
        n_components=2,
        trust=1.0,
    )

    joint = ai.align_to_references(
        particles,
        references,
        class_priors=priors,
        config=config,
        backend="cpu",
    )

    for component in range(2):
        selected = assignments == component
        independent = ai.align_to_references(
            particles[selected],
            references[component],
            config=config,
            backend="cpu",
        )
        assert np.all(joint.candidates.reference_index[selected] == component)
        for field in (
            "angle_deg",
            "shift_y_px",
            "shift_x_px",
            "score",
            "posterior",
        ):
            assert np.allclose(
                getattr(joint.candidates, field)[selected],
                getattr(independent.candidates, field),
                atol=1e-7,
            )
        assert np.array_equal(
            joint.candidates.mirror[selected], independent.candidates.mirror
        )
        assert np.allclose(
            joint.references[component], independent.references[0], atol=1e-7
        )


def test_adaptive_fixed_mra_matches_independent_k1_refinements():
    size = 24
    y, x = np.indices((size, size), dtype=np.float32)
    references = np.stack(
        (
            np.exp(-((y - 7) ** 2 + (x - 15) ** 2) / 5.0)
            + 0.6 * np.exp(-((y - 17) ** 2 + (x - 9) ** 2) / 3.0),
            np.exp(-((y - 16) ** 2 + (x - 16) ** 2) / 4.0)
            + 0.7 * np.exp(-((y - 6) ** 2 + (x - 8) ** 2) / 5.0),
        )
    ).astype(np.float32)
    assignments = np.repeat(np.arange(2, dtype=np.int32), 4)
    particles = references[assignments]
    initial = ai.PoseSet.identity(len(particles))
    config = ai.AlignmentConfig(
        search_strategy="adaptive_posterior",
        max_iterations=3,
        top_l=8,
        angle_samples=24,
        proposal_angles_per_reference=4,
        translation_range=1.0,
        coarse_angle_step=6.0,
        coarse_shift_step=1.0,
        local_angle_range=6.0,
        local_shift_range=1.0,
        adaptive_fraction=0.99,
        oversampling_order=1,
        max_adaptive_cells=8,
        rescue_uncertain_particles=True,
        rescue_normalized_entropy_threshold=0.0,
        rescue_max_fraction=0.25,
        rescue_min_score_improvement=10.0,
        temperature_start=0.05,
        temperature_end=0.05,
        halfset_diagnostics=False,
        center_references=False,
        batch_size=8,
    )
    priors = ai.make_class_priors(
        assignments=assignments,
        n_components=2,
        trust=1.0,
    )

    joint = ai.refine_alignment(
        particles,
        references,
        initial,
        class_priors=priors,
        config=config,
        backend="cpu",
    )

    assert np.array_equal(joint.reference_assignments, assignments)
    assert joint.diagnostics[0]["rescue_scheduled_count"] == 2
    assert joint.diagnostics[1]["rescue_particle_count"] == 2
    for component in range(2):
        selected = assignments == component
        independent = ai.refine_alignment(
            particles[selected],
            references[component],
            ai.PoseSet.identity(int(np.sum(selected))),
            config=config,
            backend="cpu",
        )
        assert independent.diagnostics[0]["rescue_scheduled_count"] == 1
        assert independent.diagnostics[1]["rescue_particle_count"] == 1
        assert np.all(joint.candidates.reference_index[selected] == component)
        for field in (
            "angle_deg",
            "shift_y_px",
            "shift_x_px",
            "score",
            "posterior",
        ):
            assert np.allclose(
                getattr(joint.candidates, field)[selected],
                getattr(independent.candidates, field),
                atol=1e-7,
            )
        assert np.allclose(
            joint.references[component], independent.references[0], atol=1e-7
        )
