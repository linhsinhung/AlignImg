from __future__ import annotations

import argparse
import json
from pathlib import Path

import mrcfile
import numpy as np

import alignimg as ai
from alignimg._geometry import CENTER_CONVENTION
from tools.prepare_re2dc_70s_pose_benchmark import (
    relion_to_alignimg_poses,
    select_disjoint_splits,
)
from tools.re2dc_70s_pose_benchmark import (
    compose_pose_gauge,
    fit_pose_gauge,
    main,
    pose_error_metrics,
)


def test_relion_pose_conversion_uses_inverse_angle_and_rotated_origin():
    poses = relion_to_alignimg_poses(
        np.asarray([90.0]),
        np.asarray([4.0]),
        np.asarray([6.0]),
        pixel_size_angstrom=2.0,
    )

    assert np.allclose(poses.angle_deg, [-90.0])
    assert np.allclose(poses.shift_y_px, [2.0], atol=1e-6)
    assert np.allclose(poses.shift_x_px, [-3.0], atol=1e-6)
    assert not np.any(poses.mirror)


def test_disjoint_pose_benchmark_split_is_reproducible():
    labels = np.repeat([4, 7], 10)
    confidence = np.linspace(0.9, 1.0, 20)
    arguments = dict(
        classes=(4, 7),
        reference_count=3,
        evaluation_count=4,
        minimum_confidence=0.9,
        seed=19,
    )

    first_reference, first_evaluation = select_disjoint_splits(
        labels, confidence, **arguments
    )
    second_reference, second_evaluation = select_disjoint_splits(
        labels, confidence, **arguments
    )

    assert np.array_equal(first_reference, second_reference)
    assert np.array_equal(first_evaluation, second_evaluation)
    assert np.intersect1d(first_reference, first_evaluation).size == 0
    assert np.all(labels[first_reference[0]] == 4)
    assert np.all(labels[first_evaluation[1]] == 7)


def test_disjoint_pose_benchmark_split_rejects_insufficient_class():
    with np.testing.assert_raises_regex(ValueError, "class 4 has 3 particles"):
        select_disjoint_splits(
            np.asarray([4, 4, 4]),
            np.ones(3),
            classes=(4,),
            reference_count=2,
            evaluation_count=2,
            minimum_confidence=0.9,
            seed=0,
        )


def test_pose_gauge_fit_removes_one_global_se2_transform():
    reference = ai.PoseSet(
        np.asarray([-150.0, -20.0, 0.0, 80.0, 170.0]),
        np.asarray([1.0, -2.0, 3.0, 0.0, -1.0]),
        np.asarray([-4.0, 2.0, 1.0, 5.0, 0.0]),
        np.zeros(5, dtype=bool),
    )
    gauge = ai.PoseSet(
        np.asarray([37.0]),
        np.asarray([-2.5]),
        np.asarray([1.25]),
        np.asarray([False]),
    )
    predicted = compose_pose_gauge(reference, gauge)

    fitted = fit_pose_gauge(predicted, reference)
    metrics = pose_error_metrics(predicted, reference)

    assert np.allclose(fitted.angle_deg, gauge.angle_deg, atol=1e-6)
    assert np.allclose(fitted.shift_y_px, gauge.shift_y_px, atol=1e-6)
    assert np.allclose(fitted.shift_x_px, gauge.shift_x_px, atol=1e-6)
    assert metrics["angle_absolute_error_deg"]["maximum"] < 1e-6
    assert metrics["shift_error_px"]["maximum"] < 1e-6


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


def write_synthetic_benchmark(tmp_path: Path) -> Path:
    size = 16
    y, x = np.indices((size, size), dtype=np.float32)
    references = np.stack(
        (
            np.exp(-((y - 5) ** 2 + (x - 10) ** 2) / 4.0),
            np.exp(-((y - 11) ** 2 + (x - 5) ** 2) / 3.0),
        )
    ).astype(np.float32)
    component = np.repeat(np.arange(2, dtype=np.int32), 4)
    particles = references[component]
    particle_path = tmp_path / "benchmark.particles.mrcs"
    reference_path = tmp_path / "benchmark.references.mrcs"
    truth_path = tmp_path / "benchmark.truth.npz"
    manifest_path = tmp_path / "benchmark.json"
    for path, values in ((particle_path, particles), (reference_path, references)):
        with mrcfile.new(path) as mrc:
            mrc.set_data(values)
            mrc.voxel_size = 2.5
    np.savez_compressed(
        truth_path,
        center_convention=np.asarray(CENTER_CONVENTION),
        pixel_size_angstrom=np.asarray(2.5),
        relion_class_numbers=np.asarray([4, 7], dtype=np.int32),
        component_index=component,
        relion_confidence=np.ones(len(particles)),
        relion_pose_angle_deg=np.zeros(len(particles), dtype=np.float32),
        relion_pose_shift_y_px=np.zeros(len(particles), dtype=np.float32),
        relion_pose_shift_x_px=np.zeros(len(particles), dtype=np.float32),
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "alignimg.re2dc-70s-pose-benchmark-data.v1",
                "artifacts": {
                    "particles": str(particle_path),
                    "references": str(reference_path),
                    "truth": str(truth_path),
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_pose_benchmark_runner_writes_all_workflows(tmp_path: Path):
    manifest_path = write_synthetic_benchmark(tmp_path)
    output_path = tmp_path / "pose-benchmark.json"
    args = argparse.Namespace(
        benchmark=manifest_path,
        output=output_path,
        backend="cpu",
        batch_size=8,
        memory_fraction=0.8,
        angle_samples=12,
        proposal_angles=2,
        top_l=2,
        translation_range=1.0,
        global_iterations=1,
        rf_iterations=1,
        mra_iterations=1,
        adaptive_iterations=1,
        coarse_angle_step=6.0,
        coarse_shift_step=1.0,
        local_angle_range=6.0,
        local_shift_range=1.0,
        adaptive_fraction=0.99,
        oversampling_order=1,
        max_adaptive_cells=8,
        rescue_uncertain_particles=False,
        rescue_normalized_entropy_threshold=None,
        rescue_map_posterior_threshold=None,
        rescue_max_fraction=0.1,
        rescue_min_score_improvement=0.02,
        rf_seeds=[0],
        deterministic_repeats=2,
        only=None,
    )

    main(args)

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["parameters"]["candidate_scoring"] == "raster"
    assert report["parameters"]["score_model"] == "fourier_ncc"
    assert report["parameters"]["reference_update"] == "spatial"
    assert set(report["runs"]) == {
        "oracle",
        "known_reference",
        "homogeneous_rf",
        "homogeneous_rf_summary",
        "fixed_mra",
        "open_mra",
        "adaptive_known_reference",
        "adaptive_fixed_mra",
    }
    assert report["runs"]["fixed_mra"]["assignment_accuracy"] == 1.0
    assert report["runs"]["adaptive_fixed_mra"]["assignment_accuracy"] == 1.0
    assert report["runs"]["adaptive_fixed_mra"]["config"][
        "search_strategy"
    ] == "adaptive_posterior"
    assert report["runs"]["adaptive_fixed_mra"]["initializer"][
        "assignment_accuracy"
    ] == 1.0
    assert all(
        item["maximum_angle_delta_deg"] == 0.0
        and item["maximum_shift_delta_px"] == 0.0
        and item["reference_correlation"] > 0.999999
        for item in report["runs"]["adaptive_fixed_mra"][
            "fixed_mra_k1_equivalence"
        ]
    )
    assert len(report["runs"]["known_reference"]) == 2
    assert len(report["runs"]["homogeneous_rf"]) == 2
    for run in report["runs"]["known_reference"]:
        assert run["reproducibility_repeats"][0]["comparison_to_primary"][
            "assignment_agreement"
        ] == 1.0
        with mrcfile.open(run["artifacts"]["references"]) as mrc:
            assert np.isclose(float(mrc.voxel_size.x), 2.5)
    for run in report["runs"]["adaptive_known_reference"]:
        assert run["config"]["search_strategy"] == "adaptive_posterior"
        assert run["initializer"]["assignment_accuracy"] == 1.0
        with np.load(run["artifacts"]["result"], allow_pickle=False) as saved:
            assert saved["pose_entropy"].shape == (4,)
            assert saved["map_posterior"].shape == (4,)
