from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np

from tools.align_known_reference import main


def test_known_reference_runner_writes_pose_and_average_artifacts(tmp_path: Path):
    size = 16
    y, x = np.indices((size, size), dtype=np.float32)
    reference = np.exp(-((y - 5) ** 2 + (x - 10) ** 2) / 4.0).astype(np.float32)
    particles = np.repeat(reference[None], 3, axis=0)
    particle_path = tmp_path / "particles.mrcs"
    reference_path = tmp_path / "reference.mrc"
    output = tmp_path / "result"
    with mrcfile.new(particle_path) as mrc:
        mrc.set_data(particles)
        mrc.voxel_size = 2.0
    with mrcfile.new(reference_path) as mrc:
        mrc.set_data(reference)

    main(
        [
            "--particles",
            str(particle_path),
            "--reference",
            str(reference_path),
            "--output-directory",
            str(output),
            "--backend",
            "cpu",
            "--batch-size",
            "4",
            "--global-iterations",
            "1",
            "--refine-iterations",
            "1",
            "--angle-samples",
            "12",
            "--proposal-angles",
            "2",
            "--translation-range",
            "1",
        ]
    )

    with np.load(output / "result.npz", allow_pickle=False) as result:
        assert result["angle_deg"].shape == (3,)
        assert result["responsibilities"].shape == (3, 1)
        assert result["candidate_posterior"].shape[0] == 3
        assert result["reference_history"].shape == (3, 1, size, size)
        assert np.all(result["assignments"] == 0)
        assert np.allclose(result["responsibilities"], 1.0)
        assert np.array_equal(result["class_averages"], result["soft_references"])
    with mrcfile.open(output / "class_average.mrc") as mrc:
        assert np.asarray(mrc.data).shape == (size, size)
    report = __import__("json").loads((output / "report.json").read_text())
    assert report["status"] == "completed"
    assert report["summary"]["assignment_values"] == [0]
    assert report["summary"]["responsibility_sum_max_error"] < 1e-6
    assert report["summary"]["class_average_estimator"] == ("soft_posterior_reference")
    assert report["parameters"]["refine_config"]["search_strategy"] == (
        "quadratic_refine"
    )
    assert report["parameters"]["refine_config"]["coarse_angle_step"] == 1.0
    assert report["parameters"]["refine_config"]["local_angle_range"] == 7.0
    assert report["parameters"]["refine_config"]["local_shift_range"] == 3.0


def test_known_reference_runner_can_reconstruct_final_average_from_raw(tmp_path: Path):
    size = 16
    y, x = np.indices((size, size), dtype=np.float32)
    reference = np.exp(-((y - 5) ** 2 + (x - 10) ** 2) / 4.0).astype(np.float32)
    particles = np.repeat(reference[None], 3, axis=0)
    particle_path = tmp_path / "particles.mrcs"
    reference_path = tmp_path / "reference.mrc"
    output = tmp_path / "raw-result"
    with mrcfile.new(particle_path) as mrc:
        mrc.set_data(particles)
        mrc.voxel_size = 2.0
    with mrcfile.new(reference_path) as mrc:
        mrc.set_data(reference)

    main(
        [
            "--particles",
            str(particle_path),
            "--reference",
            str(reference_path),
            "--output-directory",
            str(output),
            "--backend",
            "cpu",
            "--batch-size",
            "2",
            "--global-iterations",
            "1",
            "--refine-iterations",
            "0",
            "--angle-samples",
            "12",
            "--proposal-angles",
            "2",
            "--translation-range",
            "1",
            "--apply-final-pose-to-raw",
        ]
    )

    with np.load(output / "result.npz", allow_pickle=False) as result:
        class_averages = result["class_averages"]
        assert class_averages.shape == (1, size, size)
        assert np.all(np.isfinite(class_averages))
        assert result["soft_references"].shape == class_averages.shape
    with mrcfile.open(output / "class_average.mrc") as mrc:
        assert np.array_equal(np.asarray(mrc.data), class_averages[0])
    report = __import__("json").loads((output / "report.json").read_text())
    assert report["summary"]["class_average_estimator"] == (
        "final_map_pose_inlier_weighted_raw"
    )
