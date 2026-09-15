"""Small deterministic pipeline test for the supported CPU workflow."""

from __future__ import annotations

import numpy as np

import alignimg as ai


def test_small_synthetic_alignment_pipeline_runs_end_to_end():
    n = 5
    size = 40
    y, x = np.indices((size, size), dtype=np.float32)

    base = np.exp(-(((y - 16.0) ** 2) + ((x - 22.0) ** 2)) / (2.0 * 3.0 ** 2))
    base += 0.5 * np.exp(-(((y - 26.0) ** 2) + ((x - 14.0) ** 2)) / (2.0 * 2.0 ** 2))
    base = base.astype(np.float32)

    poses = ai.PoseSet(
        angle_deg=np.asarray([-4.0, 5.0, 0.0, -4.0, 5.0], dtype=np.float32),
        shift_y_px=np.asarray([1.0, -1.0, 0.0, 1.0, -1.0], dtype=np.float32),
        shift_x_px=np.asarray([-1.0, 2.0, 0.0, -1.0, 2.0], dtype=np.float32),
        mirror=np.zeros(n, dtype=bool),
    )
    X = ai.transform_images(np.repeat(base[None], n, axis=0), poses)

    result = ai.align_to_references(
        X,
        base,
        backend="cpu",
        config=ai.AlignmentConfig(
            max_iterations=1,
            top_l=2,
            angle_samples=36,
            translation_range=2,
            halfset_diagnostics=False,
            center_references=False,
        ),
    )
    corrected = ai.transform_images(X, result.poses)

    assert result.references.shape == (1, size, size)
    assert np.asarray(result.reference_history).shape == (2, 1, size, size)
    assert len(result.poses) == n
    assert corrected.shape == X.shape
    assert result.metadata["backend"] == "cpu"
    assert result.metadata["config"]["candidate_scoring"] == "fourier"
    assert result.metadata["config"]["reference_update"] == "fourier"
    assert np.all(np.isfinite(result.references))
    assert np.all(
        np.isfinite(
            np.stack(
                (
                    result.poses.angle_deg,
                    result.poses.shift_y_px,
                    result.poses.shift_x_px,
                ),
                axis=1,
            )
        )
    )
    assert np.all(np.isfinite(corrected))
