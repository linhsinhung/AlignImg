"""Optional real-data integration test for the public API.

Run with:
    RUN_ALIGNIMG_INTEGRATION=1 python -m pytest -m integration
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import numpy as np

import alignimg as ai


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_ALIGNIMG_INTEGRATION") != "1",
        reason="set RUN_ALIGNIMG_INTEGRATION=1 to run real-data integration tests",
    ),
]


def test_realdata_subset_runs_through_public_api():
    mrcfile = pytest.importorskip("mrcfile")
    data_path = Path("data/local/test_align.mrcs")
    if not data_path.exists():
        pytest.skip(f"real-data stack not found: {data_path}")

    with mrcfile.mmap(data_path, permissive=True, mode="r") as mrc:
        data = np.asarray(mrc.data)
        if data.ndim == 2:
            X = data[None, :, :]
        else:
            X = data[: min(4, data.shape[0])]

    X = np.asarray(X, dtype=np.float32)
    initial_ref = np.mean(X, axis=0).astype(np.float32)
    result = ai.align_to_references(
        X,
        initial_ref,
        config=ai.AlignmentConfig(
            max_iterations=1,
            top_l=2,
            angle_samples=36,
            translation_range=2,
            halfset_diagnostics=True,
            center_references=False,
        ),
    )
    corrected = ai.transform_images(X, result.poses)

    assert result.references.shape == (1, *initial_ref.shape)
    assert np.asarray(result.reference_history).shape == (2, 1, *initial_ref.shape)
    assert result.responsibilities.shape == (len(X), 1)
    assert corrected.shape == X.shape
    assert np.all(np.isfinite(result.references))
    assert np.all(np.isfinite(corrected))
    assert np.allclose(result.responsibilities, 1.0)
