from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mrcfile
import numpy as np

import alignimg as ai
from alignimg._geometry import CENTER_CONVENTION
from tools.re2dc_70s_feedback_validation import (
    jsonable,
    main,
    parse_args,
    stable_frc_metrics,
)


def test_jsonable_sanitizes_nested_nonfinite_values():
    value = {"array": np.array([1.0, np.nan, np.inf]), "scalar": -math.inf}
    assert jsonable(value) == {"array": [1.0, None, None], "scalar": None}


def test_feedback_validation_defaults_to_soft_corrective_prior(monkeypatch):
    monkeypatch.setattr("sys.argv", ["re2dc_70s_feedback_validation.py"])
    assert parse_args().corrective_prior_source == "responsibilities"
    assert parse_args().search_strategy == "adaptive_posterior"
    assert parse_args().candidate_scoring == "fourier"
    assert parse_args().reference_update == "fourier"


def test_stable_frc_metrics_exclude_low_weight_components():
    metrics = stable_frc_metrics(
        {
            "halfset_effective_weight": np.array(
                [[9.0, 12.0], [11.0, 12.0]], dtype=np.float32
            ),
            "frc_0143_stable_cutoff_cyc_per_px": np.array(
                [0.2, 0.25], dtype=np.float32
            ),
        },
        minimum_halfset_weight=10.0,
        pixel_size=2.0,
    )
    assert metrics["reliable_component_mask"].tolist() == [False, True]
    assert metrics["reliable_median_stable_cutoff_cyc_per_px"] == 0.25
    assert metrics["reliable_median_resolution_angstrom"] == 8.0


def test_feedback_validation_writes_fixed_and_corrective_artifacts(tmp_path: Path):
    size = 16
    y, x = np.indices((size, size), dtype=np.float32)
    first = np.exp(-((y - 5) ** 2 + (x - 10) ** 2) / 4.0)
    second = np.exp(-((y - 11) ** 2 + (x - 5) ** 2) / 3.0)
    references = np.stack((first, second)).astype(np.float32)
    truth = np.array([0, 0, 1, 1], dtype=np.int32)
    images = references[truth]
    feedback = np.array([1, 0, 1, 1], dtype=np.int32)
    responsibilities = np.eye(2, dtype=np.float32)[feedback]

    stack_path = tmp_path / "particles.mrcs"
    reference_path = tmp_path / "references.mrcs"
    rf_result_path = tmp_path / "rf-result.npz"
    output_path = tmp_path / "feedback.json"
    with mrcfile.new(stack_path) as mrc:
        mrc.set_data(images)
        mrc.voxel_size = 2.0
    with mrcfile.new(reference_path) as mrc:
        mrc.set_data(references)
        mrc.voxel_size = 2.0
    np.savez_compressed(
        rf_result_path,
        alignimg_version=np.asarray(ai.__version__),
        center_convention=np.asarray(CENTER_CONVENTION),
        assignments=feedback,
        responsibilities=responsibilities,
        angle_deg=np.zeros(4, dtype=np.float32),
        shift_y_px=np.zeros(4, dtype=np.float32),
        shift_x_px=np.zeros(4, dtype=np.float32),
        mirror=np.zeros(4, dtype=bool),
    )
    args = argparse.Namespace(
        stack=stack_path,
        references=reference_path,
        rf_result=rf_result_path,
        output=output_path,
        backend="cpu",
        iterations=2,
        anneal_iterations=1,
        candidate_scoring="fourier",
        score_model="fourier_ncc",
        reference_update="fourier",
        corrective_trust=0.5,
        corrective_prior_source="assignments",
        minimum_frc_halfset_weight=1.0,
        batch_size=8,
        memory_fraction=0.8,
        deterministic_repeats=2,
    )

    main(args)

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["parameters"]["config"]["candidate_scoring"] == "fourier"
    assert report["parameters"]["config"]["score_model"] == "fourier_ncc"
    assert report["parameters"]["config"]["reference_update"] == "fourier"
    assert report["runs"]["fixed"]["reassignment_fraction"] == 0.0
    assert set(report["runs"]) == {"fixed", "corrective"}
    for mode in ("fixed", "corrective"):
        assert Path(report["runs"][mode]["artifacts"]["references"]).exists()
        assert Path(report["runs"][mode]["artifacts"]["result"]).exists()
        assert report["runs"][mode]["responsibility_sum_max_error"] < 1e-6
        assert report["runs"][mode]["reproducible"]
        with np.load(report["runs"][mode]["artifacts"]["result"]) as saved:
            assert str(saved["center_convention"].item()) == CENTER_CONVENTION
