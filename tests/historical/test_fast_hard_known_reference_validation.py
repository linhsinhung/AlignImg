from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import alignimg as ai
from tools.fast_hard_known_reference_validation import (
    ACCEPTED_RAW_CORRELATION,
    RAW_CORRELATION_TOLERANCE,
    parse_args,
    stage_accuracy_analysis,
)


def _result(angle: float, shift: float, scale: float) -> SimpleNamespace:
    image = np.zeros((8, 8), dtype=np.float32)
    image[2, 5] = scale
    image[5, 3] = 0.5 * scale
    return SimpleNamespace(
        poses=ai.PoseSet(
            np.asarray([angle], dtype=np.float32),
            np.asarray([shift], dtype=np.float32),
            np.asarray([-shift], dtype=np.float32),
            np.asarray([False]),
        ),
        references=image[None],
        class_averages=image[None],
        reference_assignments=np.asarray([0], dtype=np.int32),
        diagnostics=[
            {
                "mean_expected_fourier_ncc": 0.5 + scale,
                "reference_relative_change": 0.1,
            }
        ],
        metadata={"class_average_estimator": "test"},
    )


def test_known_reference_cli_freezes_10plus2_and_accuracy_gate(tmp_path):
    args = parse_args(["--output", str(tmp_path / "report.json")])

    assert args.global_iterations == 10
    assert args.precise_iterations == 2
    assert args.profile_execution is False
    assert args.deterministic_repeats == 3
    assert ACCEPTED_RAW_CORRELATION - RAW_CORRELATION_TOLERANCE == pytest.approx(
        0.9493693463625517
    )


def test_stage_accuracy_analysis_reports_precise_change_without_truth():
    supplied = _result(0.0, 0.0, 1.0).references[0]
    accepted_result = _result(1.0, 0.25, 1.0)
    accepted = {
        "angle_deg": accepted_result.poses.angle_deg,
        "shift_y_px": accepted_result.poses.shift_y_px,
        "shift_x_px": accepted_result.poses.shift_x_px,
        "mirror": accepted_result.poses.mirror,
        "soft_references": accepted_result.references,
        "class_averages": accepted_result.class_averages,
    }

    analysis = stage_accuracy_analysis(
        [_result(2.0, 0.5, 0.8), _result(1.5, 0.3, 0.9)],
        supplied,
        accepted,
    )

    assert len(analysis["stages"]) == 2
    assert analysis["stages"][1]["all_assignments_are_zero"] is True
    assert analysis["precise_change_from_fast"]["changed_assignment_count"] == 0
    assert analysis["precise_change_from_fast"]["pose_delta"]["angle_absolute_deg"][
        "median"
    ] == pytest.approx(0.5)
