from __future__ import annotations

import pytest

from tools.fast_hard_iteration_sweep import ITERATIONS, parse_args, summarize_sweep


def _run(correlation: float, seconds: float) -> dict:
    return {
        "unprofiled_median_seconds": seconds,
        "unprofiled_stage_analysis": {
            "stages": [
                {"class_average_correlation_to_supplied": correlation},
            ]
        },
    }


def test_fast_only_sweep_cli_freezes_requested_iteration_points(tmp_path):
    args = parse_args(["--output", str(tmp_path / "report.json")])

    assert ITERATIONS == (3, 5, 8, 10, 12)
    assert args.deterministic_repeats == 3
    assert args.batch_size == 512


def test_sweep_summary_finds_first_practical_and_release_floor_pass():
    correlations = (0.94, 0.948, 0.9495, 0.951, 0.952)
    runs = {
        str(iterations): _run(correlation, float(iterations))
        for iterations, correlation in zip(ITERATIONS, correlations)
    }

    summary = summarize_sweep(runs)

    assert summary["minimum_iterations_meeting_accepted_2_2_floor"] == 8
    assert summary["minimum_iterations_meeting_practical_0_95"] == 10
    assert summary["points"][1][
        "raw_correlation_delta_from_previous_point"
    ] == pytest.approx(0.008)
