from __future__ import annotations

import numpy as np

from tools.quadratic_feedback_ab_validation import (
    make_relion_subsets,
    scientific_gate,
)


def test_relion_subsets_are_mapped_confident_and_large_enough():
    subsets, report = make_relion_subsets(
        relion_class=np.array([1, 1, 2, 2, 2, 1]),
        confidence=np.array([0.9, 0.8, 0.7, 0.9, 0.2, 0.6]),
        assignments=np.array([0, 0, 1, 1, 0, 0]),
        component_count=2,
        minimum_confidence=0.5,
        minimum_component_particles=2,
    )

    assert [(component, selected.tolist()) for component, selected in subsets] == [
        (0, [0, 1, 5]),
        (1, [2, 3]),
    ]
    assert report["optimal_alignimg_to_relion_class_mapping"] == {0: 1, 1: 2}
    assert report["included_particle_count"] == 5


def accuracy(
    angle,
    shift,
    p95_angle,
    p95_shift,
    pose_correlations,
    mstep_correlations=None,
):
    if mstep_correlations is None:
        mstep_correlations = pose_correlations
    return {
        "angle_error_deg": {"median": angle, "p95": p95_angle},
        "shift_error_px": {"median": shift, "p95": p95_shift},
        "per_component": [
            {
                "component": index,
                "angle_error_deg": {"median": angle + index},
                "shift_error_px": {"median": shift + index},
                "pose_applied_average_correlation": pose_correlation,
                "mstep_reference_correlation": mstep_correlation,
            }
            for index, (pose_correlation, mstep_correlation) in enumerate(
                zip(pose_correlations, mstep_correlations, strict=True)
            )
        ],
    }


def test_scientific_gate_uses_resolution_aware_angle_and_strict_shift():
    result = scientific_gate(
        accuracy(8.0, 4.0, 20.0, 8.0, [0.8, 0.8]),
        accuracy(3.0, 2.0, 10.0, 4.0, [0.90, 0.91]),
        accuracy(4.0, 1.5, 10.5, 4.2, [0.899, 0.906]),
    )

    assert result["passed"]
    assert not result["observations_not_used_as_gates"][
        "quadratic_angle_strictly_better_than_adaptive"
    ]


def test_scientific_gate_rejects_pose_applied_average_regression():
    result = scientific_gate(
        accuracy(8.0, 4.0, 20.0, 8.0, [0.8]),
        accuracy(3.0, 2.0, 10.0, 4.0, [0.90]),
        accuracy(4.0, 1.5, 10.5, 4.2, [0.894]),
    )

    assert not result["passed"]
    assert not result["checks"]["each_pose_applied_average_correlation_within_0_005"]


def test_scientific_gate_records_mstep_reference_regression_as_observation():
    result = scientific_gate(
        accuracy(8.0, 4.0, 20.0, 8.0, [0.8], [0.8]),
        accuracy(3.0, 2.0, 10.0, 4.0, [0.90], [0.90]),
        accuracy(4.0, 1.5, 10.5, 4.2, [0.899], [0.80]),
    )

    assert result["passed"]
    assert np.isclose(
        result["observations_not_used_as_gates"][
            "mstep_reference_correlation_delta_by_component"
        ][0],
        -0.1,
    )
