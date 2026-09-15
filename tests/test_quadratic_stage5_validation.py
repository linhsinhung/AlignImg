from __future__ import annotations

import pytest

from tools.quadratic_stage5_validation import acceptance_gate, refine_configs


def summary(
    *,
    seconds=10.0,
    soft_correlation=0.9,
    raw_correlation=0.9,
    soft_raw_correlation=0.95,
    backend="cuda",
    peak_backend="native_cuda",
):
    return {
        "seconds": seconds,
        "soft_reference_correlation_to_supplied": soft_correlation,
        "raw_class_average_correlation_to_supplied": raw_correlation,
        "soft_reference_correlation_to_raw_average": soft_raw_correlation,
        "class_average_estimator": "final_map_pose_inlier_weighted_raw",
        "responsibility_sum_max_error": 1e-7,
        "metadata": {
            "backend": backend,
            "gpu_fallback_reason": None,
            "quadratic_peak_backend": peak_backend,
        },
    }


def test_stage5_configs_only_differ_in_pose_search_parameters():
    configs = refine_configs(
        iterations=2,
        batch_size=512,
        memory_fraction=0.8,
        profile_execution=False,
    )
    adaptive = configs["adaptive_posterior"]
    quadratic = configs["quadratic_refine"]

    assert adaptive.search_strategy == "adaptive_posterior"
    assert adaptive.coarse_angle_step == 6.0
    assert adaptive.local_angle_range == 15.0
    assert quadratic.search_strategy == "quadratic_refine"
    assert quadratic.coarse_angle_step == 1.0
    assert quadratic.local_angle_range == 7.0
    assert quadratic.local_shift_range == adaptive.local_shift_range == 3.0
    assert quadratic.max_iterations == adaptive.max_iterations == 2
    assert quadratic.apply_final_pose_to_raw
    assert adaptive.apply_final_pose_to_raw


def test_stage5_gate_accepts_faster_equivalent_quadratic_result():
    result = acceptance_gate(
        summary(seconds=10.0),
        summary(seconds=4.0, soft_correlation=0.897, raw_correlation=0.896),
        backend="cuda",
    )

    assert result["passed"]
    assert result["speedup_adaptive_over_quadratic"] == 2.5


def test_stage5_gate_does_not_reward_soft_reference_blur():
    result = acceptance_gate(
        summary(seconds=10.0, soft_correlation=0.98, soft_raw_correlation=0.96),
        summary(
            seconds=4.0,
            soft_correlation=0.95,
            raw_correlation=0.896,
            soft_raw_correlation=0.98,
        ),
        backend="cuda",
    )

    assert result["passed"]
    assert result["observations"]["soft_reference_correlation_delta"] == pytest.approx(
        -0.03
    )


def test_stage5_gate_rejects_slow_or_degraded_result():
    result = acceptance_gate(
        summary(seconds=10.0),
        summary(
            seconds=11.0,
            soft_correlation=0.89,
            raw_correlation=0.89,
            soft_raw_correlation=0.89,
        ),
        backend="cuda",
    )

    assert not result["passed"]
    assert not result["checks"]["quadratic_not_more_than_5_percent_slower"]
    assert not result["checks"][
        "soft_reference_coherence_with_raw_average_within_0_005"
    ]
    assert not result["checks"]["raw_average_correlation_within_0_005"]
