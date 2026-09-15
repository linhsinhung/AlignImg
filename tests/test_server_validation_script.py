"""Smoke-test the portable server validation workflow without requiring CUDA."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import alignimg as ai
from tools.server_validation import (
    _top_l_posterior_comparison,
    adaptive_refine_case,
    quadratic_refine_case,
    adaptive_rescue_case,
    classification_calibration,
    fourier_native_conformance_case,
    fourier_mstep_workflow_ab_case,
    fourier_workflow_ab_case,
    integer_center_contract_case,
    native_fused_fourier_accumulation_case,
    native_indexed_fourier_case,
    suite_cases,
    whitened_scoring_ab_case,
)


def candidate_set(order: list[int]) -> ai.CandidateSet:
    angles = np.asarray([[0.0, 5.0, 10.0]], dtype=np.float32)[:, order]
    posterior = np.asarray([[0.6, 0.3, 0.1]], dtype=np.float32)[:, order]
    return ai.CandidateSet(
        reference_index=np.zeros((1, 3), dtype=np.int32),
        angle_deg=angles,
        shift_y_px=np.zeros((1, 3), dtype=np.float32),
        shift_x_px=np.zeros((1, 3), dtype=np.float32),
        mirror=np.zeros((1, 3), dtype=np.bool_),
        score=posterior.copy(),
        posterior=posterior,
    )


def test_top_l_posterior_comparison_is_invariant_to_candidate_order():
    metrics = _top_l_posterior_comparison(
        candidate_set([0, 1, 2]), candidate_set([2, 0, 1])
    )

    assert metrics["minimum_matched_candidate_fraction"] == 1.0
    assert metrics["mean_posterior_total_variation"] == 0.0
    assert metrics["matched_score_maximum_absolute_error"] == 0.0


def test_cpu_only_validation_writes_parseable_incremental_report(tmp_path: Path):
    output = tmp_path / "validation.json"
    completed = subprocess.run(
        [
            sys.executable,
            "tools/server_validation.py",
            "--suite",
            "quick",
            "--backend",
            "cpu",
            "--only",
            "cpu_gpu_parity",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["environment"]["alignimg_module_version"] == ai.__version__
    assert report["cases"]["cpu_gpu_parity"]["status"] == "passed"
    assert report["summary"] == {
        "failed": 0,
        "passed": 1,
        "total_recorded": 1,
    }


def test_calibration_metrics_distinguish_perfect_and_wrong_predictions():
    truth = np.array([0, 1], dtype=np.int32)
    perfect = classification_calibration(np.eye(2, dtype=np.float32), truth)
    wrong = classification_calibration(
        np.array([[0.01, 0.99], [0.99, 0.01]], dtype=np.float32), truth
    )
    assert perfect["class_nll"] == 0.0
    assert perfect["class_brier_score"] == 0.0
    assert perfect["class_ece_10"] == 0.0
    assert wrong["class_nll"] > 4.0
    assert wrong["class_brier_score"] > 1.9
    assert wrong["class_ece_10"] > 0.9


def test_cpu_integer_center_server_contract():
    result = integer_center_contract_case("cpu")
    assert result["parameters"]["center"] == 16
    assert result["cpu_peaks"] == result["expected_peaks"]
    assert result["shifted_dft_origin"] == (16, 16)


def test_cpu_fourier_native_conformance_contract():
    result = fourier_native_conformance_case("cpu")
    assert result["minimum_raster_band_ncc"] >= 0.995
    assert result["backend_relative_l2_error"] == 0.0


def test_native_indexed_fourier_case_is_explicitly_cuda_only():
    result = native_indexed_fourier_case("cpu")
    assert result["skipped"]
    assert "CUDA-specific" in result["reason"]


def test_native_fused_fourier_accumulation_is_explicitly_cuda_only():
    result = native_fused_fourier_accumulation_case("cpu")
    assert result["skipped"]
    assert "CUDA-specific" in result["reason"]


def test_cpu_fourier_workflow_ab_contract():
    result = fourier_workflow_ab_case("cpu", batch_size=8)

    assert result["global"]["assignment_disagreement_fraction"] == 0.0
    assert result["adaptive_refine"]["assignment_disagreement_fraction"] == 0.0
    assert result["global"]["minimum_reference_correlation"] > 0.99
    assert result["adaptive_refine"]["minimum_reference_correlation"] > 0.99
    assert result["global"]["fourier_metadata"]["candidate_scoring"] == "fourier"
    assert (
        result["global"]["top_l_posterior"][
            "minimum_matched_candidate_fraction"
        ]
        == 1.0
    )
    assert (
        result["adaptive_refine"]["top_l_posterior"][
            "minimum_matched_candidate_fraction"
        ]
        == 1.0
    )
    assert (
        result["global"]["top_l_posterior"]["mean_posterior_total_variation"]
        <= 0.15
    )


def test_cpu_fourier_mstep_workflow_ab_contract():
    result = fourier_mstep_workflow_ab_case("cpu", batch_size=8)

    for workflow in ("global", "adaptive_refine"):
        metrics = result[workflow]
        assert metrics["assignment_disagreement_fraction"] == 0.0
        assert metrics["responsibility_mae"] == 0.0
        assert metrics["minimum_reference_correlation"] > 0.99
        assert metrics["minimum_cpu_backend_reference_correlation"] == 1.0
        assert metrics["frc_is_finite"]
        assert metrics["fourier_metadata"]["reference_update"] == "fourier"


def test_cpu_whitened_scoring_ab_contract():
    result = whitened_scoring_ab_case("cpu", batch_size=8)

    assert result["uniform"]["correct_within_5_deg"] <= 20
    assert result["whitened"]["correct_within_5_deg"] == 24
    assert result["score_weights"]["finite"]
    assert result["cpu_backend"]["maximum_angle_delta_deg"] == 0.0
    assert (
        result["whitened"]["metadata"]["score_model"]
        == "whitened_fourier_ncc"
    )


def test_cpu_adaptive_refine_server_contract():
    result = adaptive_refine_case("cpu", 4)
    assert (
        result["cpu"]["final_median_angle_error_deg"]
        < result["cpu"]["initial_median_angle_error_deg"]
    )
    assert result["cpu"]["mean_map_posterior"] > 0.0


def test_cpu_quadratic_refine_server_contract():
    result = quadratic_refine_case("cpu", 8)
    medians = result["median_errors"]
    assert medians["quadratic_angle_error_deg"] <= 0.5
    assert medians["quadratic_shift_error_px"] <= 0.2
    assert result["quadratic_metadata"]["search_strategy"] == "quadratic_refine"
    for scenario in result["noise_scenarios"].values():
        assert scenario["quadratic_median_angle_error_deg"] < scenario[
            "adaptive_median_angle_error_deg"
        ]
        assert scenario["quadratic_median_shift_error_px"] < scenario[
            "adaptive_median_shift_error_px"
        ]


def test_cpu_adaptive_rescue_server_contract():
    result = adaptive_rescue_case("cpu", 4)
    assert result["final_bad_particle_angle_error_deg"] <= 5.0
    assert result["diagnostics"][0]["boundary_hit_count"] == 0
    assert result["diagnostics"][1]["rescue_particle_count"] == 1


def test_tuning_suite_contains_bounded_ablation_cases():
    names = [name for name, _ in suite_cases("tuning", "cpu", 256)]
    assert names[-3:] == [
        "low_snr_ablation",
        "reference_free_ablation",
        "batch_sweep_k50",
    ]
