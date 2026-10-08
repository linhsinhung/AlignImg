from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

import alignimg as ai

from tools.fast_hard_validation import (
    INPUT_SCHEMA,
    _estimated_search_counts,
    _fit_common_gauge,
    baseline_config,
    fast_hard_config,
    freeze_inputs,
    git_revision,
    main,
    compare_polar_backend_parity,
    parse_args,
    polar_hard_config,
    synthetic_cases,
)


@pytest.mark.parametrize("fail_parity", [False, True])
def test_run_case_retains_every_execution_before_parity_gate(
    tmp_path, monkeypatch, fail_parity
):
    from tools import fast_hard_validation as validation

    case = synthetic_cases()["k1_mirror_off"]
    config = replace(ai.AlignmentConfig.preset("fast3"), batch_size=512)
    calls = []

    def execute(case, config, backend):
        index = len(calls)
        metadata = {
            "backend": backend,
            "class_average_batch_size": 512 - index,
            "class_average_gpu_memory_plan": {"batch_size": 512 - index},
            "class_average_gpu_memory_events": [],
            "gpu_workspace": {"budget_bytes": 1000 - index, "closed": True},
            "gpu_memory_plans": [
                {
                    "stage": "polar_hard_candidate_inference",
                    "batch_size": 90 - index,
                    "particle_storage_policy": "streaming",
                }
            ],
            "performance": {
                "gpu_memory": {"entry": {"device_used_bytes": index}},
                "stages": {
                    "workflow/alignment_engine/candidate_inference": {
                        "wall_seconds": 1.0
                    }
                },
            },
        }
        result = SimpleNamespace(
            metadata=metadata,
            index=index,
            responsibilities=np.ones((4, 1), np.float32),
            reference_assignments=np.zeros(4, np.int32),
            references=case["references"],
            diagnostics=[],
        )
        calls.append(result)
        return result

    monkeypatch.setattr(validation, "_execute", execute)
    monkeypatch.setattr(validation, "synchronize", lambda backend: None)
    monkeypatch.setattr(validation, "_check_result", lambda *args: None)
    monkeypatch.setattr(validation, "_check_hard_result", lambda *args: None)
    monkeypatch.setattr(validation, "_pose_quality", lambda *args: {})
    monkeypatch.setattr(
        validation, "result_hashes", lambda r: {"combined": str(r.index)}
    )
    monkeypatch.setattr(
        validation, "polar_result_arrays", lambda r: {"index": np.array([r.index])}
    )

    def compare(*args):
        if fail_parity:
            raise AssertionError("injected parity failure")
        return {}

    monkeypatch.setattr(validation, "compare_arrays", compare)
    records = []
    kwargs = dict(
        config=config,
        backend="cuda",
        measured_repeats=3,
        profile_execution=True,
        output=tmp_path / "run.json",
        execution_records=records,
    )
    if fail_parity:
        with pytest.raises(AssertionError, match="injected parity failure"):
            validation.run_case("fixture", case, **kwargs)
        assert len(records) == 4  # warm-up and all measured runs survive failure
    else:
        entry = validation.run_case("fixture", case, **kwargs)
        assert entry["execution_records"] is records
        assert [r["phase"] for r in records] == ["warmup"] + ["unprofiled"] * 3 + [
            "profiled"
        ]
        assert not entry["deterministic_exact_match"]
    for index, record in enumerate(records):
        assert record["gpu_memory_plans"][0]["batch_size"] == 90 - index
        assert record["gpu_workspace"]["budget_bytes"] == 1000 - index
        assert record["result_hashes"]["combined"] == str(index)
        assert record["class_average_batch_size"] == 512 - index
        assert record["class_average_gpu_memory_plan"]["batch_size"] == 512 - index
    calls[0].metadata["gpu_memory_plans"][0]["batch_size"] = 1
    assert records[0]["gpu_memory_plans"][0]["batch_size"] == 90


def test_stage0_baseline_config_freezes_balanced_side_of_future_ab():
    config = baseline_config(batch_size=512, memory_fraction=0.8)

    assert config.search_strategy == "proposal"
    assert config.max_iterations == 3
    assert config.top_l == 8
    assert config.proposal_angles_per_reference == 6
    assert config.translation_range == 4.0
    assert config.translation_step == 1.0
    assert config.robust_weighting is False
    assert config.halfset_diagnostics is False
    assert config.apply_final_pose_to_raw is False
    assert config.batch_size == 512
    assert config.memory_fraction == 0.8


def test_stage1_fast_hard_candidate_only_changes_proposals_and_top_l():
    baseline = baseline_config(batch_size=512, memory_fraction=0.8)
    candidate = fast_hard_config(batch_size=512, memory_fraction=0.8)

    differences = {
        name
        for name, value in vars(baseline).items()
        if getattr(candidate, name) != value
    }
    assert differences == {"top_l", "proposal_angles_per_reference"}
    assert candidate.search_strategy == "proposal"
    assert candidate.top_l == 1
    assert candidate.proposal_angles_per_reference == 4


def test_stage1_proposal_ablation_only_changes_proposal_count():
    proposal_4 = fast_hard_config(
        batch_size=512,
        memory_fraction=0.8,
        proposal_angles_per_reference=4,
    )
    proposal_2 = fast_hard_config(
        batch_size=512,
        memory_fraction=0.8,
        proposal_angles_per_reference=2,
    )

    differences = {
        name
        for name, value in vars(proposal_4).items()
        if getattr(proposal_2, name) != value
    }
    assert differences == {"proposal_angles_per_reference"}
    assert proposal_4.top_l == proposal_2.top_l == 1


def test_stage2_polar_hard_candidate_changes_only_the_search_contract():
    baseline = baseline_config(batch_size=16, memory_fraction=0.8)
    candidate = polar_hard_config(batch_size=16, memory_fraction=0.8)

    differences = {
        name
        for name, value in vars(baseline).items()
        if getattr(candidate, name) != value
    }
    assert differences == {
        "candidate_scoring",
        "proposal_angles_per_reference",
        "score_model",
        "search_strategy",
        "top_l",
    }
    assert candidate.search_strategy == "polar_hard"


def test_stage3_runner_accepts_explicit_gpu_polar_comparison(tmp_path):
    output = tmp_path / "stage3.json"

    args = parse_args(
        [
            "--suite",
            "synthetic",
            "--backend",
            "cuda",
            "--profile-execution",
            "--compare-polar-hard",
            "--output",
            str(output),
        ]
    )

    assert args.backend == "cuda"
    assert args.compare_polar_hard is True


def test_stage3_polar_backend_parity_uses_frozen_tolerances(tmp_path):
    reference = tmp_path / "reference.npz"
    actual = tmp_path / "actual.npz"
    common = {
        "assignments": np.asarray([0, 1], dtype=np.int32),
        "mirror": np.asarray([False, True]),
        "candidate_reference_index": np.asarray([[0], [1]], dtype=np.int32),
        "candidate_mirror": np.asarray([[False], [True]]),
        "angle_deg": np.asarray([10.0, -20.0], dtype=np.float32),
        "candidate_angle_deg": np.asarray([[10.0], [-20.0]], dtype=np.float32),
        "shift_y_px": np.asarray([1.0, -2.0], dtype=np.float32),
        "shift_x_px": np.asarray([0.5, 3.0], dtype=np.float32),
        "candidate_shift_y_px": np.asarray([[1.0], [-2.0]], dtype=np.float32),
        "candidate_shift_x_px": np.asarray([[0.5], [3.0]], dtype=np.float32),
        "candidate_score": np.asarray([[0.8], [0.7]], dtype=np.float32),
        "polar_raw_shift_y_px": np.asarray([1.0, -2.0], dtype=np.float32),
        "polar_raw_shift_x_px": np.asarray([0.5, 3.0], dtype=np.float32),
    }
    np.savez_compressed(reference, **common)
    changed = {name: value.copy() for name, value in common.items()}
    changed["candidate_score"] += np.float32(1e-6)
    changed["shift_x_px"] += np.float32(5e-5)
    changed["candidate_shift_x_px"] += np.float32(5e-5)
    np.savez_compressed(actual, **changed)
    case = {
        "images": np.zeros((2, 8, 8), dtype=np.float32),
        "references": np.zeros((2, 8, 8), dtype=np.float32),
        "priors": np.eye(2, dtype=np.float32),
    }

    parity = compare_polar_backend_parity(
        reference,
        actual,
        case,
        polar_hard_config(batch_size=512, memory_fraction=0.8),
    )

    assert parity["within_tolerance"] is True
    assert parity["errors"]["raw_shift_x_max_px"] == 0.0
    assert parity["errors"]["pose_shift_x_max_px"] > 1e-5
    assert parity["tolerances"] == {
        "angle_deg": 1e-3,
        "shift_px": 1e-5,
        "score_absolute": 2e-5,
        "objective_absolute": 1e-3,
    }


def test_synthetic_workloads_cover_required_k_snr_pose_and_mirror_cases():
    cases = synthetic_cases()

    assert set(cases) == {
        "k1_mirror_off",
        "k1_mirror_on",
        "k3_noiseless",
        "k3_snr_0_5",
        "k3_snr_0_2",
    }
    assert not bool(cases["k1_mirror_off"]["mirror_search"])
    assert bool(cases["k1_mirror_on"]["mirror_search"])
    assert np.array_equal(
        cases["k1_mirror_off"]["truth_angle_deg"],
        np.asarray([0.0, 90.0, -37.5, 143.0], dtype=np.float32),
    )
    assert np.any(cases["k1_mirror_off"]["truth_shift_y_px"] % 1.0)
    assert np.array_equal(np.unique(cases["k3_noiseless"]["component"]), np.arange(3))
    assert not np.array_equal(
        cases["k3_noiseless"]["images"], cases["k3_snr_0_5"]["images"]
    )


def test_frozen_inputs_are_reused_without_resampling(tmp_path):
    path = tmp_path / "synthetic.inputs.npz"

    first = freeze_inputs("synthetic", path)
    second = freeze_inputs("synthetic", path)

    assert first["sha256"] == second["sha256"]
    assert first["source_sha256"] == {}
    assert set(first["cases"]) == set(second["cases"])
    with np.load(path, allow_pickle=False) as saved:
        assert str(saved["schema"].item()) == INPUT_SCHEMA
        assert json.loads(str(saved["case_names_json"].item())) == list(first["cases"])


def test_frozen_inputs_reject_a_different_suite(tmp_path):
    path = tmp_path / "inputs.npz"
    freeze_inputs("synthetic", path)

    with pytest.raises(ValueError, match="does not match"):
        freeze_inputs("homogeneous", path)


def test_git_revision_is_optional_for_code_only_handoffs(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git is unavailable")

    monkeypatch.setattr("tools.fast_hard_validation.subprocess.run", unavailable)

    revision = git_revision()

    assert revision["available"] is False
    assert revision["commit"] is None
    assert revision["short_commit"] is None
    assert revision["dirty"] is None
    assert revision["error"] == "git is unavailable"


def test_search_count_distinguishes_uniform_and_fixed_reference_priors():
    cases = synthetic_cases()
    uniform = cases["k3_noiseless"]
    fixed = dict(uniform)
    component = uniform["component"]
    fixed["priors"] = np.eye(3, dtype=np.float32)[component]
    config = replace(
        baseline_config(batch_size=16, memory_fraction=0.8),
        max_iterations=1,
        top_l=1,
        proposal_angles_per_reference=4,
    )

    uniform_counts = _estimated_search_counts(uniform, config)
    fixed_counts = _estimated_search_counts(fixed, config)

    assert (
        uniform_counts["polar_proposal_count"]
        == 3 * fixed_counts["polar_proposal_count"]
    )
    assert (
        uniform_counts["exact_rescoring_count"]
        == 3 * fixed_counts["exact_rescoring_count"]
    )
    assert (
        uniform_counts["retained_candidate_count"]
        == fixed_counts["retained_candidate_count"]
    )


def test_common_pose_gauge_supports_mirrored_pose_sets():
    expected = ai.PoseSet(
        np.asarray([-40.0, 15.0, 95.0], dtype=np.float32),
        np.asarray([-2.0, 0.5, 1.0], dtype=np.float32),
        np.asarray([1.0, -1.5, 2.0], dtype=np.float32),
        np.ones(3, dtype=np.bool_),
    )
    gauge_angle = 23.0
    radians = np.deg2rad(gauge_angle)
    rotation = np.asarray(
        [[np.cos(radians), np.sin(radians)], [-np.sin(radians), np.cos(radians)]]
    )
    shift = np.column_stack((expected.shift_x_px, expected.shift_y_px))
    transformed_shift = shift @ rotation.T + np.asarray([1.25, -0.75])
    predicted = ai.PoseSet(
        expected.angle_deg + gauge_angle,
        transformed_shift[:, 1],
        transformed_shift[:, 0],
        expected.mirror,
    )

    gauge, gauged = _fit_common_gauge(predicted, expected)

    assert float(gauge.angle_deg[0]) == pytest.approx(gauge_angle, abs=1e-5)
    assert float(gauge.shift_y_px[0]) == pytest.approx(-0.75, abs=1e-5)
    assert float(gauge.shift_x_px[0]) == pytest.approx(1.25, abs=1e-5)
    assert np.allclose(gauged.angle_deg, predicted.angle_deg, atol=1e-5)
    assert np.allclose(gauged.shift_y_px, predicted.shift_y_px, atol=1e-5)
    assert np.allclose(gauged.shift_x_px, predicted.shift_x_px, atol=1e-5)
    assert np.array_equal(gauged.mirror, predicted.mirror)


def test_stage1_runner_records_same_process_baseline_and_hard_candidate(tmp_path):
    inputs = tmp_path / "synthetic.inputs.npz"
    output = tmp_path / "stage1.json"

    exit_code = main(
        [
            "--suite",
            "synthetic",
            "--backend",
            "cpu",
            "--only",
            "k1_mirror_off",
            "--batch-size",
            "8",
            "--deterministic-repeats",
            "2",
            "--profile-execution",
            "--compare-fast-hard",
            "--inputs",
            str(inputs),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text())
    case = report["cases"]["k1_mirror_off"]
    assert exit_code == 0
    assert report["schema"] == "alignimg.fast-hard-validation.v2"
    assert report["stage"] == 1
    assert report["status"] == "completed"
    assert case["baseline"]["config"]["top_l"] == 8
    assert case["fast_hard"]["config"]["top_l"] == 1
    assert case["fast_hard"]["mean_max_responsibility"] == 1.0
    assert len(case["baseline"]["profiled_candidate_inference_seconds"]) == 3
    assert len(case["fast_hard"]["profiled_candidate_inference_seconds"]) == 3
    assert case["comparison"]["candidate_inference_speedup"] > 0.0


def test_stage1_runner_records_proposal_2_vs_4_ablation(tmp_path):
    inputs = tmp_path / "synthetic.inputs.npz"
    output = tmp_path / "ablation.json"

    exit_code = main(
        [
            "--suite",
            "synthetic",
            "--backend",
            "cpu",
            "--only",
            "k1_mirror_off",
            "--batch-size",
            "8",
            "--deterministic-repeats",
            "2",
            "--profile-execution",
            "--compare-fast-hard",
            "--proposal-ablation-2-vs-4",
            "--inputs",
            str(inputs),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text())
    case = report["cases"]["k1_mirror_off"]
    assert exit_code == 0
    assert report["status"] == "completed"
    assert report["parameters"]["proposal_ablation_2_vs_4"] is True
    assert case["baseline"]["variant"] == "config_only_fast_hard_proposal_4"
    assert case["baseline"]["config"]["proposal_angles_per_reference"] == 4
    assert case["fast_hard"]["variant"] == "config_only_fast_hard_proposal_2"
    assert case["fast_hard"]["config"]["proposal_angles_per_reference"] == 2
    differences = {
        name
        for name, value in case["baseline"]["config"].items()
        if case["fast_hard"]["config"][name] != value
    }
    assert differences == {"proposal_angles_per_reference"}
