from __future__ import annotations

import json

import numpy as np
import pytest

from alignimg._fourier import prepare_stack
from alignimg._polar_hard import (
    infer_polar_hard_candidates_cpu,
    translation_center_to_pose,
)
from tools import polar_231_batch_validation as validation


@pytest.fixture
def candidates():
    images, references, centers, _, config = validation.batch_fixture(2)
    return infer_polar_hard_candidates_cpu(
        prepare_stack(images, config),
        prepare_stack(references, config),
        config,
        np.full((2, 3), 1 / 3, dtype=np.float32),
        0.08,
        None,
        translation_centers=centers,
    )


def test_angle_roundoff_can_change_canonical_shift_without_moving_sampling_center(
    candidates,
):
    expected = {name: values.copy() for name, values in candidates.items()}
    expected["angle_deg"][0, 0] = 45.0
    expected["_polar_center_y_px"][0, 0] = 0.0
    expected["_polar_center_x_px"][0, 0] = 7.0
    expected["shift_y_px"][0, 0], expected["shift_x_px"][0, 0] = (
        translation_center_to_pose(45.0, 0.0, 7.0)
    )
    actual = {name: values.copy() for name, values in expected.items()}
    actual["angle_deg"][0, 0] = 45.0002
    actual["shift_y_px"][0, 0], actual["shift_x_px"][0, 0] = translation_center_to_pose(
        float(actual["angle_deg"][0, 0]), 0.0, 7.0
    )
    assert abs(float(actual["shift_x_px"][0, 0] - expected["shift_x_px"][0, 0])) > 1e-5
    errors = validation.compare_batch_candidates(expected, actual)
    assert 1e-5 < errors["shift_x_px"] < 1e-3
    for name in (
        "_polar_center_y_px",
        "_polar_center_x_px",
        "_polar_raw_shift_y_px",
        "_polar_raw_shift_x_px",
    ):
        assert errors[name] == 0.0


@pytest.mark.parametrize(
    "field,delta",
    [
        ("shift_y_px", 1.1e-3),
        ("shift_x_px", 1.1e-3),
        ("angle_deg", 1.1e-3),
        ("_polar_center_y_px", 1.1e-5),
        ("_polar_center_x_px", 1.1e-5),
        ("_polar_raw_shift_y_px", 1.1e-5),
        ("_polar_raw_shift_x_px", 1.1e-5),
        ("score", 2.1e-5),
        ("_polar_objective_margin", 1.1e-3),
    ],
)
def test_candidate_field_contract_rejects_excess_error(candidates, field, delta):
    actual = {name: values.copy() for name, values in candidates.items()}
    actual[field][0, 0] += delta
    with pytest.raises(AssertionError, match=field):
        validation.compare_batch_candidates(candidates, actual)


def test_candidate_objective_gate_remains_stricter_than_score_when_temperature_is_low(
    candidates,
):
    actual = {name: values.copy() for name, values in candidates.items()}
    actual["score"][0, 0] += 1e-5
    with pytest.raises(AssertionError, match="candidate objective"):
        validation.compare_batch_candidates(candidates, actual, temperature=0.005)


def test_failed_batch_report_keeps_completed_cases_and_failure_stage(
    tmp_path, monkeypatch
):
    def fail_batches(engine, frozen, *, report):
        report["cases"] = {"fixed_False_batch_1": {"t2_same_batch_exact": True}}
        report["active_case"] = {"particle_batch": 7, "check": "cross_batch"}
        raise AssertionError("injected cross-batch failure")

    monkeypatch.setattr(validation, "load_frozen_backend", lambda: object())
    monkeypatch.setattr(validation.mirror, "runtime_identity", lambda *args: {})
    monkeypatch.setattr(validation.mirror, "environment_info", lambda *args: {})
    monkeypatch.setattr(validation, "validate_batches", fail_batches)
    output = tmp_path / "failed.json"
    assert validation.main(["--backend", "cuda", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert "injected cross-batch failure" in report["error"]
    batches = report["checks"]["resident_batches"]
    assert batches["cases"]["fixed_False_batch_1"]["t2_same_batch_exact"]
    assert batches["active_case"] == {"particle_batch": 7, "check": "cross_batch"}


def test_batch_runner_refuses_existing_report(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text('{"frozen": true}\n')
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        validation.main(["--backend", "cuda", "--output", str(output)])
    assert json.loads(output.read_text()) == {"frozen": True}


def test_frozen_solver_is_hash_checked(tmp_path):
    path = tmp_path / "backend.py"
    path.write_text("# not accepted T2 source\n")
    with pytest.raises(RuntimeError, match="frozen T2 backend source mismatch"):
        validation.load_frozen_backend(path)


def test_missing_frozen_solver_fails_without_gpu_access(tmp_path, monkeypatch):
    def missing():
        raise FileNotFoundError("missing frozen T2 source")

    monkeypatch.setattr(validation, "load_frozen_backend", missing)
    monkeypatch.setattr(
        validation.mirror,
        "runtime_identity",
        lambda *args: pytest.fail("GPU accessed before baseline check"),
    )
    output = tmp_path / "missing.json"
    assert validation.main(["--backend", "cuda", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert "missing frozen T2 source" in report["error"]
    assert report["checks"] == {}
