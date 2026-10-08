from copy import deepcopy
import inspect
import json

import numpy as np
import pytest

from alignimg import _engine
from tools import polar_center_ab_validation as validation


def center_values():
    return {
        "reference_index": np.array([[0], [1]], np.int32),
        "angle_deg": np.array([[31.2], [-93.592102]], np.float32),
        "shift_y_px": np.array([[0.25], [0.031188607]], np.float32),
        "shift_x_px": np.array([[-0.5], [-0.001957929]], np.float32),
        "_polar_center_y_px": np.array([[1 / 64], [0]], np.float32),
        "_polar_center_x_px": np.array([[-3 / 64], [-1 / 32]], np.float32),
    }


def test_frozen_helper_ast_identity():
    assert validation.legacy_helper_sha256() == validation.LEGACY_HELPER_SHA256


def test_helper_hash_matches_python312_ast_format(monkeypatch):
    original_dump = validation.ast.dump
    options = (
        {"show_empty": True}
        if "show_empty" in inspect.signature(original_dump).parameters
        else {}
    )

    def python312_dump(node):
        return original_dump(node, **options)

    monkeypatch.setattr(validation.ast, "dump", python312_dump)
    assert validation.legacy_helper_sha256() == validation.LEGACY_HELPER_SHA256


@pytest.mark.parametrize("variant", ["current", "legacy"])
def test_scoped_helper_restores_after_failure_and_limits_warmup_trace(variant):
    original = _engine._apply_reference_center_shifts
    values = center_values()
    traces = []
    with pytest.raises(RuntimeError, match="injected"):
        with validation.center_controller(variant, traces, 1):
            _engine._apply_reference_center_shifts(values, [(0.25, -0.125), (0, 0)])
            _engine._apply_reference_center_shifts(values, [(0, 0), (0, 0)])
            raise RuntimeError("injected")
    assert _engine._apply_reference_center_shifts is original
    assert len(traces) == 1 and traces[0]["phase"] == "warmup"
    assert traces[0]["zero_shift_particle_count"] == 1
    if variant == "current":
        assert traces[0]["zero_shift_center_max_delta_px"] == 0
    else:
        assert traces[0]["zero_shift_center_max_delta_px"] > 0


def test_nonzero_center_correction_matches_legacy():
    old, new = center_values(), center_values()
    shifts = [(0.25, -0.125), (-1, 2)]
    validation.legacy_center_update(old, shifts)
    _engine._apply_reference_center_shifts(new, shifts)
    for name in old:
        np.testing.assert_array_equal(old[name], new[name])


def test_comparison_retains_original_gate_and_reports_all_fields():
    expected = {
        "angle_deg": np.array([179.9998, 0], np.float32),
        "shift_x_px": np.zeros(2, np.float32),
        "assignments": np.zeros(2, np.int32),
        "references": np.arange(4, dtype=np.float32).reshape(1, 2, 2),
    }
    actual = deepcopy(expected)
    actual["angle_deg"][0] = -179.9998
    actual["new_diagnostic"] = np.ones(1)
    assert validation.compare_results(expected, actual)["within_frozen_tolerance"]
    actual["angle_deg"][1] = 1.4
    actual["shift_x_px"][1] = 1
    actual["references"][0, 0, 0] += 0.01
    compared = validation.compare_results(expected, actual)
    assert not compared["within_frozen_tolerance"] and "angle_deg" in compared["error"]
    assert not compared["fields"]["shift_x_px"]["within_frozen_tolerance"]
    assert not compared["fields"]["references"]["within_frozen_tolerance"]
    assert compared["fields"]["assignments"]["exact"]
    assert set(compared["fields"]) == set(expected)


@pytest.mark.parametrize(
    "current,legacy,expected",
    [
        (False, True, "legacy_roundtrip_recovers_frozen_baseline"),
        (False, False, "inconclusive_neither_matches_baseline"),
        (True, True, "mismatch_not_reproduced"),
        (True, False, "current_only_matches_baseline"),
    ],
)
def test_conclusion_does_not_waive_gate(current, legacy, expected):
    assert validation.conclusion(current, legacy) == expected


@pytest.mark.parametrize("artifact", ["report", "current", "legacy"])
def test_existing_report_or_result_is_not_overwritten(tmp_path, artifact):
    output = tmp_path / "test.json"
    existing = (
        output if artifact == "report" else validation.result_path(output, artifact)
    )
    existing.write_bytes(b"frozen")
    with pytest.raises(FileExistsError):
        validation.main(["--output", str(output)])
    assert existing.read_bytes() == b"frozen"


def test_missing_gpu_is_explicit_failed_report(tmp_path, monkeypatch):
    def missing(*args):
        raise RuntimeError("native GPU unavailable")

    monkeypatch.setattr(validation.integration, "runtime_info", missing)
    output = tmp_path / "failed.json"
    original = _engine._apply_reference_center_shifts
    assert validation.main(["--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["diagnostic_only"] and report["status"] == "failed"
    assert "GPU unavailable" in report["error"] and report["controller_restored"]
    assert _engine._apply_reference_center_shifts is original


def test_completed_diagnostic_with_baseline_mismatch_is_not_t6_pass(
    tmp_path, monkeypatch
):
    def experiment(engine, output, report):
        report["variants"] = {
            "current": {"vs_frozen_baseline": {"within_frozen_tolerance": False}},
            "legacy": {"vs_frozen_baseline": {"within_frozen_tolerance": True}},
        }
        report["conclusion"] = validation.conclusion(False, True)

    monkeypatch.setattr(validation, "run_experiment", experiment)
    output = tmp_path / "completed.json"
    assert validation.main(["--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["status"] == "completed" and report["diagnostic_only"]
    assert report["controller_restored"]
    assert report["conclusion"] == "legacy_roundtrip_recovers_frozen_baseline"
    assert not report["variants"]["current"]["vs_frozen_baseline"][
        "within_frozen_tolerance"
    ]
