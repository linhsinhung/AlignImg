"""Historical pre-instrumentation fixtures; requires private frozen artifacts."""

import json

import numpy as np
import pytest

from tools.performance_fixtures import DEFAULT_BASELINE, sha256
from tools.performance_validation import frozen_fixture_check, main


def test_frozen_preinstrumentation_fixtures():
    if not DEFAULT_BASELINE.exists():
        pytest.skip(
            "frozen local baseline artifacts are not included in the distribution"
        )
    report = frozen_fixture_check(DEFAULT_BASELINE)
    assert set(report) == {"global", "adaptive"}
    # frozen_fixture_check already enforces the field-specific tolerances in
    # compare_arrays.  Different NumPy/CPU builds need not be bitwise equal.
    assert all(
        np.isfinite(value["maximum_absolute_error"])
        and np.isfinite(value["relative_l2_error"])
        for case in report.values()
        for value in case.values()
    )
    manifest = json.loads((DEFAULT_BASELINE / "manifest.json").read_text())
    assert sha256(DEFAULT_BASELINE / "source.tar.gz") == manifest["archive_sha256"]



def test_fixture_check_rejects_missing_outputs(monkeypatch):
    if not DEFAULT_BASELINE.exists():
        pytest.skip("requires frozen local fixtures")
    from tools import performance_validation

    monkeypatch.setattr(performance_validation, "capture_iteration", lambda *args: {})
    with pytest.raises(AssertionError, match="fields differ"):
        frozen_fixture_check(DEFAULT_BASELINE)



def test_performance_runner_separates_profile_from_repeated_timing(tmp_path):
    if not DEFAULT_BASELINE.exists():
        pytest.skip("requires frozen local fixtures")
    output = tmp_path / "run.json"
    assert (
        main(
            [
                "--backend",
                "cpu",
                "--only",
                "global",
                "--repeats",
                "2",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    case = report["cases"]["global"]
    assert len(case["unprofiled_wall_seconds"]) == 2
    assert not case["config"]["profile_execution"]
    assert case["performance"]["profiled"]
    assert report["summary"]["performance_acceptance"] == "baseline_only"
    assert (tmp_path / "run.global.inputs.npz").exists()
