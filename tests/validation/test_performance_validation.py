"""Regression tests for performance reports and source provenance."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.performance_fixtures import source_manifest
from tools.performance_validation import check_result, compare_arrays, main


def test_comparison_rejects_nonfinite_and_changed_pose():
    with pytest.raises(AssertionError):
        compare_arrays({"references": np.ones(1)}, {"references": np.array([np.nan])})
    with pytest.raises(AssertionError):
        compare_arrays({"angle_deg": np.zeros(1)}, {"angle_deg": np.ones(1)})



def test_result_check_allows_float32_responsibility_sum_roundoff():
    result = SimpleNamespace(
        metadata={"backend": "cuda"},
        references=np.ones((1, 2, 2), dtype=np.float32),
        class_averages=np.ones((1, 2, 2), dtype=np.float32),
        responsibilities=np.array([[1.000002384185791]], dtype=np.float32),
        inlier_weights=np.ones(1, dtype=np.float32),
    )
    check_result(result, "cuda")
    result.responsibilities[0, 0] = 1.000006
    with pytest.raises(AssertionError):
        check_result(result, "cuda")



def test_source_manifest_ignores_appledouble_metadata(tmp_path):
    for name in (
        "pyproject.toml",
        "MANIFEST.in",
        "README.md",
        "LICENSE",
        "GPL-3.0.txt",
        "THIRD_PARTY_NOTICES.md",
    ):
        (tmp_path / name).write_text(name)
    for name in (
        "src",
        "packages",
        "tools",
        "tests",
        "docs",
        "examples",
        "third_party",
    ):
        (tmp_path / name).mkdir()
    (tmp_path / "tests/test_example.py").write_text("VALUE = 1\n")
    (tmp_path / "tests/._test_example.py").write_bytes(b"appledouble")

    manifest = source_manifest(tmp_path)

    assert "tests/test_example.py" in manifest["files"]
    assert "tests/._test_example.py" not in manifest["files"]



def test_performance_runner_writes_failure_report(tmp_path):
    output = tmp_path / "failed.json"
    assert (
        main(
            [
                "--backend",
                "cpu",
                "--baseline",
                str(tmp_path / "missing"),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and "traceback" in report
    with pytest.raises(SystemExit):
        main(["--backend", "cpu", "--output", str(output)])



def test_source_manifest_failure_is_also_recorded(tmp_path, monkeypatch):
    from tools import performance_validation

    def missing_source():
        raise FileNotFoundError("missing source file")

    monkeypatch.setattr(performance_validation, "source_manifest", missing_source)
    output = tmp_path / "missing-source.json"
    assert main(["--backend", "cpu", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["error"] == "missing source file"
