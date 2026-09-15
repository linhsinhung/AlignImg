import json
import subprocess
import sys

import pytest
import alignimg

from tools.stage1_validation import ROOT, compare_reports, unpack_baseline


def test_stage1_runs_frozen_and_current_engines_and_checks_artifacts(tmp_path):
    if alignimg.__version__ != "2.1.0.dev1":
        pytest.skip("historical stage 1 runner requires the frozen dev1 checkout")
    if not (
        ROOT / "validation-results/performance/stage-0/dev0-source-clean-delivery"
    ).exists():
        pytest.skip("requires the accepted local source snapshot")
    output = tmp_path / "ab.json"
    command = [
        sys.executable,
        str(ROOT / "tools/stage1_validation.py"),
        "--backend",
        "cpu",
        "--only",
        "global",
        "--repeats",
        "1",
        "--output",
        str(output),
    ]
    run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    report = json.loads(output.read_text())
    assert report["status"] in {"passed", "repeat_required"}, (
        run.stdout,
        run.stderr,
        report,
    )
    old = json.loads(output.with_suffix(".baseline.json").read_text())
    new = json.loads(output.with_suffix(".current.json").read_text())
    assert old["version"] == "2.1.0.dev0"
    assert new["environment"]["alignimg_module_version"] == "2.1.0.dev1"
    assert all(
        v["maximum_absolute_error"] == 0
        for v in report["cases"]["global"]["numeric_errors"].values()
    )
    before = old["cases"]["global"]["performance"]["counters"][
        "polar_angular_fft_calls"
    ]
    after = new["cases"]["global"]["performance"]["counters"]["polar_angular_fft_calls"]
    assert (before, after) == (
        16,
        10,
    )  # One-hot priors allow one reference per particle.
    result = tmp_path / "ab.current.global.result.npz"
    result.write_bytes(b"corrupt result")
    with pytest.raises(AssertionError, match="hash mismatch"):
        compare_reports(
            output.with_suffix(".baseline.json"), output.with_suffix(".current.json")
        )
    retry = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert retry.returncode != 0 and "output exists" in retry.stderr


def test_baseline_unpack_rejects_unrecognized_source(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "manifest.json").write_text(
        json.dumps({"source_sha256": "wrong", "files": {}})
    )
    with pytest.raises(ValueError, match="accepted dev0"):
        unpack_baseline(baseline, tmp_path / "extracted")
