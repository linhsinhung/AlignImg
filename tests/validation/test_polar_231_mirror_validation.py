from __future__ import annotations

import json

import pytest

from tools import polar_231_mirror_validation as validation


def test_mirror_runner_refuses_to_overwrite_existing_report(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text('{"frozen": true}\n')
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        validation.main(["--backend", "cuda", "--output", str(output)])
    assert json.loads(output.read_text()) == {"frozen": True}


def test_missing_native_is_failed_not_skipped(tmp_path, monkeypatch):
    def missing_native(engine):
        raise RuntimeError("native CUDA extension is missing")

    monkeypatch.setattr(validation, "runtime_identity", missing_native)
    output = tmp_path / "missing.json"
    assert validation.main(["--backend", "cuda", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert "native CUDA extension is missing" in report["error"]
    assert report["checks"] == {}
    assert report["source"]["source_sha256"]


def test_stale_gpu_source_fails_before_device_work(tmp_path, monkeypatch):
    from alignimg_gpu import backend

    stale = tmp_path / "backend.py"
    stale.write_text("# stale installed GPU source\n")
    monkeypatch.setattr(backend, "__file__", str(stale))
    monkeypatch.setattr(validation, "runtime_identity", lambda engine: {})
    monkeypatch.setattr(
        validation, "environment_info",
        lambda *args: pytest.fail("device accessed before identity check"),
    )
    output = tmp_path / "stale.json"
    assert validation.main(["--backend", "cuda", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert "installed source mismatch" in report["error"]
    assert report["checks"] == {}
