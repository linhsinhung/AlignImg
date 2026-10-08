import json

import pytest

from tools import polar_231_recovery_validation as validation


def test_recovery_runner_refuses_existing_report(tmp_path):
    output = tmp_path / "frozen.json"
    output.write_text('{"frozen":true}\n')
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        validation.main(["--backend", "cuda", "--output", str(output)])
    assert json.loads(output.read_text()) == {"frozen": True}


def test_recovery_runner_missing_gpu_is_failure_not_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(validation.batch, "load_frozen_backend", lambda: None)

    def unavailable(*args):
        raise RuntimeError("GPU unavailable")

    monkeypatch.setattr(validation.mirror, "runtime_identity", unavailable)
    output = tmp_path / "missing.json"
    assert validation.main(["--backend", "cuda", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and "GPU unavailable" in report["error"]


@pytest.mark.parametrize("batch", [1, 513])
def test_recovery_runner_rejects_batch_that_cannot_exercise_tail(tmp_path, batch):
    with pytest.raises(SystemExit) as error:
        validation.main(
            [
                "--backend",
                "cuda",
                "--batch-size",
                str(batch),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
    assert error.value.code != 0


def test_injection_restores_hooks_and_non_oom_is_not_swallowed(monkeypatch):
    from alignimg_gpu import backend

    def fail(*args, **kwargs):
        raise ValueError("not an allocation error")

    monkeypatch.setattr(backend, "_gpu_polar_hard_candidate_inference_once", fail)
    names = (
        "_asdevice",
        "_ashost",
        "_polar_memory_plan",
        "_gpu_polar_batch_solver",
        "_gpu_polar_hard_candidate_inference_once",
    )
    originals = {name: getattr(backend, name) for name in names}
    with pytest.raises(ValueError, match="not an allocation error"):
        with validation.injected_oom("mid_batch", (17, 32, 32)):
            backend._gpu_polar_hard_candidate_inference_once(
                particle_storage_policy="resident",
                particle_limit=7,
            )
    assert all(getattr(backend, name) is value for name, value in originals.items())


@pytest.mark.gpu
@pytest.mark.parametrize("engine", ["cupy", "cuda"])
def test_real_gpu_t5_recovery_and_single_mstep(engine):
    validation.mirror.runtime_identity(engine)
    validation.validate_recovery(engine)
    validation.validate_workflow(engine, report={})
