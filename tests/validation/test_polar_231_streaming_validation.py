import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tools import polar_231_streaming_validation as validation


@pytest.mark.parametrize("timing_events", [False, True])
def test_memory_trace_is_observational_and_events_can_be_disabled(timing_events):
    calls = {"memory": 0, "events": 0, "hooks": 0}

    class Hook:
        def __enter__(self):
            calls["hooks"] += 1

        def __exit__(self, *args):
            calls["hooks"] -= 1

    class Event:
        def __init__(self):
            calls["events"] += 1

        def record(self, stream):
            pass

        def synchronize(self):
            pass

    def memory():
        calls["memory"] += 1
        return 800, 1000

    cp = SimpleNamespace(
        cuda=SimpleNamespace(
            MemoryHook=Hook,
            Device=lambda: SimpleNamespace(id=0),
            runtime=SimpleNamespace(memGetInfo=memory),
            Event=Event,
            get_current_stream=lambda: None,
            get_elapsed_time=lambda *args: 1.0,
        ),
        get_default_memory_pool=lambda: SimpleNamespace(
            used_bytes=lambda: 20, total_bytes=lambda: 100
        ),
    )
    plain = validation.ExecutionProfile()
    plain.attach_cuda(cp)
    with plain.stage("outer"):
        with plain.stage("kernel", cuda=timing_events):
            pass
    plain.close()
    expected_queries, expected_events = calls["memory"], calls["events"]
    calls.update(memory=0, events=0)
    traced = validation.MemoryTraceProfile(timing_events=timing_events)
    traced.attach_cuda(cp)
    with pytest.raises(RuntimeError, match="injected"):
        with traced.stage("outer"):
            with traced.stage("kernel", cuda=True):
                raise RuntimeError("injected")
    traced.close()
    assert calls == {
        "memory": expected_queries,
        "events": expected_events,
        "hooks": 0,
    }
    assert traced.memory == plain.memory
    assert len(traced.trace) == expected_queries
    assert [s["sampling_scope"] for s in traced.trace] == [
        "profile_attach",
        "outer/kernel",
        "outer",
        "profile_close",
    ]
    assert traced.trace[1]["active_parent_stack"] == ["outer"]
    assert traced.trace[1]["pending_event_pairs"] == int(timing_events)
    assert traced.trace[-1]["pending_event_pairs"] == 0
    assert all(s["device_used_bytes"] == 200 for s in traced.trace)
    assert traced.trace[-1]["monotonic_ns"] >= traced.trace[0]["monotonic_ns"]
    json.dumps(traced.trace, allow_nan=False)


@pytest.mark.parametrize("unavailable", [False, True])
def test_device_context_snapshot_is_readonly_and_unavailable_is_not_zero(
    monkeypatch, unavailable
):
    def run(command, **kwargs):
        assert command == ["nvidia-smi", "-q", "-x"]
        assert kwargs["timeout"] == 5 and kwargs["capture_output"]
        if unavailable:
            raise FileNotFoundError("no nvidia-smi")
        return SimpleNamespace(returncode=0, stdout="<nvidia_smi_log/>", stderr="")

    monkeypatch.setattr(validation.subprocess, "run", run)
    value = validation.device_context_snapshot()
    assert value["status"] == ("unavailable" if unavailable else "captured")
    assert value["pid"] > 0
    if unavailable:
        assert "no nvidia-smi" in value["error"] and "xml" not in value
    else:
        assert value["xml"] == "<nvidia_smi_log/>"


def test_runner_refuses_overwrite(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text('{"frozen":true}\n')
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        validation.main(["--backend", "cuda", "--output", str(output)])
    assert json.loads(output.read_text()) == {"frozen": True}


def test_missing_gpu_is_failure_not_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(validation.batch, "load_frozen_backend", lambda: None)

    def unavailable(engine):
        raise RuntimeError("GPU unavailable")

    monkeypatch.setattr(validation.mirror, "runtime_identity", unavailable)
    output = tmp_path / "missing.json"
    assert validation.main(["--backend", "cuda", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and "GPU unavailable" in report["error"]


def test_runner_budget_patch_caps_real_budget_and_restores_on_failure(monkeypatch):
    from alignimg_gpu import backend
    from alignimg_gpu._polar_memory import plan_polar_storage

    config = replace(validation.mirror.mirror_config(), batch_size=512)
    args = (config, 32, 515, 3, 64, 8, 9, 2)
    low, high = validation.budget_interval(config, 515)
    cap = (low + high) // 2
    free = 8 * 1024**3

    def original(*args, **kwargs):
        return plan_polar_storage(
            *args,
            free_bytes=free,
            total_bytes=free,
            workflow_budget_bytes=cap + 100,
            live_cache_bytes=200,
        )

    monkeypatch.setattr(backend, "_polar_memory_plan", original)
    with pytest.raises(ValueError, match="injected"):
        with validation.limited_polar_budget(cap):
            result = backend._polar_memory_plan(*args)
            assert result.budget_bytes == cap - 100  # cannot enlarge real budget
            assert result.live_cache_bytes == 200
            assert result.particle_storage_policy == "streaming"
            assert result.fits_minimum
            raise ValueError("injected")
    assert backend._polar_memory_plan is original


def test_failed_resource_case_keeps_plan_and_active_case(monkeypatch):
    def fail(*args, report, **kwargs):
        report["memory_records"] = [{"budget_bytes": 1024, "fits_minimum": False}]
        raise MemoryError("injected resource failure")

    monkeypatch.setattr(validation, "measure_case", fail)
    report = {}
    with pytest.raises(MemoryError, match="injected"):
        validation.validate_streaming("cuda", report=report)
    assert report["active_case"] == "fixed_False"
    assert report["cases"]["fixed_False"]["resident"]["memory_records"] == [
        {"budget_bytes": 1024, "fits_minimum": False}
    ]


@pytest.mark.parametrize("extra_bytes", [0, 1])
@pytest.mark.parametrize("trace_memory", [False, True])
def test_device_budget_failure_keeps_numeric_evidence_in_json(
    tmp_path, monkeypatch, extra_bytes, trace_memory
):
    from alignimg_gpu import backend
    from tools import polar_resource_validation as resource

    memory = {
        "tracked_pool_live_bytes_peak": 100,
        "entry": {"device_used_bytes": 4096},
        "sampled_device_used_bytes_peak": 4096 + 1024 + extra_bytes,
    }

    class Profile:
        def __init__(self, **kwargs):
            self.memory = memory
            self.counters = {"d2h_bytes": 88}
            self.trace = [{"sampling_scope": "injected"}]

        def attach_cuda(self, cp):
            pass

        def close(self):
            pass

        def asdict(self):
            return {"gpu_memory": self.memory, "counters": self.counters}

    def infer(*args, memory_records, **kwargs):
        memory_records.extend(
            [
                {
                    "budget_bytes": 1024,
                    "estimated_peak_bytes": 512,
                    "fits_minimum": True,
                    "batch_size": 1,
                    "particle_storage_policy": "streaming",
                },
                {
                    "engine": "cuda",
                    "particle_storage_policy": "streaming",
                    "maximum_winner_batch_size": 1,
                    "full_correlation_map_d2h_bytes": 0,
                },
            ]
        )
        return {"score": np.ones(1, np.float32)}

    monkeypatch.setattr(validation, "ExecutionProfile", Profile)
    monkeypatch.setattr(validation, "MemoryTraceProfile", Profile)
    monkeypatch.setattr(
        validation, "device_context_snapshot", lambda: {"status": "captured"}
    )
    monkeypatch.setattr(backend, "_cupy", lambda: None)
    monkeypatch.setattr(backend, "_gpu_candidate_inference", infer)
    monkeypatch.setattr(validation.mirror, "synchronize", lambda *args: None)
    monkeypatch.setattr(validation.batch, "compare_batch_candidates", lambda *args: {})
    monkeypatch.setattr(resource, "runtime_info", lambda *args: {})
    monkeypatch.setattr(resource.mirror, "environment_info", lambda *args: {})
    monkeypatch.setattr(resource, "planning_checks", lambda: {})

    def validate(engine, *, report, **kwargs):
        assert kwargs["trace_memory"] is trace_memory
        report["active_case"] = "fixed_False"
        entry = report.setdefault("streaming", {})
        validation.measure_case(
            engine,
            SimpleNamespace(spatial=np.zeros((1, 8, 8))),
            None,
            None,
            None,
            None,
            report=entry,
            trace_memory=trace_memory,
        )

    monkeypatch.setattr(validation, "validate_streaming", validate)
    output = tmp_path / "resources.json"
    exit_code = resource.main(
        [
            "--suite",
            "resources",
            "--backend",
            "cuda",
            "--output",
            str(output),
        ]
        + (["--trace-memory"] if trace_memory else [])
    )
    report = json.loads(output.read_text())
    assert exit_code == extra_bytes
    assert report["status"] == ("failed" if extra_bytes else "passed")
    entry = report["checks"]["storage"]["streaming"]
    assert entry["observed_device_increase_bytes"] == 1024 + extra_bytes
    assert entry["profile"]["gpu_memory"] == memory
    assert entry["memory_records"][-2]["budget_bytes"] == 1024
    if trace_memory:
        assert entry["memory_trace"] == [{"sampling_scope": "injected"}]
        assert report["memory_diagnostics"]["after"]["status"] == "captured"
        assert report["diagnostic_only"] is True
    else:
        assert "memory_trace" not in entry and "memory_diagnostics" not in report
    if extra_bytes:
        for detail in (
            "observed=1025",
            "budget=1024",
            "device_entry=4096",
            "device_peak=5121",
            "policy=streaming",
            "batch=1",
        ):
            assert detail in report["error"]


def test_trace_survives_kernel_exception_and_profile_context_is_reset(monkeypatch):
    from alignimg_gpu import backend
    from alignimg._profiling import current_profile

    class Profile:
        def __init__(self, **kwargs):
            self.trace = []

        def attach_cuda(self, cp):
            self.trace.append({"sampling_scope": "profile_attach"})

        def close(self):
            self.trace.append({"sampling_scope": "profile_close"})

        def asdict(self):
            return {"partial": True}

    def infer(*args, **kwargs):
        if current_profile() is not None:
            raise RuntimeError("injected kernel failure")
        return {"score": np.ones(1)}

    monkeypatch.setattr(validation, "MemoryTraceProfile", Profile)
    monkeypatch.setattr(backend, "_cupy", lambda: None)
    monkeypatch.setattr(backend, "_gpu_candidate_inference", infer)
    monkeypatch.setattr(validation.mirror, "synchronize", lambda *args: None)
    report = {}
    with pytest.raises(RuntimeError, match="injected kernel failure"):
        validation.measure_case(
            "cuda", None, None, None, None, None, report=report, trace_memory=True
        )
    assert current_profile() is None
    assert report["memory_trace"] == [
        {"sampling_scope": "profile_attach"},
        {"sampling_scope": "profile_close"},
    ]
    assert report["profile"] == {"partial": True}
