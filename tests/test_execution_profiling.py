from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

import alignimg as ai
from alignimg._profiling import ExecutionProfile, _ACTIVE, current_profile
from tools.performance_fixtures import (
    DEFAULT_BASELINE,
    fixture_config,
    sha256,
    synthetic_inputs,
)
from tools.performance_validation import (
    check_result,
    compare_arrays,
    execute,
    frozen_fixture_check,
    main,
    profiling_case,
    result_arrays,
)


@pytest.mark.parametrize("workflow", ["global", "refine", "reference_free"])
@pytest.mark.parametrize("raw", [False, True])
def test_profile_is_observational_and_covers_complete_workflow(workflow, raw):
    values = synthetic_inputs(count=4)
    config = replace(
        fixture_config(workflow == "refine"),
        max_iterations=2,
        apply_final_pose_to_raw=raw,
    )
    plain = execute(values, config, "cpu", workflow)
    profiled = execute(values, replace(config, profile_execution=True), "cpu", workflow)
    assert "performance" not in plain.metadata
    for key, value in result_arrays(plain).items():
        np.testing.assert_array_equal(value, result_arrays(profiled)[key], err_msg=key)
    report = profiled.metadata["performance"]
    assert report["gpu_memory"] is None
    assert len(report["iterations"]) == 2
    assert "workflow/final_raw_average" in report["stages"]
    if workflow == "reference_free":
        assert "workflow/spectral_bootstrap" in report["stages"]
    for item in report["iterations"]:
        assert (
            item["wall_seconds_including_diagnostics"]
            >= item["legacy_seconds_excluding_halfsets"]
        )
    for item in report["stages"].values():
        assert 0 <= item["exclusive_wall_seconds"] <= item["wall_seconds"]
        assert item["cuda_event_span_seconds"] is None
    json.dumps(report, allow_nan=False)
    assert current_profile() is None


def test_profile_flag_does_not_disable_default_refine_normalization():
    plain = asdict(ai.AlignmentConfig().normalized(workflow="refine"))
    profiled = asdict(
        ai.AlignmentConfig(profile_execution=True).normalized(workflow="refine")
    )
    profiled["profile_execution"] = False
    assert plain == profiled
    assert plain["max_iterations"] == 5 and plain["robust_weighting"]


def test_profile_cleanup_on_workflow_exception():
    with pytest.raises(ValueError):
        ai.align_to_references(
            np.zeros((1, 7, 7)),
            np.zeros((7, 7)),
            config=ai.AlignmentConfig(profile_execution=True),
        )
    assert current_profile() is None
    result = execute(synthetic_inputs(2), fixture_config(False), "cpu", "global")
    assert "performance" not in result.metadata


def test_nested_accounting_and_transfer_counters():
    profile = ExecutionProfile()
    token = _ACTIVE.set(profile)
    try:
        with profile.stage("outer"):
            with profile.stage("inner"):
                profile.count("h2d_bytes", 128)
        assert profile.counters["h2d_bytes"] == 128
        assert profile.stages["outer"]["counters"]["h2d_bytes"] == 128
        assert profile.stages["outer/inner"]["counters"]["h2d_bytes"] == 128
        outer, inner = profile.stages["outer"], profile.stages["outer/inner"]
        assert outer["wall_seconds"] >= inner["wall_seconds"]
        assert outer["exclusive_wall_seconds"] == pytest.approx(
            outer["wall_seconds"] - inner["wall_seconds"]
        )
    finally:
        _ACTIVE.reset(token)


def test_cuda_hook_tracks_pool_reuse_and_releases_hook_without_cuda():
    hooks = []

    class Hook:
        def __enter__(self):
            hooks.append(self)

        def __exit__(self, *_):
            hooks.remove(self)

    class Event:
        def record(self, stream):
            assert stream == "current_stream"

        def synchronize(self):
            pass

    fake_cp = SimpleNamespace(
        cuda=SimpleNamespace(
            MemoryHook=Hook,
            Device=lambda: SimpleNamespace(id=0),
            Event=Event,
            get_current_stream=lambda: "current_stream",
            get_elapsed_time=lambda a, b: 2.0,
            runtime=SimpleNamespace(memGetInfo=lambda: (800, 1000)),
        ),
        get_default_memory_pool=lambda: SimpleNamespace(
            used_bytes=lambda: 20, total_bytes=lambda: 100
        ),
    )
    profile = ExecutionProfile()
    profile.attach_cuda(fake_cp)
    hook = hooks[0]
    for key, size in [(1, 512), (2, 1024)]:
        hook.malloc_postprocess(
            device_id=0, size=size, mem_size=size, mem_ptr=key, pmem_id=key
        )
    hook.free_postprocess(device_id=0, mem_size=512, mem_ptr=1, pmem_id=1)
    hook.malloc_postprocess(device_id=0, size=512, mem_size=512, mem_ptr=1, pmem_id=3)
    assert profile.memory["tracked_pool_live_bytes_peak"] == 1536
    assert hook.live_bytes == 1536
    # Simulate more than a flush window: event handles must remain bounded.
    for _ in range(260):
        with profile.stage("kernel", cuda=True):
            pass
        assert len(profile.events) < 128
    profile.close()
    assert not hooks and not profile.events
    assert profile.stages["kernel"]["cuda_event_span_seconds"] == pytest.approx(0.52)
    assert profile.memory["sampled_device_used_bytes_peak"] == 200


def test_transfer_wrappers_count_payloads_without_extra_device_probes(monkeypatch):
    import sys
    from alignimg_gpu import backend

    class DeviceArray:
        def __init__(self, value, **kwargs):
            self.value = np.asarray(value, **kwargs)
            self.nbytes = self.value.nbytes

    fake_cp = SimpleNamespace(
        ndarray=DeviceArray,
        asarray=DeviceArray,
        asnumpy=lambda value: value.value.copy(),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)

    def forbidden_probe():
        raise AssertionError("extra CUDA device probe")

    monkeypatch.setattr(backend, "_cupy", forbidden_probe)
    values = np.arange(3, dtype=np.float32)
    profile = ExecutionProfile()
    token = _ACTIVE.set(profile)
    try:
        device = backend._asdevice(values)
        host = backend._ashost(device)
        np.testing.assert_array_equal(host, values)
        assert profile.counters == {
            "h2d_calls": 1,
            "h2d_bytes": 12,
            "d2h_calls": 1,
            "d2h_bytes": 12,
        }
        assert profile.stages["host_to_device"]["counters"]["h2d_bytes"] == 12
        backend._record_native_pose_upload(3)
        assert profile.counters["h2d_bytes"] == 51
        assert profile.counters["native_pose_h2d_calls"] == 4
    finally:
        _ACTIVE.reset(token)


def test_frozen_preinstrumentation_fixtures():
    if not DEFAULT_BASELINE.exists():
        pytest.skip(
            "frozen local baseline artifacts are not included in the distribution"
        )
    report = frozen_fixture_check(DEFAULT_BASELINE)
    assert set(report) == {"global", "adaptive"}
    assert all(
        value["maximum_absolute_error"] == 0
        for case in report.values()
        for value in case.values()
    )
    manifest = json.loads((DEFAULT_BASELINE / "manifest.json").read_text())
    assert sha256(DEFAULT_BASELINE / "source.tar.gz") == manifest["archive_sha256"]


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


def test_fixture_check_rejects_missing_outputs(monkeypatch):
    if not DEFAULT_BASELINE.exists():
        pytest.skip("requires frozen local fixtures")
    from tools import performance_validation

    monkeypatch.setattr(performance_validation, "capture_iteration", lambda *args: {})
    with pytest.raises(AssertionError, match="fields differ"):
        frozen_fixture_check(DEFAULT_BASELINE)


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


@pytest.mark.parametrize("backend", ["cuda", "cupy"])
def test_gpu_profiling_conformance(backend):
    if not ai.available_alignment_backends()[backend]["available"]:
        pytest.skip(f"{backend} unavailable")
    report = profiling_case(backend, 16)
    assert report["performance"]["gpu_memory"]["tracked_pool_live_bytes_peak"] > 0
    assert report["performance"]["counters"]["d2h_bytes"] > 0
    assert any(
        v["cuda_event_span_seconds"] is not None
        for v in report["performance"]["stages"].values()
    )
