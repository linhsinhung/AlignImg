"""T5 recovery control and failed-frame lifetime, without requiring CUDA."""

from dataclasses import replace
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from alignimg_gpu import backend
from alignimg_gpu._polar_memory import plan_polar_storage
from tools import polar_231_recovery_validation as validation
from test_polar_231_batch import emulated_backends  # noqa: F401 (pytest fixture)


class OutOfMemoryError(RuntimeError):
    pass


def controller(monkeypatch, *, streaming=False, batch=1, release=lambda: None):
    config = replace(
        backend.AlignmentConfig.preset("fast3"),
        batch_size=batch,
        angle_samples=64,
        mask_radius=8.0,
        translation_range=0.0,
        translation_step=1.0,
        mirror_search=True,
    )
    cp = SimpleNamespace(
        get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=release)
    )
    monkeypatch.setattr(backend, "_cupy", lambda: cp)

    def planner(config, *args, force_streaming=False, particle_limit=None, **kwargs):
        return plan_polar_storage(
            config,
            32,
            17,
            3,
            64,
            8,
            1,
            2,
            free_bytes=4 * 1024**3,
            total_bytes=8 * 1024**3,
            force_streaming=streaming or force_streaming,
            particle_limit=particle_limit,
        )

    monkeypatch.setattr(backend, "_polar_memory_plan", planner)
    particles = SimpleNamespace(spatial=np.zeros((17, 32, 32), np.float32))
    references = SimpleNamespace(spatial=np.zeros((3, 32, 32), np.float32))
    return particles, references, config, None, 0.08, None


def test_resident_batch_one_oom_still_switches_to_streaming(monkeypatch):
    args = controller(monkeypatch)
    calls = []

    def solve(*args, **kwargs):
        calls.append((kwargs["particle_storage_policy"], kwargs["particle_limit"]))
        if len(calls) == 1:
            raise OutOfMemoryError("injected resident upload")
        return {"done": True}

    monkeypatch.setattr(backend, "_gpu_polar_hard_candidate_inference_once", solve)
    assert backend._gpu_candidate_inference(*args) == {"done": True}
    assert calls == [("resident", 1), ("streaming", 1)]


@pytest.mark.parametrize("retain_cause", [False, True])
def test_retained_exceptions_do_not_keep_failed_device_array_alive(
    monkeypatch, retain_cause
):
    buffers, errors, causes = [], [], []

    def release():
        assert all(reference() is None for reference in buffers)

    args = controller(monkeypatch, release=release)

    def inner():
        device_buffer = np.zeros(16, np.float32)
        buffers.append(weakref.ref(device_buffer))
        cause = RuntimeError("inner allocation")
        if retain_cause:
            causes.append(cause)
        raise cause

    def solve(*args, **kwargs):
        if not errors:
            error = OutOfMemoryError("injected retained allocation failure")
            errors.append(error)
            try:
                inner()
            except RuntimeError as cause:
                raise error from cause
        return {"done": True}

    monkeypatch.setattr(backend, "_gpu_polar_hard_candidate_inference_once", solve)
    assert backend._gpu_candidate_inference(*args) == {"done": True}
    assert buffers[0]() is None
    for error in errors + causes:
        assert error.__traceback__ is error.__context__ is error.__cause__ is None


def test_streaming_batch_one_reclaims_caches_before_failing(monkeypatch):
    args = controller(monkeypatch, streaming=True)
    live = {"update", "scoring"}
    evicted = []

    def disable_cache(role, *, reason):
        if role not in live:
            return False
        live.remove(role)
        evicted.append(role)
        return True

    workspace = SimpleNamespace(disable_cache=disable_cache)

    def solve(*args, **kwargs):
        assert kwargs["particle_storage_policy"] == "streaming"
        assert kwargs["particle_limit"] == 1
        if live:
            raise OutOfMemoryError("injected cache pressure")
        return {"done": True}

    monkeypatch.setattr(backend, "_gpu_polar_hard_candidate_inference_once", solve)
    assert backend._gpu_candidate_inference(*args, workspace=workspace) == {
        "done": True
    }
    assert evicted == ["update", "scoring"]


def test_recovery_evicts_before_halving_and_records_actual_plans(monkeypatch):
    args = controller(monkeypatch, batch=8)
    live, calls, records = {"update", "scoring"}, [], []

    def disable(role, *, reason):
        if role not in live:
            return False
        live.remove(role)
        return True

    def solve(*args, **kwargs):
        calls.append((kwargs["particle_storage_policy"], kwargs["particle_limit"]))
        if kwargs["particle_limit"] > 2:
            raise OutOfMemoryError("injected pressure")
        return {"done": True}

    monkeypatch.setattr(backend, "_gpu_polar_hard_candidate_inference_once", solve)
    backend._gpu_candidate_inference(
        *args, workspace=SimpleNamespace(disable_cache=disable), memory_records=records
    )
    assert calls == [
        ("resident", 8),
        ("streaming", 8),
        ("streaming", 8),
        ("streaming", 4),
        ("streaming", 2),
    ]
    assert [r["action"] for r in records if r.get("event") == "oom_retry"] == [
        "streaming",
        "evict_caches",
        "halve_batch",
        "halve_batch",
    ]
    plans = [r for r in records if "event" not in r]
    assert [r["actual_particle_batch"] for r in plans] == [8, 8, 8, 4, 2]
    assert [r["oom_retry_count"] for r in plans] == [0, 1, 2, 3, 4]
    assert all(
        r["requested_particle_batch"] == 8
        and r["fits_minimum"]
        and r["estimated_peak_bytes"] <= r["budget_bytes"]
        for r in plans
    )


def test_terminal_batch_one_preserves_report_and_releases_arrays(monkeypatch):
    refs = []

    def release():
        assert all(ref() is None for ref in refs)

    args = controller(
        monkeypatch,
        streaming=True,
        release=release,
    )

    def solve(*args, **kwargs):
        device = np.zeros(16, np.float32)
        refs.append(weakref.ref(device))
        raise OutOfMemoryError("injected permanent failure")

    monkeypatch.setattr(backend, "_gpu_polar_hard_candidate_inference_once", solve)
    with pytest.raises(MemoryError, match="streaming batch 1") as error:
        backend._gpu_candidate_inference(*args)
    records = error.value.polar_memory_records
    assert len(refs) == 1 and refs[0]() is None
    assert records[-1]["event"] == "oom_failure"
    assert records[-1]["actual_particle_batch"] == 1
    assert records[-1]["oom_retry_count"] == 0


def test_non_oom_propagates_without_retry_or_cache_release(monkeypatch):
    released, calls = [], []
    args = controller(monkeypatch, release=lambda: released.append(1))
    original = ValueError("invalid input")

    def solve(*args, **kwargs):
        calls.append(1)
        raise original

    monkeypatch.setattr(backend, "_gpu_polar_hard_candidate_inference_once", solve)
    with pytest.raises(ValueError) as error:
        backend._gpu_candidate_inference(*args)
    assert error.value is original
    assert len(calls) == 1 and not released


@pytest.fixture
def emulated_runtime(monkeypatch, request):
    request.getfixturevalue("emulated_backends")
    cp = SimpleNamespace(**{name: getattr(np, name) for name in dir(np)})
    cp.get_default_memory_pool = lambda: SimpleNamespace(free_all_blocks=lambda: None)
    cp.cuda = SimpleNamespace(
        Device=lambda: SimpleNamespace(id=0),
        runtime=SimpleNamespace(
            memGetInfo=lambda: (8 * 1024**3, 8 * 1024**3),
            getDeviceProperties=lambda device: {"name": b"emulated device"},
        ),
    )
    monkeypatch.setattr(backend, "_cupy", lambda: cp)
    return cp


@pytest.mark.parametrize("fixed", [False, True])
def test_mid_batch_discards_partial_host_output_and_restarts_all_particles(
    emulated_runtime,
    fixed,
):
    from alignimg._fourier import prepare_stack

    images, references, centers, labels, config = validation.batch.batch_fixture(17)
    config = replace(config, batch_size=7)
    particles, refs = prepare_stack(images, config), prepare_stack(references, config)
    priors = (
        np.eye(3, dtype=np.float32)[labels]
        if fixed
        else np.full((17, 3), 1 / 3, np.float32)
    )
    arguments = (particles, refs, config, priors, 0.08, None)
    expected = backend._gpu_candidate_inference(*arguments, translation_centers=centers)
    records = []
    with validation.injected_oom("mid_batch", images.shape) as faults:
        actual = backend._gpu_candidate_inference(
            *arguments,
            translation_centers=centers,
            memory_records=records,
        )
    validation.batch.compare_batch_candidates(expected, actual)
    assert faults["attempts"] == [
        {"policy": "resident", "batch": 7},
        {"policy": "streaming", "batch": 7},
    ]
    assert faults["solver_calls"] == 5  # first two, then a fresh three batches
    assert faults["compact_result_d2h_bytes"] == (17 + 7) * 11 * 8
    assert faults["released_before_replan"] and faults["failed_arrays_released"]
    assert faults["injected_tracebacks_detached"]
    assert sum(r.get("event") == "complete" for r in records) == 1


def test_emulated_gpu_workflow_updates_once_per_success_and_closes_on_failure(
    emulated_runtime,
    monkeypatch,
):
    from alignimg._engine import _update_references_fourier

    images, references, _, labels, config = validation.batch.batch_fixture(21)
    config = replace(
        config, translation_range=0.0, batch_size=7, profile_execution=False
    )
    priors = np.eye(3, dtype=np.float32)[labels]
    steps, workspaces = [], []
    workspace_type = backend.WorkflowGpuWorkspace

    def workspace(*args):
        value = workspace_type(*args)
        workspaces.append(value)
        return value

    def update(*args, **kwargs):
        steps.append(1)
        return _update_references_fourier(
            *args, particle_fourier=kwargs.get("particle_fourier")
        )

    monkeypatch.setattr(backend, "WorkflowGpuWorkspace", workspace)
    monkeypatch.setattr(backend, "_gpu_reference_updater", update)
    arguments = dict(
        config=config,
        class_priors=priors,
        initial_poses=None,
        workflow="global",
        engine="cupy",
    )
    expected = backend._run_soft_alignment_gpu_engine(images, references, **arguments)
    steps.clear()
    with validation.injected_oom("mid_batch", images.shape):
        actual = backend._run_soft_alignment_gpu_engine(images, references, **arguments)
    validation.mirror.compare_workflows(expected, actual)
    assert len(steps) == 3 and actual.metadata["gpu_workspace"]["closed"]
    steps.clear()
    with validation.injected_oom("batch1", images.shape):
        with pytest.raises(MemoryError, match="batch 1") as error:
            backend._run_soft_alignment_gpu_engine(images, references, **arguments)
    assert not steps
    assert all(value.summary()["closed"] for value in workspaces)
    assert error.value.polar_memory_records[-1]["event"] == "release"
    assert error.value.polar_memory_records[-2]["event"] == "oom_failure"
