from types import SimpleNamespace

import numpy as np
import pytest

import alignimg as ai
from alignimg._profiling import ExecutionProfile, _ACTIVE
from alignimg_gpu._workspace import WorkflowGpuWorkspace


def fake_cupy(free_bytes, total_bytes):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(memGetInfo=lambda: (free_bytes, total_bytes))
        )
    )


def test_workspace_reuses_two_fourier_roles_and_releases_them():
    gib = 1024**3
    records = []
    workspace = WorkflowGpuWorkspace(
        fake_cupy(20 * gib, 24 * gib),
        ai.AlignmentConfig(batch_size=512, memory_fraction=0.8),
        records,
    )
    scoring = np.zeros((4, 16, 16), dtype=np.complex64)
    update = np.ones_like(scoring)
    uploads = []

    def upload(values):
        device = SimpleNamespace(source=values, nbytes=values.nbytes)
        uploads.append(device)
        return device

    profile = ExecutionProfile()
    token = _ACTIVE.set(profile)
    try:
        first, plan = workspace.acquire_fourier(
            "scoring",
            scoring,
            fixed_bytes=4096,
            bytes_per_item=8192,
            upload=upload,
        )
        reused, reused_plan = workspace.acquire_fourier(
            "scoring",
            scoring,
            fixed_bytes=4096,
            bytes_per_item=8192,
            upload=upload,
        )
        raw, _ = workspace.acquire_fourier(
            "update",
            update,
            fixed_bytes=8192,
            bytes_per_item=4096,
            upload=upload,
        )
        assert first is reused and raw is not first
        assert plan.batch_size == reused_plan.batch_size == 512
        assert len(uploads) == 2
        workspace.close()
        workspace.close()
        assert profile.counters["workspace_scoring_cache_uploads"] == 1
        assert profile.counters["workspace_scoring_cache_hits"] == 1
        assert profile.counters["workspace_update_cache_uploads"] == 1
        assert profile.counters["workspace_release_calls"] == 1
    finally:
        _ACTIVE.reset(token)

    summary = workspace.summary()
    assert summary["closed"]
    assert summary["peak_committed_cache_bytes"] == scoring.nbytes + update.nbytes
    assert summary["released_cache_bytes"] == scoring.nbytes + update.nbytes
    assert summary["roles"]["scoring"]["requests"] == 2
    assert summary["roles"]["scoring"]["cache_hits"] == 1
    assert records[0]["event"] == "create"
    assert records[-1]["event"] == "release"


def test_workspace_streams_when_shared_budget_cannot_fit_cache():
    gib = 1024**3
    records = []
    workspace = WorkflowGpuWorkspace(
        fake_cupy(1 * gib, 8 * gib),
        ai.AlignmentConfig(batch_size=512, memory_fraction=0.8),
        records,
    )
    values = np.zeros((2, 16, 16), dtype=np.complex64)

    def forbidden_upload(_):
        raise AssertionError("budgeted-out cache must not be uploaded")

    for _ in range(2):
        device, plan = workspace.acquire_fourier(
            "scoring",
            values,
            fixed_bytes=4096,
            bytes_per_item=8192,
            upload=forbidden_upload,
        )
        assert device is None and plan.batch_size == 1
    state = workspace.summary()["roles"]["scoring"]
    assert state["policy"] == "host_streamed"
    assert state["host_streamed_requests"] == 2


def test_workspace_upload_oom_falls_back_and_live_cache_can_be_evicted():
    gib = 1024**3
    values = np.zeros((2, 16, 16), dtype=np.complex64)

    class OutOfMemoryError(RuntimeError):
        pass

    failed = WorkflowGpuWorkspace(
        fake_cupy(20 * gib, 24 * gib), ai.AlignmentConfig(batch_size=8), []
    )
    device, _ = failed.acquire_fourier(
        "scoring",
        values,
        fixed_bytes=0,
        bytes_per_item=8192,
        upload=lambda _: (_ for _ in ()).throw(OutOfMemoryError()),
    )
    assert device is None
    assert failed.summary()["roles"]["scoring"]["upload_oom_fallbacks"] == 1

    records = []
    workspace = WorkflowGpuWorkspace(
        fake_cupy(20 * gib, 24 * gib), ai.AlignmentConfig(batch_size=8), records
    )
    workspace.acquire_fourier(
        "update",
        values,
        fixed_bytes=0,
        bytes_per_item=8192,
        upload=lambda source: SimpleNamespace(nbytes=source.nbytes),
    )
    assert workspace.disable_cache("update", reason="test")
    assert not workspace.disable_cache("update", reason="test")
    device, _ = workspace.acquire_fourier(
        "update",
        values,
        fixed_bytes=0,
        bytes_per_item=8192,
        upload=lambda _: pytest.fail("evicted cache must remain streamed"),
    )
    assert device is None
    summary = workspace.summary()
    assert summary["roles"]["update"]["evictions"] == 1
    assert summary["released_cache_bytes"] == values.nbytes


def test_workspace_rejects_source_replacement_and_use_after_close():
    gib = 1024**3
    workspace = WorkflowGpuWorkspace(
        fake_cupy(20 * gib, 24 * gib), ai.AlignmentConfig(batch_size=8), []
    )
    values = np.zeros((2, 16, 16), dtype=np.complex64)
    kwargs = dict(
        fixed_bytes=0,
        bytes_per_item=8192,
        upload=lambda source: SimpleNamespace(nbytes=source.nbytes),
    )
    workspace.acquire_fourier("scoring", values, **kwargs)
    with pytest.raises(ValueError, match="source changed"):
        workspace.acquire_fourier("scoring", values.copy(), **kwargs)
    workspace.close()
    with pytest.raises(RuntimeError, match="closed"):
        workspace.acquire_fourier("update", values, **kwargs)


def test_workspace_honors_smaller_retry_batch_after_cache_eviction():
    gib = 1024**3
    workspace = WorkflowGpuWorkspace(
        fake_cupy(20 * gib, 24 * gib), ai.AlignmentConfig(batch_size=512), []
    )
    values = np.zeros((2, 16, 16), dtype=np.complex64)
    kwargs = dict(
        fixed_bytes=0,
        bytes_per_item=8192,
        upload=lambda source: SimpleNamespace(nbytes=source.nbytes),
    )
    workspace.acquire_fourier(
        "scoring", values, requested_batch_size=512, **kwargs
    )
    assert workspace.disable_cache("scoring", reason="test_retry")
    device, plan = workspace.acquire_fourier(
        "scoring", values, requested_batch_size=64, **kwargs
    )
    assert device is None
    assert plan.batch_size == 64
