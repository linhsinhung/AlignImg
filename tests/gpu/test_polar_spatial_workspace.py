"""Workflow lifetime and Fourier-priority checks without GPU allocations."""

from types import SimpleNamespace

import numpy as np
import pytest

import alignimg as ai
from alignimg._profiling import ExecutionProfile, _ACTIVE
from alignimg_gpu._workspace import WorkflowGpuWorkspace


def workspace_with_budget(budget=10_000, batch=8):
    cp = SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(memGetInfo=lambda: (256 * 1024**2 + budget, 1024**3))
        )
    )
    records = []
    workspace = WorkflowGpuWorkspace(
        cp, ai.AlignmentConfig(batch_size=batch, memory_fraction=1.0), records
    )
    assert workspace.allocation_budget()["workflow_budget_bytes"] == budget
    return workspace, records


def upload(values):
    return SimpleNamespace(nbytes=values.nbytes)


def forbidden_upload(_):
    pytest.fail("spatial cache must not upload again")


def test_spatial_upload_once_hits_and_close_share_live_accounting(monkeypatch):
    workspace, records = workspace_with_budget()
    values = np.zeros(1000, dtype=np.float32)
    monkeypatch.setattr(workspace, "_plan", lambda *a, **k: pytest.fail("caller plans admission"))
    profile = ExecutionProfile()
    token = _ACTIVE.set(profile)
    try:
        first = workspace.acquire_spatial(values, upload=upload)
        for _ in range(2):
            assert workspace.acquire_spatial(values, upload=forbidden_upload) is first
        assert workspace.cache_bytes("polar_spatial") == values.nbytes
        assert workspace.cache_bytes("scoring") == 0
        assert workspace.allocation_budget()["live_cache_bytes"] == values.nbytes
        assert workspace.spatial_cache_allowed()
        assert profile.counters["workspace_polar_spatial_cache_requests"] == 3
        assert profile.counters["workspace_polar_spatial_cache_uploads"] == 1
        assert profile.counters["workspace_polar_spatial_cache_upload_bytes"] == values.nbytes
        assert profile.counters["workspace_polar_spatial_cache_hits"] == 2
        workspace.close()
        workspace.close()
        assert profile.counters["workspace_release_calls"] == 1
        assert profile.counters["workspace_released_bytes"] == values.nbytes
    finally:
        _ACTIVE.reset(token)
    assert not workspace.spatial_cache_allowed()
    assert workspace.cache_bytes("polar_spatial") == 0
    assert workspace.summary()["peak_committed_cache_bytes"] == values.nbytes
    spatial = [record for record in records if record["stage"] == "particle_spatial_cache"]
    assert [record["event"] for record in spatial] == ["upload", "reuse", "reuse"]
    with pytest.raises(RuntimeError, match="closed"):
        workspace.acquire_spatial(values, upload=forbidden_upload)


@pytest.mark.parametrize("change", ["identity", "shape", "dtype"])
def test_spatial_source_token_rejects_changed_source(change):
    workspace, _ = workspace_with_budget()
    values = np.zeros((8, 8), dtype=np.float32)
    workspace.acquire_spatial(values, upload=upload)
    if change == "identity":
        values = values.copy()
    elif change == "shape":
        values.shape = (4, 16)
    else:
        values = values.view(np.int32)
    with pytest.raises(ValueError, match="source changed"):
        workspace.acquire_spatial(values, upload=forbidden_upload)


@pytest.mark.parametrize("initial_upload", [False, True])
def test_spatial_disable_is_permanent_and_counts_only_live_eviction(initial_upload):
    workspace, records = workspace_with_budget()
    values = np.zeros(1000, dtype=np.float32)
    if initial_upload:
        workspace.acquire_spatial(values, upload=upload)
    assert workspace.disable_cache("polar_spatial", reason="budget_rejection") is initial_upload
    assert not workspace.disable_cache("polar_spatial", reason="repeated_rejection")
    assert not workspace.spatial_cache_allowed()
    assert workspace.cache_bytes("polar_spatial") == 0
    assert workspace.acquire_spatial(values, upload=forbidden_upload) is None
    assert workspace.allocation_budget()["live_cache_bytes"] == 0
    workspace.close()
    summary = workspace.summary()
    assert summary["released_cache_bytes"] == values.nbytes * int(initial_upload)
    assert summary["roles"]["polar_spatial"]["evictions"] == int(initial_upload)
    events = [record for record in records if record["event"] in {"disable", "evict"}]
    assert len(events) == 1
    assert events[0]["evicted_bytes"] == values.nbytes * int(initial_upload)
    assert events[0]["event"] == ("evict" if initial_upload else "disable")


@pytest.mark.parametrize("oom", [False, True])
def test_spatial_upload_error_propagates_unchanged_and_only_oom_disables(oom):
    workspace, _ = workspace_with_budget()
    values = np.zeros(1000, dtype=np.float32)
    error = RuntimeError("out of memory" if oom else "invalid device operation")

    def fail(_):
        raise error

    with pytest.raises(RuntimeError) as caught:
        workspace.acquire_spatial(values, upload=fail)
    assert caught.value is error
    assert workspace.cache_bytes("polar_spatial") == 0
    assert workspace.allocation_budget()["live_cache_bytes"] == 0
    assert workspace.summary()["roles"]["polar_spatial"]["uploads"] == 0
    assert workspace.spatial_cache_allowed() is not oom
    if oom:
        assert workspace.acquire_spatial(values, upload=forbidden_upload) is None
        assert workspace.summary()["roles"]["polar_spatial"]["upload_ooms"] == 1
    else:
        assert workspace.acquire_spatial(values, upload=upload) is not None


def test_spatial_runner_opt_out_never_plans_or_uploads(monkeypatch):
    workspace, _ = workspace_with_budget()
    monkeypatch.setattr(workspace, "spatial_cache_allowed", lambda: False)
    monkeypatch.setattr(workspace, "_plan", lambda *a, **k: pytest.fail("disabled planner"))
    assert workspace.acquire_spatial(np.zeros(4, dtype=np.float32), upload=forbidden_upload) is None
    assert workspace.allocation_budget()["live_cache_bytes"] == 0


@pytest.mark.parametrize("role", ["scoring", "update"])
@pytest.mark.parametrize(
    "spatial_size,fourier_size,fixed,per_item,requested,evicted,expected_batch,device_cache",
    [
        (4000, 2000, 500, 1000, 2, False, 2, True),
        (4000, 2000, 500, 1000, 8, True, 7, True),
        (6000, 6000, 100, 1000, 8, True, 3, True),
        # Neither Fourier cache fits, but the host-streamed M-step needs eviction.
        (8000, 20000, 2500, 500, 8, True, 8, False),
        # Spatial eviction cannot solve an independently infeasible fixed workspace.
        (4000, 20000, 15000, 500, 8, False, 1, False),
    ],
)
def test_fourier_priority_admission_uses_live_budget_once(
    role, spatial_size, fourier_size, fixed, per_item, requested,
    evicted, expected_batch, device_cache,
):
    workspace, records = workspace_with_budget()
    spatial = np.zeros(spatial_size // 4, dtype=np.float32)
    fourier = np.zeros(fourier_size // 8, dtype=np.complex64)
    workspace.acquire_spatial(spatial, upload=upload)
    device, plan = workspace.acquire_fourier(
        role, fourier, fixed_bytes=fixed, bytes_per_item=per_item,
        requested_batch_size=requested, upload=upload,
    )
    assert (device is not None) is device_cache
    assert plan.batch_size == expected_batch
    expected_live = spatial_size * int(not evicted) + fourier_size * int(device_cache)
    assert workspace.allocation_budget()["live_cache_bytes"] == expected_live
    assert plan.fixed_bytes == fixed + expected_live
    assert workspace.cache_bytes("polar_spatial") == spatial_size * int(not evicted)
    assert workspace.cache_bytes(role) == fourier_size * int(device_cache)
    assert workspace.spatial_cache_allowed() is not evicted
    events = [record for record in records if record["event"] == "evict"]
    assert len(events) == int(evicted)
    if evicted:
        assert events[0]["reason"] == f"{role}_fourier_priority"
        assert workspace.acquire_spatial(spatial, upload=forbidden_upload) is None
    workspace.close()
    assert workspace.summary()["released_cache_bytes"] == spatial_size + fourier_size * int(device_cache)


@pytest.mark.parametrize("cached_fourier", [False, True])
def test_fourier_reuse_also_evicts_spatial_before_workspace_allocation(cached_fourier):
    workspace, _ = workspace_with_budget()
    fourier = np.zeros(250 if cached_fourier else 2500, dtype=np.complex64)
    first, _ = workspace.acquire_fourier(
        "update", fourier, fixed_bytes=0, bytes_per_item=500, upload=upload
    )
    assert (first is not None) is cached_fourier
    spatial = np.zeros(1000 if cached_fourier else 2000, dtype=np.float32)
    workspace.acquire_spatial(spatial, upload=upload)
    device, plan = workspace.acquire_fourier(
        "update", fourier, fixed_bytes=0 if cached_fourier else 2500,
        bytes_per_item=1000 if cached_fourier else 500, upload=forbidden_upload,
    )
    assert device is first
    assert plan.fits_minimum and plan.batch_size == 8
    assert workspace.cache_bytes("polar_spatial") == 0
    assert workspace.cache_bytes("update") == (fourier.nbytes if cached_fourier else 0)
    assert not workspace.spatial_cache_allowed()
