"""Spatial-source retention changes allocation lifetime, never polar arithmetic."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from alignimg._fourier import prepare_stack
from alignimg_gpu import backend
from alignimg_gpu._polar_memory import plan_polar_storage, polar_allocation_inventory
from alignimg_gpu._workspace import WorkflowGpuWorkspace
from tools import polar_231_batch_validation as validation
from test_polar_231_batch import emulated_backends  # noqa: F401
from test_polar_231_recovery import emulated_runtime as emulated_runtime


GIB = 1024**3


def test_cached_source_can_use_small_solver_batch_without_relaxing_inventory():
    config = replace(validation.mirror.mirror_config(), batch_size=512)
    args = (config, 32, 515, 3, 64, 8, 9, 2)
    fixed, item = polar_allocation_inventory(32, 3, 64, 8, 9, 2)
    source_bytes = 515 * 32 * 32 * 4
    budget = sum(fixed.values()) + source_bytes + 7 * sum(item.values())
    options = dict(free_bytes=8 * GIB, total_bytes=8 * GIB, workflow_budget_bytes=budget)
    legacy = plan_polar_storage(*args, **options)
    cached = plan_polar_storage(*args, **options, allow_spatial_cache=True)
    assert legacy.particle_storage_policy == "streaming"
    assert cached.particle_storage_policy == "cached" and cached.batch_size == 7
    assert cached.target_particle_batch == cached.requested_batch_size == 512
    assert cached.item_components == item
    assert cached.fixed_components == {**fixed, "new_spatial_cache_fp32": source_bytes}
    assert cached.fits_minimum and cached.estimated_peak_bytes == budget

    reused = plan_polar_storage(
        *args, **{**options, "free_bytes": 8 * GIB - source_bytes},
        allow_spatial_cache=True, live_cache_bytes=source_bytes,
        spatial_cache_bytes=source_bytes,
    )
    assert reused.particle_storage_policy == "cached" and reused.batch_size == 7
    assert reused.fixed_components == fixed
    assert reused.budget_bytes == budget - source_bytes
    assert reused.estimated_peak_bytes + reused.live_cache_bytes == budget


def test_cache_admission_keeps_streaming_when_whole_source_cannot_fit():
    config = replace(validation.mirror.mirror_config(), batch_size=512)
    args = (config, 32, 515, 3, 64, 8, 9, 2)
    fixed, item = polar_allocation_inventory(32, 3, 64, 8, 9, 2)
    budget = sum(fixed.values()) + sum(item.values()) + 32 * 32 * 4
    plan = plan_polar_storage(
        *args, free_bytes=8 * GIB, total_bytes=8 * GIB,
        workflow_budget_bytes=budget, allow_spatial_cache=True,
    )
    assert plan.particle_storage_policy == "streaming" and plan.batch_size == 1
    assert plan.fits_minimum and plan.estimated_peak_bytes == budget
    forced = plan_polar_storage(
        *args, free_bytes=8 * GIB, total_bytes=8 * GIB,
        allow_spatial_cache=True, force_streaming=True,
    )
    assert forced.particle_storage_policy == "streaming"
    with pytest.raises(ValueError, match="cache size"):
        plan_polar_storage(
            *args, free_bytes=8 * GIB, total_bytes=8 * GIB,
            allow_spatial_cache=True, spatial_cache_bytes=1,
        )


@pytest.mark.parametrize("fixed", [False, True])
def test_repeated_candidates_upload_source_once_but_refresh_references(
    emulated_runtime, monkeypatch, fixed
):
    images, references, centers, labels, config = validation.batch_fixture(17)
    config = replace(config, batch_size=7)
    particles = prepare_stack(images, config)
    priors = np.eye(3, dtype=np.float32)[labels] if fixed else np.full((17, 3), 1 / 3)
    records, uploads = [], []
    workspace = WorkflowGpuWorkspace(emulated_runtime, config, records)
    original = backend._asdevice

    def upload(values, **kwargs):
        uploads.append(id(values))
        return original(values, **kwargs)

    monkeypatch.setattr(backend, "_asdevice", upload)
    for iteration in range(3):
        # New references and centers each call; only the spatial source persists.
        refs = prepare_stack(np.roll(references, iteration, axis=1), config)
        current_centers = centers + iteration * 0.03125
        args = (particles, refs, config, priors, 0.08, None)
        actual = backend._gpu_candidate_inference(
            *args, translation_centers=current_centers, workspace=workspace,
            memory_records=records,
        )
        source_uploads = uploads.count(id(particles.spatial))
        expected = backend._gpu_polar_hard_candidate_inference_once(
            *args, translation_centers=current_centers, particle_limit=7,
            particle_storage_policy="streaming", engine="cupy",
        )
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
        assert source_uploads == 1
    state = workspace.summary()["roles"]["polar_spatial"]
    assert state["uploads"] == 1 and state["cache_hits"] == 2
    assert all(r["full_correlation_map_d2h_bytes"] == 0
               for r in records if r.get("event") == "complete")
    workspace.close()
    assert workspace.summary()["released_cache_bytes"] == particles.spatial.nbytes


@pytest.mark.parametrize("shared", [False, True])
def test_mstep_oom_evicts_spatial_before_fourier_and_retries_same_batch(monkeypatch, shared):
    from test_gpu_workspace import fake_cupy

    config = backend.AlignmentConfig(batch_size=7)
    workspace = WorkflowGpuWorkspace(fake_cupy(6 * GIB, 8 * GIB), config, [])
    spatial = np.zeros((3, 8, 8), np.float32)
    workspace.acquire_spatial(spatial, upload=lambda a: a.copy())
    calls = []

    def update(*args, **kwargs):
        calls.append(args[4].batch_size)
        if len(calls) == 1:
            raise MemoryError("out of memory in update")
        assert workspace.cache_bytes("polar_spatial") == 0
        return "updated"

    suffix = "_gpu_shared_reference_updater" if shared else "_gpu_reference_updater"
    monkeypatch.setattr(backend, suffix + "_once", update)
    monkeypatch.setattr(backend, "_memory_plan", lambda *a: SimpleNamespace(batch_size=7))
    monkeypatch.setattr(backend, "_cupy", lambda: SimpleNamespace(
        get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: None)
    ))
    kwargs = {"first_half": np.array([True, False, True])} if shared else {}
    result = getattr(backend, suffix)(spatial, {}, None, 1, config, workspace=workspace, **kwargs)
    assert result == "updated" and calls == [7, 7]
    assert not workspace.spatial_cache_allowed()


@pytest.mark.parametrize("fault", [None, "resident_upload", "mid_batch"])
def test_workflow_owns_spatial_cache_and_oom_falls_back_without_duplicate_update(
    emulated_runtime, monkeypatch, fault
):
    from contextlib import nullcontext
    from alignimg._engine import _update_references_fourier
    from tools import polar_231_recovery_validation as recovery

    images, references, _, labels, config = validation.batch_fixture(21)
    config = replace(config, translation_range=0.0, batch_size=7, profile_execution=False)
    steps = []
    # Real uploads own device storage; np.asarray may otherwise alias host data
    # and give the fault injector a weakref to the still-live PreparedStack.
    monkeypatch.setattr(
        backend, "_asdevice", lambda a, **kw: np.array(a, dtype=kw.get("dtype"), copy=True)
    )

    def update(*args, **kwargs):
        steps.append(1)
        return _update_references_fourier(
            *args, particle_fourier=kwargs.get("particle_fourier")
        )

    monkeypatch.setattr(backend, "_gpu_reference_updater", update)
    arguments = dict(
        config=config, class_priors=np.eye(3, dtype=np.float32)[labels],
        initial_poses=None, workflow="global", engine="cupy",
    )
    context = recovery.injected_oom(fault, images.shape) if fault else nullcontext()
    with context as faults:
        actual = backend._run_soft_alignment_gpu_engine(images, references, **arguments)
    state = actual.metadata["gpu_workspace"]
    spatial = state["roles"]["polar_spatial"]
    assert len(steps) == 3 and state["closed"]
    assert spatial["cache_hits"] == (2 if fault is None else 0)
    assert spatial["uploads"] == (0 if fault == "resident_upload" else 1)
    if fault:
        assert faults["released_before_replan"] and faults["failed_arrays_released"]
        assert faults["injected_tracebacks_detached"]
        assert faults["attempts"][0]["policy"] == "cached"
        assert faults["attempts"][1]["policy"] == "streaming"
        assert all(a["policy"] != "cached" for a in faults["attempts"][1:])
        expected = backend._run_soft_alignment_gpu_engine(images, references, **arguments)
        recovery.mirror.compare_workflows(expected, actual)
