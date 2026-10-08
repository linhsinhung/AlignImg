"""T4 memory-policy boundaries and shared-solver resident/streamed parity."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from alignimg._fourier import prepare_stack
from alignimg_gpu import backend
from alignimg_gpu._polar_memory import plan_polar_storage, polar_allocation_inventory
from alignimg_gpu._workspace import WorkflowGpuWorkspace
from tools import polar_231_batch_validation as validation
from test_polar_231_batch import emulated_backends  # noqa: F401 (pytest fixture)


GIB = 1024**3


def plan(count=1000, *, budget=None, cache=0, free=8 * GIB, config=None):
    if config is None:
        config = backend.AlignmentConfig(batch_size=512, memory_fraction=0.8)
    return plan_polar_storage(
        config,
        128,
        count,
        10,
        256,
        52,
        81,
        1,
        free_bytes=free,
        total_bytes=8 * GIB,
        workflow_budget_bytes=budget,
        live_cache_bytes=cache,
    )


def test_large_n_streaming_is_feasible_and_gpu_estimate_is_independent_of_n():
    plans = [plan(n) for n in (1000, 105_000, 140_000)]
    for current in plans:
        assert current.fits_minimum
        assert current.particle_storage_policy == "streaming"
        assert current.batch_size <= 512
        assert current.estimated_peak_bytes <= current.budget_bytes
        assert "resident_particles_fp32" not in current.fixed_components
        assert current.item_components["streamed_particle_images_fp32"] == 128**2 * 4
    assert all(p.fixed_components == plans[0].fixed_components for p in plans)
    assert all(p.item_components == plans[0].item_components for p in plans)
    assert len({p.batch_size for p in plans}) == 1


def test_resident_requires_target_batch_not_merely_minimum():
    config = replace(validation.mirror.mirror_config(), batch_size=7)
    args = (config, 32, 515, 3, 64, 8, 9, 2)
    roomy = plan_polar_storage(*args, free_bytes=8 * GIB, total_bytes=8 * GIB)
    assert roomy.particle_storage_policy == "resident"
    requirement = roomy.fixed_bytes + 7 * roomy.bytes_per_item
    exact = plan_polar_storage(
        *args,
        free_bytes=8 * GIB,
        total_bytes=8 * GIB,
        workflow_budget_bytes=requirement,
    )
    assert exact.particle_storage_policy == "resident" and exact.batch_size == 7
    small = plan_polar_storage(
        *args,
        free_bytes=8 * GIB,
        total_bytes=8 * GIB,
        workflow_budget_bytes=requirement - 1,
    )
    assert small.particle_storage_policy == "streaming" and small.batch_size == 7
    assert small.fits_minimum


def test_recovery_forces_streaming_and_caps_batch_without_changing_request():
    config = replace(validation.mirror.mirror_config(), batch_size=512)
    args = (config, 32, 515, 3, 64, 8, 9, 2)
    roomy = plan_polar_storage(*args, free_bytes=8 * GIB, total_bytes=8 * GIB)
    recovered = plan_polar_storage(
        *args, free_bytes=8 * GIB, total_bytes=8 * GIB,
        force_streaming=True, particle_limit=7,
    )
    assert roomy.particle_storage_policy == "resident"
    assert recovered.particle_storage_policy == "streaming"
    assert recovered.batch_size == 7
    assert recovered.requested_batch_size == recovered.target_particle_batch == 512
    assert "resident_particles_fp32" not in recovered.fixed_components
    assert recovered.estimated_peak_bytes <= recovered.budget_bytes


def test_recovery_batch_cap_still_obeys_current_budget():
    config = replace(validation.mirror.mirror_config(), batch_size=512)
    args = (config, 32, 515, 3, 64, 8, 9, 2)
    streamed = plan_polar_storage(
        *args, free_bytes=8 * GIB, total_bytes=8 * GIB, force_streaming=True,
    )
    recovered = plan_polar_storage(
        *args, free_bytes=8 * GIB, total_bytes=8 * GIB,
        force_streaming=True, particle_limit=7,
        workflow_budget_bytes=streamed.fixed_bytes + streamed.bytes_per_item,
    )
    assert recovered.batch_size == 1 and recovered.fits_minimum
    assert recovered.requested_batch_size == 512


@pytest.mark.parametrize("batch,expected", [(None, 256), (512, 512), (7, 7)])
def test_requested_upper_bound_and_automatic_soft_cap(batch, expected):
    config = replace(validation.mirror.mirror_config(), batch_size=batch)
    p = plan_polar_storage(
        config, 32, 515, 3, 64, 8, 1, 1, free_bytes=8 * GIB, total_bytes=8 * GIB
    )
    assert p.particle_storage_policy == "resident"
    assert p.target_particle_batch == p.batch_size == expected
    p = plan_polar_storage(
        config, 32, 3, 3, 64, 8, 1, 1, free_bytes=8 * GIB, total_bytes=8 * GIB
    )
    assert p.batch_size == p.target_particle_batch == 3


def test_live_cache_counted_once_and_external_usage_limits_current_budget():
    initial = plan()
    cached = plan(budget=initial.budget_bytes, cache=GIB, free=7 * GIB)
    assert cached.budget_bytes == initial.budget_bytes - GIB
    assert cached.live_cache_bytes == GIB
    external = plan(budget=initial.budget_bytes, cache=GIB, free=5 * GIB)
    assert external.budget_bytes == initial.budget_bytes - 3 * GIB
    released_external = plan(budget=2 * GIB, cache=GIB, free=8 * GIB)
    assert released_external.budget_bytes == GIB


def test_minimum_and_itemized_dtypes():
    p = plan()
    minimum = p.fixed_bytes + p.bytes_per_item
    exact = plan(budget=minimum)
    short = plan(budget=minimum - 1)
    assert exact.fits_minimum and exact.batch_size == 1
    assert short.batch_size == 1 and not short.fits_minimum
    fixed, item = polar_allocation_inventory(128, 10, 256, 52, 81, 2)
    assert item["peak_selector_copy_fp64"] == 81 * 2 * 10 * 256 * 8
    assert item["rings_fp32"] == 81 * 2 * 256 * 52 * 4
    assert item["subject_fft_complex64"] == 81 * 2 * 129 * 52 * 8
    assert fixed["polar_offsets_fp64"] == 2 * 256 * 52 * 8
    assert sum(p.fixed_components.values()) == p.fixed_bytes
    assert sum(p.item_components.values()) == p.bytes_per_item
    assert p.asdict()["estimated_peak_bytes"] == p.estimated_peak_bytes


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("batch,count", [(1, 17), (7, 17), (256, 515), (512, 515)])
@pytest.mark.usefixtures("emulated_backends")
def test_streaming_same_solver_outputs_and_bounded_uploads(
    monkeypatch, fixed, batch, count
):
    images, references, centers, labels, config = validation.batch_fixture(count)
    particles, refs = prepare_stack(images, config), prepare_stack(references, config)
    priors = (
        np.eye(3, dtype=np.float32)[labels]
        if fixed
        else np.full((count, 3), 1 / 3, dtype=np.float32)
    )
    args = (particles, refs, config, priors, 0.08, None)
    kwargs = dict(translation_centers=centers, engine="cupy", particle_limit=batch)
    resident = backend._gpu_polar_hard_candidate_inference_once(*args, **kwargs)
    uploaded, sources, downloads = [], [], []
    original = backend._gpu_polar_batch_solver

    def upload(values, **kw):
        result = np.asarray(values, dtype=kw.get("dtype"))
        uploaded.append(result.shape)
        return result

    def solve(images, indices, *args, **kwargs):
        sources.append((images.shape, indices.copy()))
        return original(images, indices, *args, **kwargs)

    def download(values):
        downloads.append(values.shape)
        return np.asarray(values)

    monkeypatch.setattr(backend, "_asdevice", upload)
    monkeypatch.setattr(backend, "_ashost", download)
    monkeypatch.setattr(backend, "_gpu_polar_batch_solver", solve)
    records = []
    streamed = backend._gpu_polar_hard_candidate_inference_once(
        *args, **kwargs, particle_storage_policy="streaming", memory_records=records
    )
    for name in resident:
        np.testing.assert_array_equal(resident[name], streamed[name], err_msg=name)
    assert images.shape not in uploaded
    assert max(shape[0] for shape in uploaded) <= max(batch, config.angle_samples)
    assert all(shape == (len(indices), 32, 32) for shape, indices in sources)
    for _, indices in sources:
        np.testing.assert_array_equal(indices, np.arange(len(indices)))
    assert sum(shape[0] for shape in downloads) == count
    assert all(shape[1] == 11 for shape in downloads)
    assert records[-1]["particle_storage_policy"] == "streaming"
    assert records[-1]["full_correlation_map_d2h_bytes"] == 0


@pytest.mark.parametrize("leave_after_one", [False, True])
def test_controller_evicts_update_then_scoring_and_remeasures(
    monkeypatch, leave_after_one
):
    config = replace(
        validation.mirror.mirror_config(), batch_size=7, translation_range=0
    )
    fixed, item = polar_allocation_inventory(32, 3, 64, 8, 1, 2)
    minimum = sum(fixed.values()) + sum(item.values()) + 32 * 32 * 4
    cache_values = np.zeros(max(1, minimum // 32), dtype=np.complex64)
    initial_budget = minimum + (cache_values.nbytes if leave_after_one else 0)
    total = 8 * GIB
    reserve = int(total * (1 - 0.8))
    workspace = None
    measurements = []

    def memory_info():
        cache = (
            0
            if workspace is None
            else workspace.allocation_budget()["live_cache_bytes"]
        )
        measurements.append(cache)
        return reserve + initial_budget - cache, total

    cp = SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=memory_info)),
        get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: None),
    )
    records = []
    workspace = WorkflowGpuWorkspace(cp, config, records)
    for role in ("update", "scoring"):
        workspace.acquire_fourier(
            role,
            cache_values,
            fixed_bytes=0,
            bytes_per_item=1,
            upload=lambda x: SimpleNamespace(nbytes=x.nbytes),
        )
    monkeypatch.setattr(backend, "_cupy", lambda: cp)
    solved = []

    def solve(*args, **kwargs):
        solved.append(kwargs)
        return {"done": True}

    monkeypatch.setattr(backend, "_gpu_polar_hard_candidate_inference_once", solve)
    particles = SimpleNamespace(spatial=np.zeros((17, 32, 32), np.float32))
    refs = SimpleNamespace(spatial=np.zeros((3, 32, 32), np.float32))
    result = backend._gpu_candidate_inference(
        particles,
        refs,
        config,
        None,
        0.08,
        None,
        workspace=workspace,
        memory_records=records,
    )
    assert result == {"done": True}
    assert len(solved) == 1 and solved[0]["particle_storage_policy"] == "streaming"
    assert solved[0]["particle_limit"] == 1
    evicted = [r["workspace_role"] for r in records if r.get("event") == "evict"]
    assert evicted == (["update"] if leave_after_one else ["update", "scoring"])
    assert len(measurements) == 2 + len(evicted)
    assert records[-1]["fits_minimum"]
    workspace.close()
    assert workspace.summary()["closed"]
    with pytest.raises(RuntimeError, match="closed"):
        workspace.allocation_budget()


@pytest.mark.gpu
@pytest.mark.parametrize("engine", ["cupy", "cuda"])
def test_real_gpu_t4_resident_streaming_and_resources(engine, tmp_path):
    from tools import polar_231_streaming_validation as streaming

    output = tmp_path / f"{engine}-streaming.json"
    exit_code = streaming.main(["--backend", engine, "--output", str(output)])
    report = json.loads(output.read_text())
    assert exit_code == 0, f"{report.get('error')}; full report: {output}"
