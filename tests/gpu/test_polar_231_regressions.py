"""Regression reproductions frozen before the 2.3.1 polar fixes."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from alignimg._geometry import mirror_x_integer_origin
from alignimg._polar_hard import sample_spatial_polar
from alignimg_gpu import backend


class _SpatialShape:
    def __init__(self, count, size):
        self.shape = (count, size, size)

    def __len__(self):
        return self.shape[0]


def test_polar_planner_reproduces_infeasible_minimum_batches(monkeypatch):
    # Preserve the T0/T1 reproduction against the immutable T2 runtime. T4's
    # current planner must stream these counts instead of rejecting them.
    from tools.polar_231_batch_validation import load_frozen_backend

    frozen = load_frozen_backend()
    gib = 1024**3
    fake_cupy = SimpleNamespace(
        get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: None),
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(memGetInfo=lambda: (8 * gib, 8 * gib))
        ),
    )
    monkeypatch.setattr(frozen, "_cupy", lambda: fake_cupy)
    config = backend.AlignmentConfig(batch_size=512, memory_fraction=0.8)

    evidence = {}
    for particles in (1_000, 105_000, 140_000):
        plan = frozen._polar_memory_plan(config, 128, particles, 10, 256, 52, 81, 1)
        minimum_bytes = plan.fixed_bytes + plan.bytes_per_item
        evidence[particles] = (plan, minimum_bytes)

    assert evidence[1_000][1] <= evidence[1_000][0].budget_bytes
    assert evidence[1_000][0].fits_minimum
    for particles in (105_000, 140_000):
        plan, minimum_bytes = evidence[particles]
        assert plan.batch_size == 1
        assert minimum_bytes > plan.budget_bytes
        assert not plan.fits_minimum


@pytest.mark.parametrize("count", [105_000, 140_000])
def test_infeasible_polar_budget_fails_before_upload_or_oom_retry(monkeypatch, count):
    gib = 1024**3
    calls = []
    fake_cupy = SimpleNamespace(
        get_default_memory_pool=lambda: SimpleNamespace(
            free_all_blocks=lambda: calls.append("release")
        ),
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(memGetInfo=lambda: (int(1.65 * gib), 8 * gib))
        ),
    )
    monkeypatch.setattr(backend, "_cupy", lambda: fake_cupy)
    monkeypatch.setattr(
        backend, "_asdevice", lambda *args, **kwargs: pytest.fail("particle upload")
    )
    monkeypatch.setattr(
        backend,
        "_gpu_polar_hard_candidate_inference_once",
        lambda *args, **kwargs: pytest.fail("infeasible solver entered"),
    )
    records = []
    with pytest.raises(MemoryError) as error:
        backend._gpu_candidate_inference(
            SimpleNamespace(spatial=_SpatialShape(count, 128)),
            SimpleNamespace(spatial=_SpatialShape(10, 128)),
            replace(backend.AlignmentConfig.preset("fast3"), batch_size=512),
            None,
            0.08,
            None,
            memory_records=records,
        )
    message = str(error.value)
    for field in (
        "budget_bytes=",
        "fixed_bytes=",
        "minimum_working_bytes=",
        "requested_batch=512",
    ):
        assert field in message
    assert calls == ["release"]
    assert len(records) == 1
    assert records[0]["fits_minimum"] is False


def test_feasible_polar_budget_preserves_batch_selection(monkeypatch):
    gib = 1024**3
    fake_cupy = SimpleNamespace(
        get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: None),
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(memGetInfo=lambda: (8 * gib, 8 * gib))
        ),
    )
    monkeypatch.setattr(backend, "_cupy", lambda: fake_cupy)
    batches = []

    def solver(*args, **kwargs):
        batches.append(kwargs["particle_limit"])
        return {"status": "reached"}

    monkeypatch.setattr(backend, "_gpu_polar_hard_candidate_inference_once", solver)
    config = replace(
        backend.AlignmentConfig.preset("fast3"), batch_size=512, angle_samples=256
    )
    expected = backend._polar_memory_plan(config, 128, 1000, 10, 256, 55, 81, 1)
    result = backend._gpu_candidate_inference(
        SimpleNamespace(spatial=_SpatialShape(1000, 128)),
        SimpleNamespace(spatial=_SpatialShape(10, 128)),
        config,
        None,
        0.08,
        None,
    )
    assert result == {"status": "reached"}
    assert batches == [expected.batch_size]


def test_fractional_mirror_quantization_order_reproduces_sampler_difference():
    size = 32
    image = gaussian_filter(
        np.random.default_rng(12).normal(size=(size, size)).astype(np.float32),
        sigma=1.0,
    ).astype(np.float32)
    radius = 8.0
    angle_samples = 64
    radial_bins = 8
    offset_y, offset_x = backend._polar_offsets(radius, angle_samples, radial_bins)

    center_y = 1.0 / 64.0
    center_x = -1.0 / 64.0
    cpu_expected = sample_spatial_polar(
        mirror_x_integer_origin(image),
        center_y=center_y,
        center_x=center_x,
        radius=radius,
        angle_samples=angle_samples,
    )
    gpu_emulated = _sample_gpu_arithmetic(image, center_y, center_x, offset_y, offset_x)

    assert np.count_nonzero(cpu_expected != gpu_emulated) == 89
    np.testing.assert_allclose(
        np.max(np.abs(cpu_expected - gpu_emulated)),
        0.0122323632,
        rtol=0.0,
        atol=1e-6,
    )


def _sample_gpu_arithmetic(image, center_y, center_x, offset_y, offset_x):
    size = image.shape[0]
    origin = size // 2
    source_y = origin + center_y + offset_y
    source_x = 2 * origin - (origin + center_x + offset_x)
    quantized_y = np.floor(source_y * 32.0 + 0.5).astype(np.int32)
    quantized_x = np.floor(source_x * 32.0 + 0.5).astype(np.int32)
    y0 = quantized_y >> 5
    x0 = quantized_x >> 5
    y1 = np.minimum(y0 + 1, size - 1)
    x1 = np.minimum(x0 + 1, size - 1)
    weight_y = ((quantized_y & 31) / 32.0).astype(np.float32)
    weight_x = ((quantized_x & 31) / 32.0).astype(np.float32)
    top = image[y0, x0] * (1.0 - weight_x) + image[y0, x1] * weight_x
    bottom = image[y1, x0] * (1.0 - weight_x) + image[y1, x1] * weight_x
    return (top * (1.0 - weight_y) + bottom * weight_y).astype(np.float32)
