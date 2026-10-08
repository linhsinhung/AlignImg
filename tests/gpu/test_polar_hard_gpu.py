from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import alignimg as ai
from alignimg._fourier import prepare_stack
from alignimg._polar_hard import (
    infer_polar_hard_candidates_cpu,
    sample_spatial_polar,
)
from alignimg._profiling import ExecutionProfile, _ACTIVE
from alignimg._transform import transform_image
from alignimg_gpu import backend


class FakeArray:
    next_pointer = 100

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape))
        self.dtype = dtype
        self.data = SimpleNamespace(ptr=FakeArray.next_pointer)
        FakeArray.next_pointer += 100

    def __len__(self):
        return self.shape[0]

    def astype(self, dtype):
        return FakeArray(self.shape, dtype)


def _sample_polar_numpy(
    images,
    image_indices,
    centers_y,
    centers_x,
    mirrors,
    offsets_y,
    offsets_x,
):
    values = np.asarray(images)
    indices = np.asarray(image_indices)
    center_y = np.asarray(centers_y)
    center_x = np.asarray(centers_x)
    mirror_values = np.asarray(mirrors)
    size = values.shape[1]
    origin = size // 2
    output = np.empty((len(indices), *offsets_y.shape), dtype=np.float32)
    for item, source_index in enumerate(indices):
        source_y = origin + center_y[item] + offsets_y
        source_x = origin + center_x[item] + offsets_x
        quantized_y = np.floor(source_y * 32.0 + 0.5).astype(np.int32)
        quantized_x = np.floor(source_x * 32.0 + 0.5).astype(np.int32)
        y0 = quantized_y >> 5
        x0 = quantized_x >> 5
        y1 = np.minimum(y0 + 1, size - 1)
        x1 = np.minimum(x0 + 1, size - 1)
        if mirror_values[item]:
            x0 = (-x0) % size
            x1 = (-x1) % size
        wy = ((quantized_y & 31) / 32.0).astype(np.float32)
        wx = ((quantized_x & 31) / 32.0).astype(np.float32)
        source = values[source_index]
        top = source[y0, x0] * (1.0 - wx) + source[y0, x1] * wx
        bottom = source[y1, x0] * (1.0 - wx) + source[y1, x1] * wx
        output[item] = (top * (1.0 - wy) + bottom * wy).astype(np.float32)
    return output


def test_device_sampler_matches_platform_independent_cpu_authority():
    size = 48
    angle_samples = 256
    radius = 16.0
    radial_bins = 16
    rng = np.random.default_rng(12)
    image = rng.normal(size=(size, size)).astype(np.float32)
    expected = sample_spatial_polar(
        image,
        center_y=0.0,
        center_x=0.0,
        radius=radius,
        angle_samples=angle_samples,
    )
    offset_y, offset_x = backend._polar_offsets(radius, angle_samples, radial_bins)
    actual = _sample_polar_numpy(
        image[None],
        np.asarray([0]),
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([False]),
        offset_y,
        offset_x,
    )[0]

    np.testing.assert_array_equal(actual, expected)


def test_device_algorithm_matches_cpu_authority_with_chunked_centers(monkeypatch):
    size = 32
    y, x = np.mgrid[:size, :size]
    first = (
        np.exp(-((y - 8) ** 2 + (x - 12) ** 2) / 6.0)
        + 0.7 * np.exp(-((y - 21) ** 2 + (x - 19) ** 2) / 5.0)
    ).astype(np.float32)
    second = np.roll(first, (3, -4), axis=(0, 1)).copy()
    references = np.stack((first, second))
    images = np.stack(
        (
            transform_image(
                first,
                angle_deg=31.2,
                shift_y_px=1.0,
                shift_x_px=-1.0,
                mirror=False,
            ),
            transform_image(
                second,
                angle_deg=-47.5,
                shift_y_px=-1.0,
                shift_x_px=0.0,
                mirror=True,
            ),
        )
    )
    config = ai.AlignmentConfig(
        search_strategy="polar_hard",
        candidate_scoring="polar",
        score_model="polar_ring_ccf",
        reference_update="fourier",
        top_l=1,
        max_iterations=1,
        angle_samples=64,
        translation_range=1.0,
        translation_step=1.0,
        mirror_search=True,
        robust_weighting=False,
        halfset_diagnostics=False,
        center_references=False,
    ).normalized(workflow="global")
    particles = prepare_stack(images, config)
    prepared_references = prepare_stack(references, config)
    priors = np.eye(2, dtype=np.float32)
    expected = infer_polar_hard_candidates_cpu(
        particles,
        prepared_references,
        config,
        priors,
        0.08,
        None,
    )
    monkeypatch.setattr(backend, "_cupy", lambda: np)
    monkeypatch.setattr(
        backend,
        "_asdevice",
        lambda value, **kwargs: np.asarray(value, dtype=kwargs.get("dtype")),
    )
    monkeypatch.setattr(backend, "_ashost", np.asarray)
    monkeypatch.setattr(backend, "_polar_sample_cupy", _sample_polar_numpy)
    counters = {}

    def count(name, amount=1):
        counters[name] = counters.get(name, 0) + amount

    monkeypatch.setattr(backend, "profile_count", count)

    actual = backend._gpu_polar_hard_candidate_inference_once(
        particles,
        prepared_references,
        config,
        priors,
        0.08,
        None,
        engine="cupy",
        particle_limit=5,
    )

    for name in (
        "reference_index",
        "mirror",
        "_polar_discrete_angle_bin",
        "_polar_quadratic_fit_accepted",
    ):
        np.testing.assert_array_equal(actual[name], expected[name])
    for name in (
        "angle_deg",
        "shift_y_px",
        "shift_x_px",
        "_polar_center_y_px",
        "_polar_center_x_px",
        "_polar_raw_shift_y_px",
        "_polar_raw_shift_x_px",
    ):
        np.testing.assert_allclose(
            actual[name], expected[name], atol=1e-6, err_msg=name
        )
    np.testing.assert_allclose(
        actual["score"], expected["score"], atol=2e-5, rtol=0.0, err_msg="score"
    )
    np.testing.assert_allclose(
        actual["_polar_objective_margin"],
        expected["_polar_objective_margin"],
        atol=1e-3,
        rtol=0.0,
        err_msg="_polar_objective_margin",
    )
    assert counters["polar_gpu_correlation_curves"] == 2 * 2 * 9


def test_native_polar_wrappers_pass_only_device_pointers(monkeypatch):
    calls = []

    class Session:
        def __init__(self, device, size, angle_samples, radial_bins):
            calls.append(("create", device, size, angle_samples, radial_bins))

        def sample_device(self, *args):
            calls.append(("sample", *args))

        def angular_peak_device(self, *args):
            calls.append(("peak", *args))

    fake_cp = SimpleNamespace(
        float32=np.float32,
        float64=np.float64,
        int32=np.int32,
        uint8=np.uint8,
        bool_=np.bool_,
        ascontiguousarray=lambda value, dtype: value,
        empty=lambda shape, dtype: FakeArray(
            (shape,) if isinstance(shape, int) else shape, dtype
        ),
        cuda=SimpleNamespace(
            Device=lambda: SimpleNamespace(id=0),
            get_current_stream=lambda: SimpleNamespace(ptr=999),
        ),
    )
    monkeypatch.setattr(backend, "_cupy", lambda: fake_cp)
    monkeypatch.setattr(
        backend,
        "_native_module",
        lambda: SimpleNamespace(PolarHardSession=Session),
    )
    backend._POLAR_NATIVE_SESSIONS.clear()
    images = FakeArray((3, 32, 32), np.float32)
    indices = FakeArray((2,), np.int32)
    center_y = FakeArray((2,), np.float64)
    center_x = FakeArray((2,), np.float64)
    mirrors = FakeArray((2,), np.uint8)
    offset_y = FakeArray((64, 8), np.float64)
    offset_x = FakeArray((64, 8), np.float64)

    profile = ExecutionProfile()
    token = _ACTIVE.set(profile)
    try:
        rings = backend._polar_sample_cuda(
            images,
            indices,
            center_y,
            center_x,
            mirrors,
            offset_y,
            offset_x,
        )
        curves = FakeArray((5, 64), np.float64)
        peak = backend._polar_peak_cuda(curves, 32, 8)
    finally:
        _ACTIVE.reset(token)
        backend._POLAR_NATIVE_SESSIONS.clear()

    assert rings.shape == (2, 64, 8)
    assert rings.dtype == np.float32
    assert peak[0].shape == (5,)
    assert calls[0] == ("create", 0, 32, 64, 8)
    sample = calls[1]
    assert sample[0] == "sample"
    assert sample[1:9] == (
        images.data.ptr,
        indices.data.ptr,
        center_y.data.ptr,
        center_x.data.ptr,
        mirrors.data.ptr,
        offset_y.data.ptr,
        offset_x.data.ptr,
        rings.data.ptr,
    )
    assert sample[-3:] == (3, 2, 999)
    peak_call = calls[2]
    assert peak_call[0] == "peak"
    assert peak_call[1] == curves.data.ptr
    assert peak_call[-2:] == (5, 999)
    assert profile.counters["native_polar_sampling_calls"] == 1
    assert profile.counters["native_polar_peak_calls"] == 1


def test_native_build_lists_dedicated_polar_translation_unit():
    cmake = Path("packages/alignimg-gpu/CMakeLists.txt").read_text()
    bindings = Path(
        "packages/alignimg-gpu/src/alignimg_gpu/native/bindings.cpp"
    ).read_text()
    native = Path(
        "packages/alignimg-gpu/src/alignimg_gpu/native/polar_hard_cuda.cu"
    ).read_text()

    assert "native/polar_hard_cuda.cu" in cmake
    assert "py::class_<alignimg_gpu::PolarHardSession>" in bindings
    assert "__fadd_rn" in backend._POLAR_SAMPLE_KERNEL
    assert "__fmul_rn" in backend._POLAR_SAMPLE_KERNEL
    assert "__fadd_rn" in native
    assert "__fmul_rn" in native
    assert "float* output" in backend._POLAR_SAMPLE_KERNEL
    assert "float* output" in native


def test_polar_memory_plan_accounts_for_ring_and_curve_tensors(monkeypatch):
    gib = 1024**3
    events = []
    fake_cp = SimpleNamespace(
        get_default_memory_pool=lambda: SimpleNamespace(
            free_all_blocks=lambda: events.append("release")
        ),
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(
                memGetInfo=lambda: (events.append("measure") or (20 * gib, 24 * gib))
            )
        ),
    )
    monkeypatch.setattr(backend, "_cupy", lambda: fake_cp)
    config = ai.AlignmentConfig(batch_size=512, memory_fraction=0.8)

    plan = backend._polar_memory_plan(config, 128, 384, 3, 256, 52, 81, 2)

    assert events == ["release", "measure"]
    assert plan.requested_batch_size == 512
    assert 1 <= plan.batch_size <= 384
    assert plan.particle_storage_policy == "streaming"
    assert plan.target_particle_batch == 384
    assert plan.bytes_per_item >= 81 * 2 * 256 * 52 * 18
    assert plan.fixed_bytes >= 3 * 256 * 52 * 12
    assert plan.fixed_bytes + plan.batch_size * plan.bytes_per_item <= plan.budget_bytes
