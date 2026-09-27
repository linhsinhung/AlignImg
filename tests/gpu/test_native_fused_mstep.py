from types import SimpleNamespace

import numpy as np

from alignimg._profiling import ExecutionProfile, _ACTIVE
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


def test_native_fused_fourier_accumulator_passes_device_pointers(monkeypatch):
    calls = []

    class Session:
        def __init__(self, device, size):
            calls.append(("create", device, size))

        def accumulate_fourier_indexed_device(self, *args):
            calls.append(("accumulate", *args))

    fake_cp = SimpleNamespace(
        complex64=np.complex64,
        float64=np.float64,
        int32=np.int32,
        cuda=SimpleNamespace(
            Device=lambda: SimpleNamespace(id=0),
            get_current_stream=lambda: SimpleNamespace(ptr=999),
        ),
    )
    source = FakeArray((3, 8, 8), np.complex64)
    particle_indices = FakeArray((4,), np.int32)
    sums_real = FakeArray((2, 8, 8), np.float64)
    sums_imag = FakeArray((2, 8, 8), np.float64)
    total_weights = FakeArray((2,), np.float64)
    monkeypatch.setattr(backend, "_cupy", lambda: fake_cp)
    monkeypatch.setattr(
        backend, "_native_module", lambda: SimpleNamespace(TransformSession=Session)
    )
    monkeypatch.setattr(backend, "_asdevice", lambda value, **kwargs: value)
    backend._NATIVE_SESSIONS.clear()

    profile = ExecutionProfile()
    token = _ACTIVE.set(profile)
    try:
        backend._accumulate_fourier_indexed_batch_cuda(
            source,
            particle_indices,
            np.asarray([0.0, 10.0, -20.0, 30.0], dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.bool_),
            np.asarray([0, 1, 0, 1], dtype=np.int32),
            np.asarray([1.0, 0.5, 0.25, 0.125], dtype=np.float32),
            sums_real,
            sums_imag,
            total_weights,
        )
    finally:
        _ACTIVE.reset(token)
        backend._NATIVE_SESSIONS.clear()

    call = calls[1]
    assert call[0] == "accumulate"
    assert call[1:7] == (
        source.data.ptr,
        particle_indices.data.ptr,
        sums_real.data.ptr,
        sums_imag.data.ptr,
        total_weights.data.ptr,
        3,
    )
    assert call[-1] == 999
    assert profile.counters["native_fused_fourier_accumulation_calls"] == 1
    assert profile.counters["native_fused_fourier_accumulation_candidates"] == 4
    assert profile.counters["fourier_transformed_batch_bytes_avoided"] == 4 * 8 * 8 * 8
    assert profile.counters["native_indexed_fourier_transform_candidates"] == 4


def test_native_fused_fourier_accumulator_rejects_bad_accumulator_ids(monkeypatch):
    fake_cp = SimpleNamespace(
        complex64=np.complex64,
        float64=np.float64,
        int32=np.int32,
    )
    monkeypatch.setattr(backend, "_cupy", lambda: fake_cp)
    monkeypatch.setattr(backend, "_native_module", lambda: object())
    monkeypatch.setattr(backend, "_asdevice", lambda value, **kwargs: value)
    source = FakeArray((2, 8, 8), np.complex64)
    sums_real = FakeArray((2, 8, 8), np.float64)
    sums_imag = FakeArray((2, 8, 8), np.float64)
    total_weights = FakeArray((2,), np.float64)

    with np.testing.assert_raises_regex(ValueError, "accumulator ids"):
        backend._accumulate_fourier_indexed_batch_cuda(
            source,
            FakeArray((1,), np.int32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.bool_),
            np.asarray([2], dtype=np.int32),
            np.ones(1, dtype=np.float32),
            sums_real,
            sums_imag,
            total_weights,
        )
