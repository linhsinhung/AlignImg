from types import SimpleNamespace

import numpy as np

from alignimg._profiling import ExecutionProfile, _ACTIVE
from alignimg_gpu import backend


def test_cached_fourier_prefers_indexed_transform_without_gather():
    cache = object()
    indices = np.array([2, 0], dtype=np.int32)
    calls = []

    def indexed(*args):
        calls.append(args)
        return "indexed"

    def standard(*args):
        raise AssertionError("cached CUDA path must not materialize a gather")

    result = backend._transform_cached_fourier(
        cache,
        np.zeros((3, 4, 4), dtype=np.complex64),
        indices,
        np.array([10, 20], dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.bool_),
        fourier_transform_batch=standard,
        indexed_fourier_transform_batch=indexed,
    )

    assert result == "indexed"
    assert calls[0][0] is cache
    assert calls[0][1] is indices


def test_cached_fourier_keeps_host_streaming_fallback():
    host = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4).astype(np.complex64)
    indices = np.array([2, 0], dtype=np.int32)

    def standard(source, *poses):
        np.testing.assert_array_equal(source, host[indices])
        return source

    result = backend._transform_cached_fourier(
        None,
        host,
        indices,
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.bool_),
        fourier_transform_batch=standard,
        indexed_fourier_transform_batch=lambda *args: None,
    )

    np.testing.assert_array_equal(result, host[indices])


def test_native_indexed_wrapper_passes_cache_and_index_device_pointers(monkeypatch):
    class FakeArray:
        next_pointer = 100

        def __init__(self, shape, dtype):
            self.shape = tuple(shape)
            self.ndim = len(self.shape)
            self.size = int(np.prod(self.shape))
            self.dtype = dtype
            self.data = SimpleNamespace(ptr=FakeArray.next_pointer)
            FakeArray.next_pointer += 100

    calls = []

    class Session:
        def __init__(self, device, size):
            calls.append(("create", device, size))

        def transform_fourier_indexed_device(self, *args):
            calls.append(("transform", *args))

    fake_cp = SimpleNamespace(
        complex64=np.complex64,
        int32=np.int32,
        empty=lambda shape, dtype: FakeArray(shape, dtype),
        cuda=SimpleNamespace(
            Device=lambda: SimpleNamespace(id=0),
            get_current_stream=lambda: SimpleNamespace(ptr=999),
        ),
    )
    source = FakeArray((3, 8, 8), np.complex64)
    indices = FakeArray((2,), np.int32)
    monkeypatch.setattr(backend, "_cupy", lambda: fake_cp)
    monkeypatch.setattr(
        backend, "_native_module", lambda: SimpleNamespace(TransformSession=Session)
    )
    monkeypatch.setattr(backend, "_asdevice", lambda value, **kwargs: value)
    backend._NATIVE_SESSIONS.clear()

    profile = ExecutionProfile()
    token = _ACTIVE.set(profile)
    try:
        output = backend._transform_fourier_indexed_batch_cuda(
            source,
            indices,
            np.array([10, 20], dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.bool_),
        )
    finally:
        _ACTIVE.reset(token)
        backend._NATIVE_SESSIONS.clear()

    assert output.shape == (2, 8, 8)
    transform = calls[1]
    assert transform[1:5] == (source.data.ptr, indices.data.ptr, output.data.ptr, 3)
    assert transform[-1] == 999
    assert profile.counters["native_indexed_fourier_transform_calls"] == 1
    assert profile.counters["native_indexed_fourier_transform_candidates"] == 2
    assert profile.counters["device_gather_bytes_avoided"] == 2 * 8 * 8 * 8
