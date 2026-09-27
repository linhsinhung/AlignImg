from types import SimpleNamespace

import numpy as np

import alignimg as ai
from alignimg._fourier import soft_circular_mask
from alignimg._profiling import ExecutionProfile, _ACTIVE
from alignimg_gpu import backend


def fake_cupy():
    return SimpleNamespace(
        add=np.add,
        asarray=np.asarray,
        float32=np.float32,
        float64=np.float64,
        int32=np.int32,
        zeros=np.zeros,
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(memGetInfo=lambda: (800_000_000, 1_000_000_000))
        ),
        get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: None),
    )


def install_numpy_device(monkeypatch):
    host_copies = []
    monkeypatch.setattr(backend, "_cupy", fake_cupy)
    monkeypatch.setattr(
        backend,
        "_asdevice",
        lambda values, dtype=None, order=None: np.asarray(values, dtype=dtype),
    )

    def ashost(values):
        host_copies.append(np.asarray(values).copy())
        return host_copies[-1]

    monkeypatch.setattr(backend, "_ashost", ashost)
    return host_copies


def test_gpu_raw_average_accumulates_fp64_and_downloads_only_k_images(monkeypatch):
    host_copies = install_numpy_device(monkeypatch)
    images = np.stack(
        [np.full((8, 8), value, dtype=np.float32) for value in (1, 3, 5, 7)]
    )
    poses = ai.PoseSet.identity(4)
    assignments = np.asarray([0, 1, 0, 1], dtype=np.int32)
    weights = np.asarray([1.0, 1.0, 3.0, 1.0], dtype=np.float64)
    references = np.stack(
        [np.zeros((8, 8), dtype=np.float32), np.ones((8, 8), dtype=np.float32)]
    )
    config = ai.AlignmentConfig(batch_size=2)
    profile = ExecutionProfile()
    token = _ACTIVE.set(profile)
    try:
        averages, metadata = backend._final_raw_class_averages_gpu_engine(
            images,
            poses,
            assignments,
            weights,
            references,
            config,
            transform_batch=lambda values, *_: np.asarray(values),
            engine="cupy",
        )
    finally:
        _ACTIVE.reset(token)

    expected = np.stack(
        [np.full((8, 8), 4.0), np.full((8, 8), 5.0)]
    ).astype(np.float32)
    expected *= soft_circular_mask(8, None, 4.0)[None]
    np.testing.assert_allclose(averages, expected)
    assert len(host_copies) == 1 and host_copies[0].shape == (2, 8, 8)
    assert metadata["class_average_transform_backend"] == "cupy"
    assert metadata["class_average_batch_size"] == 2
    assert metadata["class_average_empty_components"].size == 0
    assert profile.counters["final_raw_gpu_accumulation_batches"] == 2
    assert profile.counters["final_raw_gpu_accumulation_particles"] == 4
    assert profile.counters["final_raw_aligned_d2h_bytes_avoided"] == 4 * 8 * 8 * 4
    assert profile.counters["final_raw_output_d2h_bytes"] == 2 * 8 * 8 * 4


def test_gpu_raw_average_preserves_empty_reference(monkeypatch):
    install_numpy_device(monkeypatch)
    images = np.ones((2, 8, 8), dtype=np.float32)
    references = np.stack(
        [np.zeros((8, 8), dtype=np.float32), np.full((8, 8), 9.0, dtype=np.float32)]
    )
    averages, metadata = backend._final_raw_class_averages_gpu_engine(
        images,
        ai.PoseSet.identity(2),
        np.zeros(2, dtype=np.int32),
        np.ones(2, dtype=np.float64),
        references,
        ai.AlignmentConfig(batch_size=2),
        transform_batch=lambda values, *_: np.asarray(values),
        engine="cuda",
    )

    np.testing.assert_array_equal(averages[1], references[1])
    np.testing.assert_array_equal(metadata["class_average_empty_components"], [1])


def test_gpu_raw_average_retries_with_smaller_vram_bounded_batch(monkeypatch):
    install_numpy_device(monkeypatch)

    class OutOfMemoryError(RuntimeError):
        pass

    attempted = []

    def transform(values, *_):
        attempted.append(len(values))
        if len(values) > 2:
            raise OutOfMemoryError("out of memory")
        return np.asarray(values)

    averages, metadata = backend._final_raw_class_averages_gpu_engine(
        np.ones((4, 8, 8), dtype=np.float32),
        ai.PoseSet.identity(4),
        np.zeros(4, dtype=np.int32),
        np.ones(4, dtype=np.float64),
        np.zeros((1, 8, 8), dtype=np.float32),
        ai.AlignmentConfig(batch_size=4),
        transform_batch=transform,
        engine="cupy",
    )

    assert attempted == [4, 2, 2]
    assert averages.shape == (1, 8, 8)
    assert metadata["class_average_batch_size"] == 2
    assert metadata["class_average_gpu_memory_events"][-1] == {
        "stage": "final_raw_average",
        "event": "oom_retry",
        "batch_size": 2,
    }
