"""GPU policy tests that do not require a CUDA host."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import alignimg as ai
from alignimg_gpu import backend, memory


def test_requested_batch_is_a_cap_not_a_vram_override():
    gib = 1024**3
    plan = memory.plan_batch_size(
        free_bytes=20 * gib,
        total_bytes=24 * gib,
        memory_fraction=0.75,
        fixed_bytes=1 * gib,
        bytes_per_item=64 * 1024**2,
        requested_batch_size=256,
    )
    assert plan.reserve_bytes == 6 * gib
    assert plan.batch_size == 208
    assert plan.batch_size < plan.requested_batch_size


def test_low_free_memory_degrades_to_one_item_safely():
    mib = 1024**2
    plan = memory.plan_batch_size(
        free_bytes=300 * mib,
        total_bytes=24 * 1024**3,
        memory_fraction=0.75,
        fixed_bytes=100 * mib,
        bytes_per_item=32 * mib,
        requested_batch_size=None,
    )
    assert plan.batch_size == 1
    assert plan.budget_bytes == 0


def test_automatic_batch_has_soft_cap_but_explicit_batch_can_exceed_it():
    gib = 1024**3
    automatic = memory.plan_batch_size(
        free_bytes=20 * gib,
        total_bytes=24 * gib,
        memory_fraction=0.8,
        fixed_bytes=0,
        bytes_per_item=1024**2,
        requested_batch_size=None,
    )
    explicit = memory.plan_batch_size(
        free_bytes=20 * gib,
        total_bytes=24 * gib,
        memory_fraction=0.8,
        fixed_bytes=0,
        bytes_per_item=1024**2,
        requested_batch_size=1024,
    )
    assert automatic.batch_size == 256
    assert explicit.batch_size == 1024


def test_public_backend_registry_exposes_explicit_engines():
    backends = ai.available_alignment_backends()
    assert {"cpu", "cuda", "cupy", "gpu", "auto"} <= set(backends)
    assert backends["cpu"]["available"] is True
    assert backends["auto"]["available"] is True


def test_adaptive_gpu_entry_records_particle_batch_without_cuda(monkeypatch):
    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(memGetInfo=lambda: (8 * 1024**3, 24 * 1024**3))
        ),
        float32=np.float32,
        empty=np.empty,
        meshgrid=np.meshgrid,
    )
    monkeypatch.setattr(backend, "_cupy", lambda: fake_cupy)
    monkeypatch.setattr(
        backend,
        "_asdevice",
        lambda values, **kwargs: np.asarray(values, dtype=kwargs.get("dtype")),
    )
    monkeypatch.setattr(
        backend,
        "_acquire_particle_fourier",
        lambda *args, **kwargs: (
            None,
            SimpleNamespace(batch_size=512, asdict=lambda: {"batch_size": 512}),
        ),
    )
    captured = {}

    def fake_controller(*args, **kwargs):
        captured.update(kwargs)
        return {"status": "reached"}

    monkeypatch.setattr(backend, "infer_adaptive_candidates", fake_controller)
    particles = SimpleNamespace(
        spatial=np.zeros((3, 8, 8), dtype=np.float32),
        fourier=np.zeros((3, 8, 8), dtype=np.complex64),
        score_weight_profiles=np.ones((1, 5), dtype=np.float32),
        score_weight_bins=np.zeros((8, 8), dtype=np.int32),
        frequency_mask=np.ones((8, 8), dtype=np.float32),
    )
    references = SimpleNamespace(
        spatial=np.zeros((1, 8, 8), dtype=np.float32),
        fourier=np.zeros((1, 8, 8), dtype=np.complex64),
    )
    records = []

    result = backend._gpu_adaptive_candidate_inference_once(
        particles,
        references,
        ai.AlignmentConfig(search_strategy="adaptive_posterior"),
        np.ones((3, 1), dtype=np.float32),
        0.04,
        ai.PoseSet.identity(3),
        rescue_mask=None,
        memory_records=records,
    )

    assert result == {"status": "reached"}
    assert callable(captured["score_candidate_batches"])
    assert records[-1] == {
        "stage": "adaptive_particle_batching",
        "particle_batch_size": 3,
        "candidate_batch_size": 512,
    }
