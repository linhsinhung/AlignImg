"""T2 authority checks and explicit, non-skipping server GPU cases."""

from __future__ import annotations

import numpy as np
import pytest

from alignimg._engine import _apply_reference_center_shifts, run_soft_alignment_cpu
from alignimg_gpu import backend
from tools import polar_231_mirror_validation as validation
from test_polar_hard_gpu import _sample_polar_numpy


def test_zero_center_shift_preserves_polar_sampling_grid_exactly():
    values = {
        "reference_index": np.array([[0], [1]], dtype=np.int32),
        "angle_deg": np.array([[-93.592102], [31.2]], dtype=np.float32),
        "shift_y_px": np.array([[-0.031188607], [0.25]], dtype=np.float32),
        "shift_x_px": np.array([[-0.001957929], [-0.5]], dtype=np.float32),
        "_polar_center_y_px": np.array([[0.0], [1 / 64]], dtype=np.float32),
        "_polar_center_x_px": np.array([[-1 / 32], [-3 / 64]], dtype=np.float32),
    }
    before = {name: value.copy() for name, value in values.items()}
    _apply_reference_center_shifts(values, [(0.0, 0.0), (0.0, 0.0)])
    for name in values:
        np.testing.assert_array_equal(values[name], before[name])
    _apply_reference_center_shifts(values, [(0.0, 0.0), (0.25, -0.125)])
    for name in values:
        np.testing.assert_array_equal(values[name][0], before[name][0])
    assert values["shift_y_px"][1, 0] == 0.5
    assert values["shift_x_px"][1, 0] == -0.625
    angle = np.deg2rad(float(values["angle_deg"][1, 0]))
    np.testing.assert_allclose(
        values["_polar_center_y_px"][1, 0],
        -np.sin(angle) * -0.625 - np.cos(angle) * 0.5,
        atol=1e-7,
        rtol=0,
    )


@pytest.fixture
def gpu_emulation(monkeypatch):
    monkeypatch.setattr(backend, "_cupy", lambda: np)
    monkeypatch.setattr(
        backend,
        "_asdevice",
        lambda value, **kw: np.asarray(value, dtype=kw.get("dtype")),
    )
    monkeypatch.setattr(backend, "_ashost", np.asarray)
    monkeypatch.setattr(backend, "_polar_sample_cupy", _sample_polar_numpy)


@pytest.mark.parametrize("radius", [4.0, 8.0])
def test_quantize_then_periodic_mirror_matches_unchanged_cpu(radius):
    image = validation.mirror_fixture()[1][0]
    centers = validation.sampler_centers()
    centers = centers[np.abs(centers).max(axis=1) <= (13 if radius == 4 else 1)]
    centers = np.repeat(centers, 2, axis=0)
    mirrors = np.tile([False, True], len(centers) // 2)
    offsets = backend._polar_offsets(radius, 64, int(radius))
    actual = _sample_polar_numpy(
        image[None],
        np.zeros(len(centers), dtype=np.int32),
        centers[:, 0],
        centers[:, 1],
        mirrors,
        *offsets,
    )
    expected = validation.sampler_expected(image, centers, mirrors, radius)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mirror", [False, True])
def test_fractional_candidate_matches_cpu(gpu_emulation, monkeypatch, mirror):
    def solver(*args, **kwargs):
        return backend._gpu_polar_hard_candidate_inference_once(
            *args, **kwargs, particle_limit=2
        )

    # Exercise the solver without a real device memory query.
    monkeypatch.setattr(backend, "_gpu_candidate_inference", solver)
    validation.validate_candidate("cupy", mirror)


@pytest.mark.parametrize("mirror", [False, True])
def test_fast3_fractional_centers_across_iterations(gpu_emulation, mirror):
    images, references, _ = validation.mirror_fixture()
    config = validation.mirror_config(mirror)
    centers_seen = []

    def solver(*args, **kwargs):
        centers = kwargs.get("translation_centers")
        if centers is not None:
            centers_seen.append(centers.copy())
        return backend._gpu_polar_hard_candidate_inference_once(
            *args,
            **kwargs,
            particle_limit=2,
            engine="cupy",
        )

    kwargs = dict(
        config=config, class_priors=None, initial_poses=None, workflow="global"
    )
    expected = run_soft_alignment_cpu(images, references, **kwargs)
    actual = run_soft_alignment_cpu(
        images, references, _candidate_inference=solver, **kwargs
    )
    validation.compare_workflows(expected, actual)
    assert len(centers_seen) == 2
    assert all(np.any(values != np.rint(values)) for values in centers_seen)


@pytest.mark.gpu
@pytest.mark.parametrize("engine", ["cupy", "cuda"])
def test_real_gpu_sampler_authority(engine):
    validation.runtime_identity(engine)
    validation.validate_sampler(engine)


@pytest.mark.gpu
@pytest.mark.parametrize("engine", ["cupy", "cuda"])
@pytest.mark.parametrize("mirror", [False, True])
def test_real_gpu_fractional_candidates(engine, mirror):
    validation.runtime_identity(engine)
    validation.validate_candidate(engine, mirror)


@pytest.mark.gpu
@pytest.mark.parametrize("engine", ["cupy", "cuda"])
@pytest.mark.parametrize("mirror", [False, True])
def test_real_gpu_fast3_mirror(engine, mirror):
    validation.runtime_identity(engine)
    validation.validate_fast3(engine, mirror)
