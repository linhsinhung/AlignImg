"""T3 same-arithmetic refactor and bounded resident batch scratch."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from alignimg._engine import run_soft_alignment_cpu
from alignimg._fourier import prepare_stack
from alignimg._polar_hard import infer_polar_hard_candidates_cpu
from alignimg_gpu import backend
from tools import polar_231_batch_validation as validation
from test_polar_hard_gpu import _sample_polar_numpy


@pytest.fixture
def emulated_backends(monkeypatch):
    frozen = validation.load_frozen_backend()
    for module in (backend, frozen):
        monkeypatch.setattr(module, "_cupy", lambda: np)
        monkeypatch.setattr(
            module,
            "_asdevice",
            lambda value, **kw: np.asarray(value, dtype=kw.get("dtype")),
        )
        monkeypatch.setattr(module, "_ashost", np.asarray)
        monkeypatch.setattr(module, "_polar_sample_cupy", _sample_polar_numpy)
    return frozen


@pytest.mark.parametrize("batch,count", [(1, 17), (7, 17), (256, 515), (512, 515)])
@pytest.mark.parametrize("fixed", [False, True])
def test_t2_exact_arithmetic_particle_order_and_tail(
    emulated_backends, monkeypatch, batch, count, fixed
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
    before_counts, after_counts = {}, {}

    def counter(values):
        def count(name, value=1):
            values[name] = values.get(name, 0) + value

        return count

    monkeypatch.setattr(emulated_backends, "profile_count", counter(before_counts))
    monkeypatch.setattr(backend, "profile_count", counter(after_counts))
    expected = emulated_backends._gpu_polar_hard_candidate_inference_once(
        *args, **kwargs
    )
    records = []
    actual = backend._gpu_polar_hard_candidate_inference_once(
        *args, **kwargs, memory_records=records
    )
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    assert after_counts == before_counts
    assert np.unique(actual["_polar_evaluated_center_count"]).size > 1
    if fixed:
        np.testing.assert_array_equal(actual["reference_index"][:, 0], labels)
    assert records[-1]["particle_storage_policy"] == "resident"
    assert records[-1]["maximum_winner_batch_size"] == min(batch, count)
    assert records[-1]["result_download_batches"] == (count + batch - 1) // batch


def test_only_batch_grids_winners_and_compact_results_are_on_device(
    emulated_backends, monkeypatch
):
    images, references, centers, labels, config = validation.batch_fixture(17)
    particles, refs = prepare_stack(images, config), prepare_stack(references, config)
    priors = np.eye(3, dtype=np.float32)[labels]
    uploads, downloads, winner_sizes, source_indices = [], [], [], []

    def upload(value, **kwargs):
        uploads.append(np.asarray(value).shape)
        return np.asarray(value, dtype=kwargs.get("dtype"))

    def download(value):
        downloads.append(value.shape)
        return np.asarray(value)

    def full(shape, *args, **kwargs):
        if isinstance(shape, int):
            winner_sizes.append(shape)
        return np.full(shape, *args, **kwargs)

    def solve(*args, **kwargs):
        source_indices.append(args[1].copy())
        return original_solver(*args, **kwargs)

    original_solver = backend._gpu_polar_batch_solver
    cp = SimpleNamespace(**{name: getattr(np, name) for name in dir(np)})
    cp.full = full
    monkeypatch.setattr(backend, "_cupy", lambda: cp)
    monkeypatch.setattr(backend, "_asdevice", upload)
    monkeypatch.setattr(backend, "_ashost", download)
    monkeypatch.setattr(backend, "_gpu_polar_batch_solver", solve)
    backend._gpu_polar_hard_candidate_inference_once(
        particles,
        refs,
        config,
        priors,
        0.08,
        None,
        translation_centers=centers,
        engine="cupy",
        particle_limit=7,
    )
    assert uploads.count(images.shape) == 1  # resident source upload, not per batch
    assert not any(shape and shape[0] == 17 for shape in uploads if len(shape) < 3)
    assert max(winner_sizes) == 7
    assert downloads == [(7, 11), (7, 11), (3, 11)]
    np.testing.assert_array_equal(np.concatenate(source_indices), np.arange(17))


def test_padded_center_ties_preserve_cpu_reference_mirror_center_order(
    emulated_backends,
):
    _, _, centers, _, config = validation.batch_fixture(7)
    # All candidates tie; particles have different legal center counts.
    particles = SimpleNamespace(spatial=np.zeros((7, 32, 32), dtype=np.float32))
    refs = SimpleNamespace(spatial=np.zeros((3, 32, 32), dtype=np.float32))
    priors = np.full((7, 3), 1 / 3, dtype=np.float32)
    args = (particles, refs, config, priors, 0.08, None)
    expected = infer_polar_hard_candidates_cpu(*args, translation_centers=centers)
    actual = backend._gpu_polar_hard_candidate_inference_once(
        *args, translation_centers=centers, engine="cupy", particle_limit=3
    )
    validation.compare_batch_candidates(expected, actual)


def test_fixed_mra_matches_per_class_k1_in_emulation(emulated_backends):
    images, references, _, labels, config = validation.batch_fixture(21)
    config = replace(config, translation_range=0.0)
    priors = np.eye(3, dtype=np.float32)[labels]

    def solve(*args, **kwargs):
        return backend._gpu_polar_hard_candidate_inference_once(
            *args, **kwargs, engine="cupy", particle_limit=7
        )

    def run(data, refs, priors=None):
        return run_soft_alignment_cpu(
            data,
            refs,
            config=config,
            class_priors=priors,
            initial_poses=None,
            workflow="global",
            _candidate_inference=solve,
        )

    joint = run(images, references, priors)
    np.testing.assert_array_equal(joint.reference_assignments, labels)
    for index in range(3):
        selected = labels == index
        single = run(images[selected], references[index][None])
        delta = (
            joint.poses.angle_deg[selected] - single.poses.angle_deg + 180
        ) % 360 - 180
        np.testing.assert_allclose(delta, 0, atol=1e-3, rtol=0)
        np.testing.assert_allclose(
            joint.poses.shift_y_px[selected], single.poses.shift_y_px, atol=1e-3, rtol=0
        )
        np.testing.assert_allclose(
            joint.poses.shift_x_px[selected], single.poses.shift_x_px, atol=1e-3, rtol=0
        )
        np.testing.assert_array_equal(joint.poses.mirror[selected], single.poses.mirror)
        assert (
            np.corrcoef(joint.references[index].ravel(), single.references[0].ravel())[
                0, 1
            ]
            >= 0.9999
        )


def test_fixed_workflow_frozen_solver_accepts_current_controller_in_emulation(
    emulated_backends, monkeypatch
):
    # Exercise the actual current controller -> frozen T2 function boundary,
    # which the isolated candidate and CPU workflow tests did not cover.
    monkeypatch.setattr(
        backend,
        "_polar_memory_plan",
        lambda *args, **kwargs: backend.plan_polar_storage(
            *args, free_bytes=8 * 1024**3, total_bytes=8 * 1024**3,
            force_streaming=kwargs.get("force_streaming", False),
            particle_limit=kwargs.get("particle_limit"),
        ),
    )
    gpu_backend = backend

    def align(images, references, *, config, backend: str, class_priors=None):
        def candidates(*args, **kwargs):
            return gpu_backend._gpu_candidate_inference(
                *args, **kwargs, engine=backend
            )

        result = run_soft_alignment_cpu(
            images,
            references,
            config=config,
            class_priors=class_priors,
            initial_poses=None,
            workflow="global",
            _candidate_inference=candidates,
            _backend_name=backend,
        )
        result.metadata["gpu_workspace"] = {"closed": True}
        return result

    monkeypatch.setattr(validation.ai, "align_to_references", align)
    result = validation.validate_fixed_workflow("cupy", emulated_backends)
    assert len(result["per_class"]) == 3


@pytest.mark.gpu
@pytest.mark.parametrize("engine", ["cupy", "cuda"])
def test_real_gpu_t3_batch_conformance(engine):
    validation.mirror.runtime_identity(engine)
    frozen = validation.load_frozen_backend()
    validation.validate_batches(engine, frozen)
    validation.validate_fixed_workflow(engine, frozen)
