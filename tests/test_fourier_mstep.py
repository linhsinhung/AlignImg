from __future__ import annotations

import numpy as np
import pytest

import alignimg as ai
from alignimg._engine import _update_references, _update_references_fourier
from alignimg._fourier import soft_circular_mask
from alignimg._transform import transform_image


def smooth_image(size: int = 32) -> np.ndarray:
    y, x = np.indices((size, size), dtype=np.float32)
    center = size / 2.0
    image = np.exp(-((y - (center - 5)) ** 2 + (x - (center + 4)) ** 2) / 10.0)
    image += 0.6 * np.exp(
        -((y - (center + 6)) ** 2 + (x - (center - 5)) ** 2) / 6.0
    )
    return image.astype(np.float32)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def test_reference_update_config_is_explicit_and_validated():
    assert ai.AlignmentConfig().reference_update == "fourier"
    assert (
        ai.AlignmentConfig(reference_update=" FOURIER ").normalized().reference_update
        == "fourier"
    )
    with pytest.raises(ValueError, match="reference_update"):
        ai.AlignmentConfig(reference_update="unknown").normalized()


def test_cpu_fourier_mstep_accumulates_weighted_particle_spectra():
    first = smooth_image()
    second = np.roll(first, 2, axis=0)
    images = np.stack((first, second))
    candidates = {
        "reference_index": np.zeros((2, 1), dtype=np.int32),
        "angle_deg": np.zeros((2, 1), dtype=np.float32),
        "shift_y_px": np.zeros((2, 1), dtype=np.float32),
        "shift_x_px": np.zeros((2, 1), dtype=np.float32),
        "mirror": np.zeros((2, 1), dtype=np.bool_),
        "posterior": np.ones((2, 1), dtype=np.float32),
    }
    inlier_weights = np.asarray([1.0, 0.25], dtype=np.float32)
    config = ai.AlignmentConfig(center_references=False)

    references, weights, shifts = _update_references_fourier(
        images,
        candidates,
        inlier_weights,
        1,
        config,
        particle_fourier=np.fft.fft2(images, axes=(-2, -1)),
    )

    mask = soft_circular_mask(len(first), config.mask_radius, config.mask_soft_edge)
    expected = (first + 0.25 * second) / 1.25 * mask
    assert np.allclose(references[0], expected, atol=2e-6)
    assert np.allclose(weights, [1.25])
    assert shifts == [(0.0, 0.0)]


def test_cpu_fourier_mstep_applies_reference_center_correction():
    image = np.roll(smooth_image(), (3, -2), axis=(0, 1))[None]
    candidates = {
        "reference_index": np.zeros((1, 1), dtype=np.int32),
        "angle_deg": np.zeros((1, 1), dtype=np.float32),
        "shift_y_px": np.zeros((1, 1), dtype=np.float32),
        "shift_x_px": np.zeros((1, 1), dtype=np.float32),
        "mirror": np.zeros((1, 1), dtype=np.bool_),
        "posterior": np.ones((1, 1), dtype=np.float32),
    }

    centered, _, shifts = _update_references_fourier(
        image,
        candidates,
        np.ones(1, dtype=np.float32),
        1,
        ai.AlignmentConfig(center_references=True),
        particle_fourier=np.fft.fft2(image, axes=(-2, -1)),
    )

    assert shifts[0] != (0.0, 0.0)
    assert np.all(np.isfinite(centered))


def test_cpu_fourier_mstep_conforms_for_rotation_translation_and_mirror():
    image = smooth_image(64)
    images = np.repeat(image[None], 4, axis=0)
    candidates = {
        "reference_index": np.asarray([[0], [0], [1], [1]], dtype=np.int32),
        "angle_deg": np.asarray([[0.0], [23.5], [-41.25], [90.0]], dtype=np.float32),
        "shift_y_px": np.asarray([[0.0], [1.25], [-2.0], [3.0]], dtype=np.float32),
        "shift_x_px": np.asarray([[0.0], [-0.75], [1.5], [-2.0]], dtype=np.float32),
        "mirror": np.asarray([[False], [False], [True], [True]]),
        "posterior": np.ones((4, 1), dtype=np.float32),
    }
    inlier_weights = np.asarray([1.0, 0.8, 0.6, 0.4], dtype=np.float32)
    config = ai.AlignmentConfig(center_references=False)

    spatial, spatial_weights, _ = _update_references(
        images, candidates, inlier_weights, 2, config
    )
    fourier, fourier_weights, _ = _update_references_fourier(
        images,
        candidates,
        inlier_weights,
        2,
        config,
        particle_fourier=np.fft.fft2(images, axes=(-2, -1)),
    )

    assert np.allclose(fourier_weights, spatial_weights)
    assert min(correlation(fourier[i], spatial[i]) for i in range(2)) > 0.99


def test_cpu_fourier_mstep_matches_spatial_workflow_and_keeps_diagnostics():
    reference = smooth_image(64)
    source_poses = ai.PoseSet(
        angle_deg=np.asarray([18.0, -27.0, 11.0, -8.0], dtype=np.float32),
        shift_y_px=np.asarray([1.0, -1.0, 0.0, 2.0], dtype=np.float32),
        shift_x_px=np.asarray([-1.0, 2.0, 1.0, 0.0], dtype=np.float32),
        mirror=np.zeros(4, dtype=np.bool_),
    )
    particles = np.stack(
        [
            transform_image(
                reference,
                angle_deg=float(source_poses.angle_deg[index]),
                shift_y_px=float(source_poses.shift_y_px[index]),
                shift_x_px=float(source_poses.shift_x_px[index]),
            )
            for index in range(len(source_poses))
        ]
    )
    common = dict(
        candidate_scoring="fourier",
        max_iterations=1,
        top_l=4,
        angle_samples=72,
        proposal_angles_per_reference=4,
        translation_range=2.0,
        robust_weighting=True,
        halfset_diagnostics=True,
        center_references=False,
    )
    spatial = ai.align_to_references(
        particles,
        reference,
        config=ai.AlignmentConfig(reference_update="spatial", **common),
        backend="cpu",
    )
    fourier = ai.align_to_references(
        particles,
        reference,
        config=ai.AlignmentConfig(reference_update="fourier", **common),
        backend="cpu",
    )

    assert fourier.metadata["reference_update"] == "fourier"
    assert "Fourier-domain soft M-step" in fourier.metadata["fft_policy"]
    assert np.array_equal(fourier.poses.angle_deg, spatial.poses.angle_deg)
    assert np.array_equal(fourier.poses.shift_y_px, spatial.poses.shift_y_px)
    assert np.array_equal(fourier.poses.shift_x_px, spatial.poses.shift_x_px)
    assert np.allclose(fourier.inlier_weights, spatial.inlier_weights)
    assert correlation(fourier.references[0], spatial.references[0]) > 0.99
    diagnostics = fourier.diagnostics[-1]
    assert np.all(np.isfinite(diagnostics["frc"]))
    assert np.allclose(
        diagnostics["halfset_effective_weight"],
        spatial.diagnostics[-1]["halfset_effective_weight"],
    )


@pytest.mark.parametrize("backend", ["cupy", "cuda"])
def test_gpu_fourier_mstep_matches_cpu_when_available(backend: str):
    if not ai.available_alignment_backends()[backend]["available"]:
        pytest.skip(f"{backend} backend is unavailable")
    reference = smooth_image(32)
    particles = np.stack(
        (
            transform_image(reference, angle_deg=12.0, shift_y_px=1.0),
            transform_image(reference, angle_deg=-18.0, shift_x_px=-1.0),
        )
    )
    config = ai.AlignmentConfig(
        candidate_scoring="fourier",
        reference_update="fourier",
        max_iterations=1,
        top_l=2,
        angle_samples=36,
        proposal_angles_per_reference=2,
        translation_range=1.0,
        halfset_diagnostics=False,
        center_references=False,
        batch_size=2,
    )

    cpu = ai.align_to_references(particles, reference, config=config, backend="cpu")
    gpu = ai.align_to_references(particles, reference, config=config, backend=backend)

    assert gpu.metadata["reference_update"] == "fourier"
    assert np.allclose(gpu.poses.angle_deg, cpu.poses.angle_deg, atol=1e-3)
    assert np.allclose(gpu.poses.shift_y_px, cpu.poses.shift_y_px, atol=1e-3)
    assert np.allclose(gpu.poses.shift_x_px, cpu.poses.shift_x_px, atol=1e-3)
    assert np.allclose(gpu.candidates.posterior, cpu.candidates.posterior, atol=2e-4)
    assert correlation(gpu.references[0], cpu.references[0]) > 0.9999
    assert any(
        item.get("stage") == "fourier_reference_update"
        for item in gpu.metadata["gpu_memory_plans"]
    )
