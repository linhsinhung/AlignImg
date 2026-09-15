from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

import alignimg as ai
from alignimg._fourier import prepare_stack, score_weights_for_particle
from alignimg._transform import transform_image


def asymmetric_reference(size: int = 48) -> np.ndarray:
    y, x = np.indices((size, size), dtype=np.float32)
    image = np.exp(-((y - 13) ** 2 + (x - 31) ** 2) / 2.0)
    image += 0.8 * np.exp(-((y - 32) ** 2 + (x - 17) ** 2) / 3.0)
    image -= 0.5 * np.exp(-((y - 25) ** 2 + (x - 34) ** 2) / 1.5)
    return image.astype(np.float32)


def angle_error(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return np.abs((actual - expected + 180.0) % 360.0 - 180.0)


def test_score_model_is_explicit_and_validated():
    assert ai.AlignmentConfig().score_model == "fourier_ncc"
    assert (
        ai.AlignmentConfig(score_model=" WHITENED_FOURIER_NCC ")
        .normalized()
        .score_model
        == "whitened_fourier_ncc"
    )
    with pytest.raises(ValueError, match="score_model"):
        ai.AlignmentConfig(score_model="unknown").normalized()


def test_empirical_whitening_weights_are_radial_finite_and_normalized():
    rng = np.random.default_rng(7)
    images = np.stack(
        [gaussian_filter(rng.normal(size=(48, 48)), 2.5) for _ in range(12)]
    ).astype(np.float32)
    uniform = prepare_stack(images, ai.AlignmentConfig(score_model="fourier_ncc"))
    whitened = prepare_stack(
        images, ai.AlignmentConfig(score_model="whitened_fourier_ncc")
    )

    active = whitened.frequency_mask > 0.0
    uniform_weights = score_weights_for_particle(uniform, 0)
    whitened_weights = score_weights_for_particle(whitened, 0)
    assert np.array_equal(uniform_weights, uniform.frequency_mask)
    assert len(whitened.score_weight_profiles) == len(images)
    assert np.all(np.isfinite(whitened.score_weight_profiles))
    assert np.all(whitened_weights[~active] == 0.0)
    assert np.isclose(np.mean(whitened_weights[active]), 1.0, atol=1e-6)

    frequency = np.fft.fftfreq(48)
    fy, fx = np.meshgrid(frequency, frequency, indexing="ij")
    rings = np.rint(np.hypot(fy, fx) * 48).astype(np.int32)
    for ring in np.unique(rings[active]):
        values = whitened_weights[(rings == ring) & active]
        assert np.all(values == values[0])
    assert np.mean(whitened_weights[rings == 3]) < np.mean(
        whitened_weights[rings == 12]
    )


def test_whitened_scoring_improves_colored_noise_pose_ranking():
    rng = np.random.default_rng(5)
    reference = asymmetric_reference()
    source_angles = rng.choice(np.arange(-150, 181, 30), size=24).astype(np.float32)
    clean = np.stack(
        [transform_image(reference, angle_deg=float(angle)) for angle in source_angles]
    )
    colored_noise = np.stack(
        [gaussian_filter(rng.normal(size=reference.shape), 2.5) for _ in source_angles]
    ).astype(np.float32)
    colored_noise /= colored_noise.std(axis=(1, 2), keepdims=True)
    images = clean + colored_noise * clean.std() * 6.0
    common = dict(
        candidate_scoring="fourier",
        max_iterations=1,
        top_l=4,
        angle_samples=72,
        proposal_angles_per_reference=72,
        translation_range=0.0,
        halfset_diagnostics=False,
        center_references=False,
    )

    uniform = ai.align_to_references(
        images,
        reference,
        config=ai.AlignmentConfig(score_model="fourier_ncc", **common),
        backend="cpu",
    )
    whitened = ai.align_to_references(
        images,
        reference,
        config=ai.AlignmentConfig(score_model="whitened_fourier_ncc", **common),
        backend="cpu",
    )
    expected = (-source_angles + 180.0) % 360.0 - 180.0
    uniform_correct = int(np.sum(angle_error(uniform.poses.angle_deg, expected) <= 5.0))
    whitened_correct = int(
        np.sum(angle_error(whitened.poses.angle_deg, expected) <= 5.0)
    )

    assert uniform_correct <= 20
    assert whitened_correct == len(images)
    assert whitened.metadata["score_model"] == "whitened_fourier_ncc"


def test_per_particle_whitening_keeps_fixed_mra_equivalent_to_k1_runs():
    first = asymmetric_reference()
    second = np.roll(np.flip(first, axis=0), (3, -4), axis=(0, 1))
    references = np.stack((first, second))
    classes = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
    source_angles = np.asarray(
        [30.0, -45.0, 60.0, -90.0, 120.0, -135.0, 150.0, -180.0],
        dtype=np.float32,
    )
    particles = np.stack(
        [
            transform_image(references[classes[index]], angle_deg=float(angle))
            for index, angle in enumerate(source_angles)
        ]
    )
    config = ai.AlignmentConfig(
        candidate_scoring="fourier",
        score_model="whitened_fourier_ncc",
        reference_update="fourier",
        max_iterations=2,
        top_l=4,
        angle_samples=72,
        proposal_angles_per_reference=8,
        translation_range=1.0,
        halfset_diagnostics=False,
        center_references=False,
    )
    priors = ai.make_class_priors(
        assignments=classes, n_components=2, trust=1.0
    )
    joint = ai.align_to_references(
        particles,
        references,
        class_priors=priors,
        config=config,
        backend="cpu",
    )

    for component in range(2):
        selected = classes == component
        independent = ai.align_to_references(
            particles[selected],
            references[component],
            config=config,
            backend="cpu",
        )
        assert np.array_equal(
            joint.poses.angle_deg[selected], independent.poses.angle_deg
        )
        assert np.array_equal(
            joint.poses.shift_y_px[selected], independent.poses.shift_y_px
        )
        assert np.array_equal(
            joint.poses.shift_x_px[selected], independent.poses.shift_x_px
        )
        assert np.allclose(
            joint.references[component], independent.references[0], atol=1e-6
        )


@pytest.mark.parametrize("backend", ["cupy", "cuda"])
def test_gpu_whitened_workflow_matches_cpu_when_available(backend: str):
    if not ai.available_alignment_backends()[backend]["available"]:
        pytest.skip(f"{backend} backend is unavailable")
    reference = asymmetric_reference(48)
    particles = np.stack(
        (
            transform_image(reference, angle_deg=30.0),
            transform_image(reference, angle_deg=-45.0),
        )
    )
    config = ai.AlignmentConfig(
        candidate_scoring="fourier",
        score_model="whitened_fourier_ncc",
        max_iterations=1,
        top_l=4,
        angle_samples=72,
        proposal_angles_per_reference=8,
        translation_range=1.0,
        halfset_diagnostics=False,
        center_references=False,
        batch_size=2,
    )

    cpu = ai.align_to_references(particles, reference, config=config, backend="cpu")
    gpu = ai.align_to_references(particles, reference, config=config, backend=backend)

    assert gpu.metadata["score_model"] == "whitened_fourier_ncc"
    assert np.allclose(gpu.poses.angle_deg, cpu.poses.angle_deg, atol=1e-3)
    assert np.allclose(gpu.poses.shift_y_px, cpu.poses.shift_y_px, atol=1e-3)
    assert np.allclose(gpu.poses.shift_x_px, cpu.poses.shift_x_px, atol=1e-3)
    assert np.allclose(gpu.candidates.posterior, cpu.candidates.posterior, atol=2e-4)
