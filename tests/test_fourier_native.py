from __future__ import annotations

import numpy as np
import pytest

import alignimg as ai
from alignimg._adaptive import CELL_DTYPE
from alignimg._fourier import _fourier_ncc, _shift_fourier
from alignimg._fourier_native import (
    score_fourier_candidates_cpu,
    transform_fourier_cpu,
)
from alignimg._transform import transform_image


def smooth_image(size: int = 32) -> np.ndarray:
    y, x = np.indices((size, size), dtype=np.float32)
    center = size / 2.0
    image = np.exp(-((y - (center - 7)) ** 2 + (x - (center + 5)) ** 2) / 14.0)
    image += 0.7 * np.exp(-((y - (center + 8)) ** 2 + (x - (center - 6)) ** 2) / 7.0)
    image += 0.25 * np.exp(-((y - (center + 1)) ** 2 + (x - (center + 9)) ** 2) / 3.0)
    return image.astype(np.float32)


def frequency_mask(size: int, maximum: float = 0.35) -> np.ndarray:
    frequency = np.fft.fftfreq(size)
    fy, fx = np.meshgrid(frequency, frequency, indexing="ij")
    radius = np.hypot(fy, fx)
    return ((radius > 0.0) & (radius <= maximum)).astype(np.float32)


def test_fourier_native_identity_and_translation_are_exact():
    source = np.fft.fft2(smooth_image()).astype(np.complex64)

    identity = transform_fourier_cpu(source)
    shifted = transform_fourier_cpu(source, shift_y_px=1.5, shift_x_px=-2.0)

    assert np.allclose(identity, source, atol=1e-6)
    assert np.allclose(shifted, _shift_fourier(source, 1.5, -2.0), atol=1e-6)


def test_fourier_native_mirror_matches_discrete_integer_origin_mirror():
    image = smooth_image()
    source = np.fft.fft2(image).astype(np.complex64)
    expected = np.fft.fft2(transform_image(image, mirror=True)).astype(np.complex64)

    actual = transform_fourier_cpu(source, mirror=True)

    assert np.allclose(actual, expected, atol=2e-5)


def test_fourier_native_rotation_conforms_in_scoring_band():
    image = smooth_image(64)
    source = np.fft.fft2(image).astype(np.complex64)
    mask = frequency_mask(64)
    expected = np.fft.fft2(
        transform_image(image, angle_deg=23.5, shift_y_px=1.25, shift_x_px=-0.75)
    ).astype(np.complex64)
    actual = transform_fourier_cpu(
        source,
        angle_deg=23.5,
        shift_y_px=1.25,
        shift_x_px=-0.75,
    )

    assert _fourier_ncc(actual, expected, mask) > 0.98


def test_fourier_native_candidate_scorer_ranks_the_correct_angle():
    reference = smooth_image()
    particle = transform_image(reference, angle_deg=27.0)
    cells = np.asarray(
        [
            (0, -27.0, 0.0, 0.0, False),
            (0, -9.0, 0.0, 0.0, False),
            (0, 42.0, 0.0, 0.0, False),
        ],
        dtype=CELL_DTYPE,
    )

    scores = score_fourier_candidates_cpu(
        np.fft.fft2(particle),
        np.fft.fft2(reference)[None],
        cells,
        frequency_mask(len(reference)),
    )

    assert int(np.argmax(scores)) == 0
    assert scores[0] > scores[1] + 0.1


def test_candidate_scoring_config_is_explicit_and_validated():
    assert ai.AlignmentConfig().candidate_scoring == "fourier"
    assert (
        ai.AlignmentConfig(candidate_scoring=" FOURIER ").normalized().candidate_scoring
        == "fourier"
    )
    with pytest.raises(ValueError, match="candidate_scoring"):
        ai.AlignmentConfig(candidate_scoring="unknown").normalized()


@pytest.mark.parametrize("search_strategy", ["proposal", "adaptive_posterior"])
def test_cpu_workflow_can_select_fourier_native_candidate_scoring(
    search_strategy: str,
):
    reference = smooth_image(64)
    source_angles = np.asarray([18.0, -31.0], dtype=np.float32)
    particles = np.stack(
        [transform_image(reference, angle_deg=float(angle)) for angle in source_angles]
    )
    initial_poses = None
    if search_strategy == "adaptive_posterior":
        initial_poses = ai.PoseSet(
            -source_angles + np.asarray([3.0, -3.0], dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.bool_),
        )
    common = dict(
        max_iterations=1,
        top_l=4,
        angle_samples=72,
        proposal_angles_per_reference=4,
        translation_range=2.0,
        search_strategy=search_strategy,
        coarse_angle_step=3.0,
        local_angle_range=6.0,
        local_shift_range=1.0,
        adaptive_fraction=0.95,
        oversampling_order=0,
        halfset_diagnostics=False,
        center_references=False,
    )
    raster_config = ai.AlignmentConfig(candidate_scoring="raster", **common)
    fourier_config = ai.AlignmentConfig(candidate_scoring="fourier", **common)
    if initial_poses is None:
        raster = ai.align_to_references(
            particles, reference, config=raster_config, backend="cpu"
        )
        fourier = ai.align_to_references(
            particles, reference, config=fourier_config, backend="cpu"
        )
    else:
        raster = ai.refine_alignment(
            particles, reference, initial_poses, config=raster_config, backend="cpu"
        )
        fourier = ai.refine_alignment(
            particles, reference, initial_poses, config=fourier_config, backend="cpu"
        )

    assert fourier.metadata["candidate_scoring"] == "fourier"
    assert "Fourier-native" in fourier.metadata["fft_policy"]
    assert np.allclose(fourier.poses.angle_deg, raster.poses.angle_deg, atol=3.0)
    assert np.allclose(fourier.poses.shift_y_px, raster.poses.shift_y_px, atol=1.0)
    assert np.allclose(fourier.poses.shift_x_px, raster.poses.shift_x_px, atol=1.0)
    assert (
        np.corrcoef(fourier.references[0].ravel(), raster.references[0].ravel())[0, 1]
        > 0.99
    )


@pytest.mark.parametrize("backend", ["cupy", "cuda"])
def test_gpu_fourier_native_transform_matches_cpu_when_available(backend: str):
    if not ai.available_alignment_backends()[backend]["available"]:
        pytest.skip(f"{backend} backend is unavailable")
    from alignimg_gpu.backend import transform_fourier_cuda, transform_fourier_cupy

    image = smooth_image(64)
    fourier = np.repeat(np.fft.fft2(image)[None], 4, axis=0).astype(np.complex64)
    poses = ai.PoseSet(
        angle_deg=np.asarray([0.0, 23.5, -41.25, 90.0], dtype=np.float32),
        shift_y_px=np.asarray([0.0, 1.25, -2.0, 3.0], dtype=np.float32),
        shift_x_px=np.asarray([0.0, -0.75, 1.5, -2.0], dtype=np.float32),
        mirror=np.asarray([False, False, True, True]),
    )
    expected = np.stack(
        [
            transform_fourier_cpu(
                fourier[index],
                angle_deg=poses.angle_deg[index],
                shift_y_px=poses.shift_y_px[index],
                shift_x_px=poses.shift_x_px[index],
                mirror=poses.mirror[index],
            )
            for index in range(len(poses))
        ]
    )
    transform = transform_fourier_cuda if backend == "cuda" else transform_fourier_cupy

    actual = transform(fourier, poses)

    assert np.allclose(actual, expected, rtol=3e-5, atol=3e-4)


@pytest.mark.parametrize("backend", ["cupy", "cuda"])
def test_gpu_fourier_native_scores_match_cpu_when_available(backend: str):
    if not ai.available_alignment_backends()[backend]["available"]:
        pytest.skip(f"{backend} backend is unavailable")
    from alignimg_gpu.backend import (
        score_fourier_candidates_cuda,
        score_fourier_candidates_cupy,
    )

    reference = smooth_image(64)
    particle = transform_image(reference, angle_deg=19.0, shift_y_px=1.0)
    references = np.stack([reference, np.roll(reference, 5, axis=0)])
    cells = np.asarray(
        [
            (0, -19.0, -1.0, 0.0, False),
            (0, -7.0, -1.0, 0.0, False),
            (1, -19.0, -1.0, 0.0, False),
            (0, 31.0, 2.0, -1.5, True),
        ],
        dtype=CELL_DTYPE,
    )
    particle_fourier = np.fft.fft2(particle).astype(np.complex64)
    reference_fourier = np.fft.fft2(references, axes=(-2, -1)).astype(np.complex64)
    mask = frequency_mask(64)
    expected = score_fourier_candidates_cpu(
        particle_fourier,
        reference_fourier,
        cells,
        mask,
    )
    scorer = (
        score_fourier_candidates_cuda
        if backend == "cuda"
        else score_fourier_candidates_cupy
    )

    actual = scorer(
        particle_fourier,
        reference_fourier,
        cells,
        mask,
        ai.AlignmentConfig(batch_size=2),
    )

    assert np.allclose(actual, expected, atol=2e-5)


@pytest.mark.parametrize("backend", ["cupy", "cuda"])
def test_gpu_workflow_uses_fourier_native_candidate_scoring_when_available(
    backend: str,
):
    if not ai.available_alignment_backends()[backend]["available"]:
        pytest.skip(f"{backend} backend is unavailable")
    reference = smooth_image(64)
    particles = np.stack(
        [
            transform_image(reference, angle_deg=18.0),
            transform_image(reference, angle_deg=-31.0),
        ]
    )
    config = ai.AlignmentConfig(
        candidate_scoring="fourier",
        max_iterations=1,
        top_l=4,
        angle_samples=72,
        proposal_angles_per_reference=4,
        translation_range=2.0,
        halfset_diagnostics=False,
        center_references=False,
        batch_size=2,
    )

    cpu = ai.align_to_references(particles, reference, config=config, backend="cpu")
    gpu = ai.align_to_references(particles, reference, config=config, backend=backend)

    assert gpu.metadata["candidate_scoring"] == "fourier"
    assert "Fourier-native" in gpu.metadata["fft_policy"]
    assert np.allclose(gpu.poses.angle_deg, cpu.poses.angle_deg, atol=1e-3)
    assert np.allclose(gpu.poses.shift_y_px, cpu.poses.shift_y_px, atol=1e-3)
    assert np.allclose(gpu.poses.shift_x_px, cpu.poses.shift_x_px, atol=1e-3)
    assert np.allclose(gpu.candidates.posterior, cpu.candidates.posterior, atol=2e-4)
