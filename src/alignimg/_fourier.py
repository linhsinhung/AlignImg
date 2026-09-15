"""Fourier/polar candidate generation and Cartesian Fourier scoring."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np

from ._geometry import integer_center, mirror_x_integer_origin, validate_even_square
from .models import AlignmentConfig, PoseSet
from ._transform import transform_image
from ._profiling import count as profile_count, profile_call, profile_stage
from ._frequency_tables import frequency_tables


@dataclass(frozen=True)
class PreparedStack:
    spatial: np.ndarray
    fourier: np.ndarray
    polar: np.ndarray
    mask: np.ndarray
    frequency_mask: np.ndarray
    score_weight_profiles: np.ndarray
    score_weight_bins: np.ndarray

    @cached_property
    def polar_fourier(self) -> np.ndarray:
        # Prepared stacks are replaced, rather than mutated, on reference update.
        profile_count("polar_angular_fft_calls", len(self.polar))
        result = np.fft.fft(self.polar, axis=1)
        result.flags.writeable = False
        profile_count("polar_fft_cache_bytes", result.nbytes)
        return result

    @cached_property
    def _reference_norm_cache(self) -> dict:
        # One weight profile only: O(K) norms even for particle-wise whitening.
        return {}


def reference_norms_for_particle(particles, references, particle_index):
    profile_index = 0 if len(particles.score_weight_profiles) == 1 else particle_index
    cache = references._reference_norm_cache
    profile_count("reference_norm_requests")
    if cache.get("particles") is particles and cache.get("profile_index") == profile_index:
        profile_count("reference_norm_cache_hits")
        return cache["norms"]
    weights = score_weights_for_particle(particles, particle_index)
    norms = np.asarray([
        float(np.sum(weights * np.abs(reference) ** 2))
        for reference in references.fourier
    ])
    profile_count("reference_norm_evaluations", len(norms))
    cache.update(particles=particles, profile_index=profile_index, norms=norms)
    return norms


def soft_circular_mask(size: int, radius: float | None, edge: float) -> np.ndarray:
    center = integer_center(size)
    y, x = np.indices((size, size), dtype=np.float32)
    distance = np.sqrt((y - center) ** 2 + (x - center) ** 2)
    outer = float(radius) if radius is not None else max(1.0, size / 2.0 - 2.0)
    edge = max(0.0, float(edge))
    if edge == 0:
        return (distance <= outer).astype(np.float32)
    inner = max(0.0, outer - edge)
    result = np.zeros((size, size), dtype=np.float32)
    result[distance <= inner] = 1.0
    transition = (distance > inner) & (distance < outer)
    result[transition] = 0.5 * (
        1.0 + np.cos(np.pi * (distance[transition] - inner) / max(edge, 1e-6))
    )
    return result


def _normalize_images(images: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(images, dtype=np.float32)
    count = max(float(mask.sum()), 1e-6)
    means = np.sum(values * mask[None], axis=(1, 2), keepdims=True) / count
    centered = (values - means) * mask[None]
    norms = np.sqrt(np.sum(centered * centered, axis=(1, 2), keepdims=True))
    return (centered / np.maximum(norms, 1e-8)).astype(np.float32)


@profile_stage("polar_descriptors")
def _polar_stack(fourier: np.ndarray, angle_samples: int) -> np.ndarray:
    count, size, _ = fourier.shape
    validate_even_square(tuple(fourier.shape[1:]))
    radial_bins = max(4, size // 2 - 2)
    center_value = integer_center(size)
    center = (center_value, center_value)
    polar = np.empty((count, angle_samples, radial_bins), dtype=np.float32)
    for index in range(count):
        magnitude = np.log1p(np.abs(np.fft.fftshift(fourier[index]))).astype(np.float32)
        rings = cv2.warpPolar(
            magnitude,
            (radial_bins, angle_samples),
            center,
            float(radial_bins),
            cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR,
        )
        rings -= np.mean(rings, axis=0, keepdims=True)
        norm = float(np.linalg.norm(rings))
        polar[index] = rings / max(norm, 1e-8)
    return polar


def _empirical_whitening_profiles(
    fourier: np.ndarray, frequency_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return one compact deterministic radial whitening profile per particle."""
    values = np.asarray(fourier, dtype=np.complex64)
    mask = np.asarray(frequency_mask, dtype=np.float32)
    size = int(values.shape[-1])
    fy, fx, *_ = frequency_tables(size)
    radial_bin = np.rint(np.hypot(fy, fx) * size).astype(np.int32)
    ring_count = int(radial_bin.max()) + 1
    ring_power = np.empty((len(values), ring_count), dtype=np.float64)
    for ring in range(ring_count):
        selected = radial_bin == ring
        ring_values = values[:, selected]
        ring_power[:, ring] = np.mean(
            ring_values.real.astype(np.float64) ** 2
            + ring_values.imag.astype(np.float64) ** 2,
            axis=1,
        )
    padded = np.pad(ring_power, ((0, 0), (1, 1)), mode="edge")
    smoothed_power = (
        0.25 * padded[:, :-2]
        + 0.5 * padded[:, 1:-1]
        + 0.25 * padded[:, 2:]
    )
    active_ring_count = np.bincount(
        radial_bin[mask > 0.0].ravel(), minlength=ring_count
    ).astype(np.float64)
    active_rings = active_ring_count > 0.0
    typical_power = np.median(smoothed_power[:, active_rings], axis=1)
    floor = np.maximum(typical_power * 1e-3, np.finfo(np.float64).tiny)
    profiles = 1.0 / np.maximum(smoothed_power, floor[:, None])
    active_mean = (
        profiles[:, active_rings] @ active_ring_count[active_rings]
    ) / np.sum(active_ring_count)
    profiles /= np.maximum(active_mean[:, None], np.finfo(np.float64).tiny)
    return profiles.astype(np.float32), radial_bin


def score_weights_for_particle(
    stack: PreparedStack, particle_index: int
) -> np.ndarray:
    """Expand one compact radial score profile to the Cartesian Fourier grid."""
    profile_index = 0 if len(stack.score_weight_profiles) == 1 else particle_index
    return (
        stack.frequency_mask
        * stack.score_weight_profiles[profile_index][stack.score_weight_bins]
    )


def prepare_stack(images: np.ndarray, config: AlignmentConfig) -> PreparedStack:
    values = np.asarray(images, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError("images must have non-empty shape (N, H, W).")
    validate_even_square(tuple(values.shape[1:]))
    if not np.all(np.isfinite(values)):
        raise ValueError("images must contain only finite values.")
    size = int(values.shape[1])
    mask = soft_circular_mask(size, config.mask_radius, config.mask_soft_edge)
    spatial = profile_call("normalize_images", _normalize_images, values, mask)
    fourier = profile_call("scoring_fft", np.fft.fft2, spatial, axes=(-2, -1)).astype(np.complex64)
    fy, fx, *_ = frequency_tables(size)
    radius = np.sqrt(fy * fy + fx * fx)
    frequency_mask = (
        (radius >= float(config.min_frequency))
        & (radius <= float(config.max_frequency))
        & (radius > 0.0)
    ).astype(np.float32)
    if config.score_model == "whitened_fourier_ncc":
        score_weight_profiles, score_weight_bins = _empirical_whitening_profiles(
            fourier, frequency_mask
        )
    else:
        score_weight_bins = np.rint(radius * size).astype(np.int32)
        score_weight_profiles = np.ones(
            (1, int(score_weight_bins.max()) + 1), dtype=np.float32
        )
    polar = _polar_stack(fourier, config.angle_samples)
    return PreparedStack(
        spatial,
        fourier,
        polar,
        mask,
        frequency_mask,
        score_weight_profiles,
        score_weight_bins,
    )


def _shift_fourier(fourier: np.ndarray, dy: float, dx: float) -> np.ndarray:
    size = fourier.shape[0]
    fy, fx, *_ = frequency_tables(size)
    phase = np.exp(-2j * np.pi * (fy * float(dy) + fx * float(dx)))
    return fourier * phase


def _fourier_ncc(a: np.ndarray, b: np.ndarray, weights: np.ndarray,
                 reference_norm: float | None = None) -> float:
    numerator = float(np.real(np.sum(weights * a * np.conj(b))))
    norm_a = float(np.sum(weights * np.abs(a) ** 2))
    norm_b = float(np.sum(weights * np.abs(b) ** 2)) if reference_norm is None else float(reference_norm)
    return numerator / max(np.sqrt(norm_a * norm_b), 1e-12)


def _angle_difference(a: float, b: float) -> float:
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


def _angles_per_reference(config: AlignmentConfig, reference_count: int) -> int:
    if config.proposal_angles_per_reference is not None:
        return int(config.proposal_angles_per_reference)
    return max(4, int(np.ceil(4 * config.top_l / reference_count)))


@profile_stage("polar_proposal")
def _candidate_angles(
    subject_polar: np.ndarray, reference_polar: np.ndarray, count: int,
    *, subject_fft=None, reference_fft=None,
) -> list[float]:
    if subject_fft is None:
        profile_count("polar_angular_fft_calls")
        subject_fft = np.fft.fft(subject_polar, axis=0)
    if reference_fft is None:
        profile_count("polar_angular_fft_calls")
        reference_fft = np.fft.fft(reference_polar, axis=0)
    correlation = np.real(
        np.fft.ifft(subject_fft * np.conj(reference_fft), axis=0)
    ).sum(axis=1)
    samples = len(correlation)
    count = min(max(1, int(count)), samples)
    # Fourier magnitude is centrosymmetric, so its angular correlation cannot
    # distinguish poses separated by 180 degrees. Retain both members of each
    # antipodal pair and use local maxima instead of spending the shortlist on
    # adjacent bins around a single broad peak.
    local_maxima = np.flatnonzero(
        (correlation >= np.roll(correlation, 1))
        & (correlation >= np.roll(correlation, -1))
    )
    ranked = sorted(
        local_maxima.tolist(),
        key=lambda index: (-float(correlation[index]), int(index)),
    )
    if not ranked:
        ranked = [int(np.argmax(correlation))]
    indices: list[int] = []
    half_turn = samples // 2
    for index in ranked:
        pair = (index, (index + half_turn) % samples) if samples % 2 == 0 else (index,)
        for candidate in pair:
            if candidate not in indices:
                indices.append(candidate)
            if len(indices) == count:
                break
        if len(indices) == count:
            break
    if len(indices) < count:
        fallback = sorted(
            range(samples), key=lambda index: (-float(correlation[index]), int(index))
        )
        indices.extend(index for index in fallback if index not in indices)
    # The correlation lag already maps the subject into reference coordinates.
    return [float(360.0 * index / samples) for index in indices[:count]]


def _translation_candidates(
    rotated: np.ndarray,
    reference: np.ndarray,
    config: AlignmentConfig,
    center_y: float,
    center_x: float,
    count: int,
    weights: np.ndarray | None = None,
) -> list[tuple[float, float]]:
    cross_power = rotated * np.conj(reference)
    if weights is not None:
        cross_power = cross_power * weights
    correlation = np.real(np.fft.ifft2(cross_power))
    radius = float(config.translation_range)
    step = float(config.translation_step)
    ys = np.arange(center_y - radius, center_y + radius + 0.5 * step, step)
    xs = np.arange(center_x - radius, center_x + radius + 0.5 * step, step)
    candidates: list[tuple[float, int, float, float]] = []
    size = correlation.shape[0]
    for dy in ys:
        for dx in xs:
            iy = int(round(-float(dy))) % size
            ix = int(round(-float(dx))) % size
            flat_id = iy * size + ix
            candidates.append(
                (float(correlation[iy, ix]), flat_id, float(dy), float(dx))
            )
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [(item[2], item[3]) for item in candidates[: max(1, count)]]


def infer_top_candidates(
    particles: PreparedStack,
    references: PreparedStack,
    config: AlignmentConfig,
    class_priors: np.ndarray,
    temperature: float,
    initial_poses: PoseSet | None,
) -> dict[str, np.ndarray]:
    """Infer top-L candidates and their normalized posterior probabilities."""
    particle_count = len(particles.spatial)
    reference_count = len(references.spatial)
    top_l = config.top_l
    shape = (particle_count, top_l)
    result = {
        "reference_index": np.zeros(shape, dtype=np.int32),
        "angle_deg": np.zeros(shape, dtype=np.float32),
        "shift_y_px": np.zeros(shape, dtype=np.float32),
        "shift_x_px": np.zeros(shape, dtype=np.float32),
        "mirror": np.zeros(shape, dtype=np.bool_),
        "score": np.full(shape, -np.inf, dtype=np.float32),
        "posterior": np.zeros(shape, dtype=np.float32),
    }
    for particle_index in range(particle_count):
        score_weights = score_weights_for_particle(particles, particle_index)
        reference_norms = reference_norms_for_particle(particles, references, particle_index)
        active_reference_count = int(
            np.count_nonzero(class_priors[particle_index] > 0.0)
        )
        angles_per_reference = _angles_per_reference(config, active_reference_count)
        translations_per_angle = max(1, int(np.ceil(top_l / active_reference_count)))
        all_candidates: list[
            tuple[float, float, int, float, float, float, bool, int]
        ] = []
        mirror_values = (False, True) if config.mirror_search else (False,)
        for mirrored in mirror_values:
            if mirrored:
                mirror_spatial = mirror_x_integer_origin(
                    particles.spatial[particle_index]
                )
                particle_fft = np.fft.fft2(mirror_spatial).astype(np.complex64)
                particle_polar = _polar_stack(particle_fft[None], config.angle_samples)[
                    0
                ]
            else:
                particle_fft = particles.fourier[particle_index]
                particle_polar = particles.polar[particle_index]
            polar_fft = None
            if initial_poses is None:
                if mirrored:
                    profile_count("polar_angular_fft_calls")
                    polar_fft = np.fft.fft(particle_polar, axis=0)
                else:
                    polar_fft = particles.polar_fourier[particle_index]
            for reference_index in range(reference_count):
                prior = float(class_priors[particle_index, reference_index])
                if prior <= 0:
                    continue
                if initial_poses is not None:
                    center_angle = float(initial_poses.angle_deg[particle_index])
                    angle_step = 360.0 / config.angle_samples
                    offsets = np.arange(angles_per_reference, dtype=np.float32)
                    offsets = (offsets - (angles_per_reference - 1) / 2.0) * angle_step
                    angles = [center_angle + float(offset) for offset in offsets]
                    center_y = float(initial_poses.shift_y_px[particle_index])
                    center_x = float(initial_poses.shift_x_px[particle_index])
                else:
                    angles = _candidate_angles(
                        particle_polar,
                        references.polar[reference_index],
                        angles_per_reference,
                        subject_fft=polar_fft,
                        reference_fft=references.polar_fourier[reference_index],
                    )
                    center_y = center_x = 0.0
                for angle in angles:
                    if config.candidate_scoring == "fourier":
                        from ._fourier_native import transform_fourier_cpu

                        rotated = transform_fourier_cpu(
                            particles.fourier[particle_index],
                            angle_deg=angle,
                            mirror=mirrored,
                        )
                    else:
                        rotated_spatial = transform_image(
                            particles.spatial[particle_index],
                            angle_deg=angle,
                            mirror=mirrored,
                        )
                        rotated = np.fft.fft2(rotated_spatial).astype(np.complex64)
                    translations = _translation_candidates(
                        rotated,
                        references.fourier[reference_index],
                        config,
                        center_y,
                        center_x,
                        translations_per_angle,
                        score_weights,
                    )
                    for dy, dx in translations:
                        transformed = _shift_fourier(rotated, dy, dx)
                        score = _fourier_ncc(
                            transformed,
                            references.fourier[reference_index],
                            score_weights,
                            reference_norms[reference_index],
                        )
                        log_posterior = score / temperature + np.log(max(prior, 1e-30))
                        if initial_poses is not None:
                            da = _angle_difference(
                                angle, initial_poses.angle_deg[particle_index]
                            )
                            ddy = dy - float(initial_poses.shift_y_px[particle_index])
                            ddx = dx - float(initial_poses.shift_x_px[particle_index])
                            log_posterior -= 0.5 * (da / config.pose_angle_sigma) ** 2
                            log_posterior -= 0.5 * (
                                (ddy / config.pose_shift_sigma) ** 2
                                + (ddx / config.pose_shift_sigma) ** 2
                            )
                        candidate_id = (
                            (reference_index * 2 + int(mirrored)) * config.angle_samples
                        ) + int(round((angle % 360.0) * config.angle_samples / 360.0))
                        all_candidates.append(
                            (
                                log_posterior,
                                score,
                                reference_index,
                                angle,
                                dy,
                                dx,
                                mirrored,
                                candidate_id,
                            )
                        )
        if not all_candidates:
            raise ValueError(
                f"particle {particle_index} has no allowed reference candidates."
            )
        all_candidates.sort(key=lambda item: (-item[0], item[7]))
        selected = all_candidates[:top_l]
        logits = np.asarray([item[0] for item in selected], dtype=np.float64)
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= np.sum(probabilities)
        for candidate_index, item in enumerate(selected):
            result["reference_index"][particle_index, candidate_index] = item[2]
            result["angle_deg"][particle_index, candidate_index] = (
                float(item[3]) + 180.0
            ) % 360.0 - 180.0
            result["shift_y_px"][particle_index, candidate_index] = item[4]
            result["shift_x_px"][particle_index, candidate_index] = item[5]
            result["mirror"][particle_index, candidate_index] = item[6]
            result["score"][particle_index, candidate_index] = item[1]
            result["posterior"][particle_index, candidate_index] = probabilities[
                candidate_index
            ]
    return result
