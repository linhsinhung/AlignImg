"""CPU-authoritative true polar hard candidate search."""

from __future__ import annotations

import numpy as np

from ._fourier import PreparedStack
from ._geometry import integer_center, mirror_x_integer_origin, validate_even_square
from .models import AlignmentConfig, PoseSet


def polar_radius(size: int, config: AlignmentConfig) -> float:
    """Return a shared safe radial support for translated spatial polar rings."""
    center = integer_center(size)
    boundary_limit = min(center, size - 1.0 - center)
    accumulated_margin = 2.0 * float(config.translation_range)
    default_radius = max(4.0, boundary_limit - accumulated_margin)
    configured = (
        default_radius if config.mask_radius is None else float(config.mask_radius)
    )
    return min(configured, default_radius, boundary_limit)


def translation_center_grid(
    center_y: float,
    center_x: float,
    *,
    size: int,
    radius: float,
    translation_range: float,
    translation_step: float,
) -> tuple[list[tuple[float, float]], int]:
    """Return y-major safe sampling centers and the number clipped at boundaries."""
    origin = integer_center(size)
    offsets = np.arange(
        -float(translation_range),
        float(translation_range) + 0.5 * float(translation_step),
        float(translation_step),
        dtype=np.float64,
    )
    all_centers = [
        (float(center_y + dy), float(center_x + dx)) for dy in offsets for dx in offsets
    ]
    safe = [
        (dy, dx)
        for dy, dx in all_centers
        if origin + dy - radius >= 0.0
        and origin + dy + radius <= size - 1.0
        and origin + dx - radius >= 0.0
        and origin + dx + radius <= size - 1.0
    ]
    if not safe:
        raise ValueError("polar_hard translation grid has no safe sampling centers.")
    return safe, len(all_centers) - len(safe)


def polar_offsets(
    radius: float, angle_samples: int, radial_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the platform-independent polar sampling offsets."""
    angles = np.arange(angle_samples, dtype=np.float64)[:, None] * (
        2.0 * np.pi / angle_samples
    )
    radii = np.arange(radial_bins, dtype=np.float64)[None] * (
        float(radius) / radial_bins
    )
    return (
        np.ascontiguousarray(np.sin(angles) * radii),
        np.ascontiguousarray(np.cos(angles) * radii),
    )


def sample_spatial_polar(
    image: np.ndarray,
    *,
    center_y: float,
    center_x: float,
    radius: float,
    angle_samples: int,
) -> np.ndarray:
    """Sample raw polar values with an explicit OpenCV-style bilinear table."""
    values = np.asarray(image, dtype=np.float32)
    size = validate_even_square(tuple(values.shape))
    radial_bins = max(4, int(np.floor(radius)))
    offset_y, offset_x = polar_offsets(radius, angle_samples, radial_bins)
    origin = integer_center(size)
    source_y = origin + float(center_y) + offset_y
    source_x = origin + float(center_x) + offset_x
    quantized_y = np.floor(source_y * 32.0 + 0.5).astype(np.int32)
    quantized_x = np.floor(source_x * 32.0 + 0.5).astype(np.int32)
    y0 = quantized_y >> 5
    x0 = quantized_x >> 5
    if np.any(y0 < 0) or np.any(x0 < 0) or np.any(y0 >= size) or np.any(x0 >= size):
        raise ValueError("polar sampling coordinates exceed the image boundary.")
    y1 = np.minimum(y0 + 1, size - 1)
    x1 = np.minimum(x0 + 1, size - 1)
    weight_y = ((quantized_y & 31) * np.float32(1.0 / 32.0)).astype(np.float32)
    weight_x = ((quantized_x & 31) * np.float32(1.0 / 32.0)).astype(np.float32)
    top = values[y0, x0] * (np.float32(1.0) - weight_x) + values[y0, x1] * weight_x
    bottom = values[y1, x0] * (np.float32(1.0) - weight_x) + values[y1, x1] * weight_x
    return (top * (np.float32(1.0) - weight_y) + bottom * weight_y).astype(np.float32)


def spatial_polar_rings(
    image: np.ndarray,
    *,
    center_y: float,
    center_x: float,
    radius: float,
    angle_samples: int,
) -> np.ndarray:
    """Sample and normalize weighted spatial polar rings with bilinear interpolation."""
    rings = sample_spatial_polar(
        image,
        center_y=center_y,
        center_x=center_x,
        radius=radius,
        angle_samples=angle_samples,
    )
    radial_bins = rings.shape[1]
    rings -= np.mean(rings, axis=0, keepdims=True)
    radial_weights = np.sqrt(np.arange(radial_bins, dtype=np.float32) + np.float32(0.5))
    rings *= radial_weights[None]
    rings /= np.float32(max(float(np.linalg.norm(rings)), 1e-12))
    return rings


def angular_correlation(
    subject_rings: np.ndarray,
    reference_rings: np.ndarray,
    *,
    subject_fft: np.ndarray | None = None,
    reference_fft: np.ndarray | None = None,
) -> np.ndarray:
    """Return the complete periodic polar angular correlation curve."""
    subject = np.asarray(subject_rings, dtype=np.float64)
    reference = np.asarray(reference_rings, dtype=np.float64)
    if subject.shape != reference.shape or subject.ndim != 2:
        raise ValueError(
            "subject and reference polar rings must have matching 2-D shapes."
        )
    if subject_fft is None:
        subject_fft = np.fft.rfft(subject, axis=0)
    if reference_fft is None:
        reference_fft = np.fft.rfft(reference, axis=0)
    cross_spectrum = np.sum(subject_fft * np.conj(reference_fft), axis=1)
    return np.fft.irfft(cross_spectrum, n=subject.shape[0])


def quadratic_peak_offset(
    previous: float, peak: float, following: float
) -> tuple[float, bool, str]:
    """Fit a periodic three-point concave peak, or deterministically fall back."""
    values = np.asarray([previous, peak, following], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        return 0.0, False, "non_finite"
    denominator = float(previous - 2.0 * peak + following)
    if denominator >= 0.0:
        return 0.0, False, "flat_or_convex"
    offset = 0.5 * float(previous - following) / denominator
    if not np.isfinite(offset):
        return 0.0, False, "non_finite"
    if abs(offset) > 0.5:
        return 0.0, False, "out_of_range"
    return offset, True, "accepted"


def translation_center_to_pose(
    angle_deg: float, center_y: float, center_x: float
) -> tuple[float, float]:
    """Convert a sampled source center into the canonical post-rotation pose shift."""
    radians = np.deg2rad(float(angle_deg))
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    shift_x = -(cosine * float(center_x) + sine * float(center_y))
    shift_y = sine * float(center_x) - cosine * float(center_y)
    return shift_y, shift_x


def infer_polar_hard_candidates_cpu(
    particles: PreparedStack,
    references: PreparedStack,
    config: AlignmentConfig,
    class_priors: np.ndarray,
    temperature: float,
    initial_poses: PoseSet | None,
    *,
    translation_centers: np.ndarray | None = None,
    rescue_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Search every angular bin and safe translation center, retaining hard top-1."""
    del initial_poses, rescue_mask
    particle_count = len(particles.spatial)
    if config.top_l != 1:
        raise ValueError("polar_hard requires top_l=1.")
    if translation_centers is None:
        centers = np.zeros((particle_count, 2), dtype=np.float64)
    else:
        centers = np.asarray(translation_centers, dtype=np.float64)
        if centers.shape != (particle_count, 2) or not np.all(np.isfinite(centers)):
            raise ValueError(
                "translation_centers must have finite shape (particle_count, 2)."
            )

    size = validate_even_square(tuple(particles.spatial.shape[1:]))
    radius = polar_radius(size, config)
    angle_samples = int(config.angle_samples)
    reference_rings = [
        spatial_polar_rings(
            reference,
            center_y=0.0,
            center_x=0.0,
            radius=radius,
            angle_samples=angle_samples,
        )
        for reference in references.spatial
    ]
    reference_ffts = [np.fft.rfft(rings, axis=0) for rings in reference_rings]

    shape = (particle_count, 1)
    result = {
        "reference_index": np.zeros(shape, dtype=np.int32),
        "angle_deg": np.zeros(shape, dtype=np.float32),
        "shift_y_px": np.zeros(shape, dtype=np.float32),
        "shift_x_px": np.zeros(shape, dtype=np.float32),
        "mirror": np.zeros(shape, dtype=np.bool_),
        "score": np.full(shape, -np.inf, dtype=np.float32),
        "posterior": np.ones(shape, dtype=np.float32),
        "_polar_center_y_px": np.zeros(shape, dtype=np.float32),
        "_polar_center_x_px": np.zeros(shape, dtype=np.float32),
        "_polar_raw_shift_y_px": np.zeros(shape, dtype=np.float32),
        "_polar_raw_shift_x_px": np.zeros(shape, dtype=np.float32),
        "_polar_discrete_angle_bin": np.zeros(shape, dtype=np.int32),
        "_polar_quadratic_fit_accepted": np.zeros(shape, dtype=np.bool_),
        "_polar_objective_margin": np.full(shape, np.inf, dtype=np.float32),
        "_polar_boundary_rejected_count": np.zeros(particle_count, dtype=np.int32),
        "_polar_evaluated_center_count": np.zeros(particle_count, dtype=np.int32),
    }
    mirror_values = (False, True) if config.mirror_search else (False,)
    mirror_count = len(mirror_values)
    for particle_index in range(particle_count):
        grid, boundary_rejected = translation_center_grid(
            centers[particle_index, 0],
            centers[particle_index, 1],
            size=size,
            radius=radius,
            translation_range=config.translation_range,
            translation_step=config.translation_step,
        )
        result["_polar_boundary_rejected_count"][particle_index] = boundary_rejected
        result["_polar_evaluated_center_count"][particle_index] = len(grid)
        active_references = np.flatnonzero(class_priors[particle_index] > 0.0)
        best: (
            tuple[float, int, float, int, float, float, float, bool, int, bool] | None
        ) = None
        second: (
            tuple[float, int, float, int, float, float, float, bool, int, bool] | None
        ) = None
        for mirror_index, mirrored in enumerate(mirror_values):
            image = (
                mirror_x_integer_origin(particles.spatial[particle_index])
                if mirrored
                else particles.spatial[particle_index]
            )
            for center_index, (center_y, center_x) in enumerate(grid):
                subject_rings = spatial_polar_rings(
                    image,
                    center_y=center_y,
                    center_x=center_x,
                    radius=radius,
                    angle_samples=angle_samples,
                )
                subject_fft = np.fft.rfft(subject_rings, axis=0)
                for reference_index in active_references:
                    curve = angular_correlation(
                        subject_rings,
                        reference_rings[int(reference_index)],
                        subject_fft=subject_fft,
                        reference_fft=reference_ffts[int(reference_index)],
                    )
                    peak_bin = int(np.argmax(curve))
                    offset, accepted, _ = quadratic_peak_offset(
                        curve[(peak_bin - 1) % angle_samples],
                        curve[peak_bin],
                        curve[(peak_bin + 1) % angle_samples],
                    )
                    angle = (peak_bin + offset) * 360.0 / angle_samples
                    angle = (angle + 180.0) % 360.0 - 180.0
                    score = float(curve[peak_bin])
                    if accepted:
                        score -= (
                            0.25
                            * float(
                                curve[(peak_bin - 1) % angle_samples]
                                - curve[(peak_bin + 1) % angle_samples]
                            )
                            * offset
                        )
                    objective = score / float(temperature) + np.log(
                        float(class_priors[particle_index, reference_index])
                    )
                    flat_id = (
                        (int(reference_index) * mirror_count + mirror_index) * len(grid)
                        + center_index
                    ) * angle_samples + peak_bin
                    candidate = (
                        objective,
                        flat_id,
                        score,
                        int(reference_index),
                        angle,
                        center_y,
                        center_x,
                        mirrored,
                        peak_bin,
                        accepted,
                    )
                    if (
                        best is None
                        or objective > best[0]
                        or (objective == best[0] and flat_id < best[1])
                    ):
                        second = best
                        best = candidate
                    elif (
                        second is None
                        or objective > second[0]
                        or (objective == second[0] and flat_id < second[1])
                    ):
                        second = candidate
        if best is None:
            raise ValueError(
                f"particle {particle_index} has no allowed polar_hard candidates."
            )
        (
            objective,
            _,
            score,
            reference_index,
            angle,
            center_y,
            center_x,
            mirrored,
            peak_bin,
            accepted,
        ) = best
        shift_y, shift_x = translation_center_to_pose(angle, center_y, center_x)
        result["reference_index"][particle_index, 0] = reference_index
        result["angle_deg"][particle_index, 0] = angle
        result["shift_y_px"][particle_index, 0] = shift_y
        result["shift_x_px"][particle_index, 0] = shift_x
        result["mirror"][particle_index, 0] = mirrored
        result["score"][particle_index, 0] = score
        result["_polar_center_y_px"][particle_index, 0] = center_y
        result["_polar_center_x_px"][particle_index, 0] = center_x
        result["_polar_raw_shift_y_px"][particle_index, 0] = (
            center_y - centers[particle_index, 0]
        )
        result["_polar_raw_shift_x_px"][particle_index, 0] = (
            center_x - centers[particle_index, 1]
        )
        result["_polar_discrete_angle_bin"][particle_index, 0] = peak_bin
        result["_polar_quadratic_fit_accepted"][particle_index, 0] = accepted
        if second is not None:
            result["_polar_objective_margin"][particle_index, 0] = objective - second[0]
    return result
