"""CPU-authoritative AlignImg 2.x soft alignment engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any

import cv2
import numpy as np

from ._adaptive import STANDARD_CANDIDATE_FIELDS, infer_adaptive_candidates_cpu
from ._quadratic import infer_quadratic_candidates_cpu
from ._geometry import CENTER_CONVENTION, integer_center, validate_even_square
from ._fourier import (
    _candidate_angles,
    infer_top_candidates,
    prepare_stack,
    soft_circular_mask,
)
from ._transform import transform_image
from ._profiling import current_profile, profile_call, profile_stage
from .models import AlignmentConfig, AlignmentResult, CandidateSet, PoseSet


@dataclass
class _SharedReferenceUpdate:
    references: np.ndarray
    effective_weights: np.ndarray
    center_shifts: list[tuple[float, float]]
    half_references: np.ndarray
    half_weights: np.ndarray
    halfset_finalize_seconds: float


def validate_class_priors(
    class_priors: np.ndarray | None,
    particle_count: int,
    reference_count: int,
) -> np.ndarray:
    if class_priors is None:
        return np.full(
            (particle_count, reference_count),
            1.0 / reference_count,
            dtype=np.float32,
        )
    values = np.asarray(class_priors, dtype=np.float32)
    if values.shape != (particle_count, reference_count):
        raise ValueError(
            f"class_priors must have shape {(particle_count, reference_count)}, got {values.shape}."
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("class_priors must contain finite non-negative values.")
    row_sums = values.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError(
            "every class_priors row must contain at least one positive value."
        )
    return (values / row_sums).astype(np.float32)


def _temperature(config: AlignmentConfig, iteration: int) -> float:
    anneal_iterations = (
        config.max_iterations
        if config.temperature_anneal_iterations is None
        else config.temperature_anneal_iterations
    )
    if anneal_iterations <= 1:
        return float(config.temperature_end)
    fraction = min(iteration / (anneal_iterations - 1), 1.0)
    return float(
        config.temperature_start
        * (config.temperature_end / config.temperature_start) ** fraction
    )


def _inlier_weights(
    scores: np.ndarray,
    config: AlignmentConfig,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    if not config.robust_weighting:
        return np.ones(len(scores), dtype=np.float32)
    output = np.empty(len(scores), dtype=np.float32)
    group_values = (
        np.zeros(len(scores), dtype=np.int32)
        if groups is None
        else np.asarray(groups, dtype=np.int32)
    )
    for group in np.unique(group_values):
        selected = group_values == group
        selected_scores = scores[selected]
        threshold = float(
            np.quantile(selected_scores, 1.0 - config.keep_fraction)
        )
        scale = max(
            float(config.weight_temperature),
            0.25 * float(np.std(selected_scores)),
            1e-6,
        )
        logits = np.clip(
            (selected_scores - threshold) / scale, -80.0, 80.0
        )
        output[selected] = 1.0 / (1.0 + np.exp(-logits))
    return output


def _responsibilities(
    candidate_values: dict[str, np.ndarray], reference_count: int
) -> np.ndarray:
    if "_mstep_particle_offsets" in candidate_values:
        offsets = candidate_values["_mstep_particle_offsets"]
        indices = candidate_values["_mstep_reference_index"]
        posterior = candidate_values["_mstep_posterior"]
        output = np.zeros((len(offsets) - 1, reference_count), dtype=np.float32)
        for particle in range(len(output)):
            selected = slice(int(offsets[particle]), int(offsets[particle + 1]))
            np.add.at(output[particle], indices[selected], posterior[selected])
        return output
    indices = candidate_values["reference_index"]
    posterior = candidate_values["posterior"]
    output = np.zeros((indices.shape[0], reference_count), dtype=np.float32)
    for particle in range(indices.shape[0]):
        np.add.at(output[particle], indices[particle], posterior[particle])
    return output


def _best_poses(candidate_values: dict[str, np.ndarray]) -> PoseSet:
    best = np.argmax(candidate_values["posterior"], axis=1)
    rows = np.arange(len(best))
    return PoseSet(
        candidate_values["angle_deg"][rows, best],
        candidate_values["shift_y_px"][rows, best],
        candidate_values["shift_x_px"][rows, best],
        candidate_values["mirror"][rows, best],
    )


def _center_reference(reference: np.ndarray) -> tuple[np.ndarray, float, float]:
    weight = np.abs(reference).astype(np.float64)
    total = float(weight.sum())
    if total <= 1e-12:
        return reference, 0.0, 0.0
    y, x = np.indices(reference.shape, dtype=np.float64)
    center = integer_center(reference.shape[0])
    shift_y = float(np.round(center - np.sum(y * weight) / total))
    shift_x = float(np.round(center - np.sum(x * weight) / total))
    return (
        np.roll(reference, (int(shift_y), int(shift_x)), axis=(0, 1)),
        shift_y,
        shift_x,
    )


def _finalize_spatial_reference_sums(
    sums: np.ndarray,
    weights: np.ndarray,
    config: AlignmentConfig,
    *,
    pre_shifts: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    reference_count, size, _ = sums.shape
    references = np.zeros((reference_count, size, size), dtype=np.float32)
    center_shifts: list[tuple[float, float]] = []
    mask = soft_circular_mask(size, config.mask_radius, config.mask_soft_edge)
    for reference_index in range(reference_count):
        if weights[reference_index] > 1e-8:
            reference = (sums[reference_index] / weights[reference_index]).astype(
                np.float32
            )
        else:
            reference = np.zeros((size, size), dtype=np.float32)
        if pre_shifts is not None:
            shift_y, shift_x = pre_shifts[reference_index]
            reference = np.roll(
                reference, (int(shift_y), int(shift_x)), axis=(0, 1)
            )
        if config.lowpass_sigma > 0:
            reference = cv2.GaussianBlur(reference, (0, 0), config.lowpass_sigma)
        reference *= mask
        if config.center_references:
            reference, shift_y, shift_x = _center_reference(reference)
        else:
            shift_y = shift_x = 0.0
        references[reference_index] = reference
        center_shifts.append((shift_y, shift_x))
    return references, center_shifts


def _finalize_fourier_reference_sums(
    sums: np.ndarray,
    weights: np.ndarray,
    config: AlignmentConfig,
    *,
    pre_shifts: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    reference_count, size, _ = sums.shape
    spatial = np.zeros((reference_count, size, size), dtype=np.float32)
    for reference_index in range(reference_count):
        if weights[reference_index] > 1e-8:
            spatial[reference_index] = profile_call(
                "reference_ifft",
                np.fft.ifft2,
                sums[reference_index] / weights[reference_index],
            ).real.astype(np.float32)
    return _finalize_spatial_reference_sums(
        spatial,
        np.ones(reference_count, dtype=np.float64),
        config,
        pre_shifts=pre_shifts,
    )


def _update_references(
    images: np.ndarray,
    candidate_values: dict[str, np.ndarray],
    inlier_weights: np.ndarray,
    reference_count: int,
    config: AlignmentConfig,
    *,
    subset: np.ndarray | None = None,
    particle_fourier: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    size = images.shape[1]
    sums = np.zeros((reference_count, size, size), dtype=np.float64)
    weights = np.zeros(reference_count, dtype=np.float64)
    allowed = np.ones(len(images), dtype=np.bool_) if subset is None else subset
    mstep_offsets = candidate_values.get("_mstep_particle_offsets")
    field_prefix = "_mstep_" if mstep_offsets is not None else ""
    for particle in range(len(images)):
        if not allowed[particle]:
            continue
        if mstep_offsets is None:
            candidates = range(candidate_values["posterior"].shape[1])
        else:
            candidates = range(
                int(mstep_offsets[particle]), int(mstep_offsets[particle + 1])
            )
        for candidate in candidates:
            if mstep_offsets is None:
                index = (particle, candidate)
            else:
                index = candidate
            reference_index = int(
                candidate_values[f"{field_prefix}reference_index"][index]
            )
            weight = float(candidate_values[f"{field_prefix}posterior"][index]) * float(
                inlier_weights[particle]
            )
            if weight <= 1e-12:
                continue
            aligned = transform_image(
                images[particle],
                angle_deg=float(candidate_values[f"{field_prefix}angle_deg"][index]),
                shift_y_px=float(candidate_values[f"{field_prefix}shift_y_px"][index]),
                shift_x_px=float(candidate_values[f"{field_prefix}shift_x_px"][index]),
                mirror=bool(candidate_values[f"{field_prefix}mirror"][index]),
            )
            sums[reference_index] += weight * aligned
            weights[reference_index] += weight
    references, center_shifts = _finalize_spatial_reference_sums(
        sums, weights, config
    )
    return references, weights.astype(np.float32), center_shifts


def _update_references_fourier(
    images: np.ndarray,
    candidate_values: dict[str, np.ndarray],
    inlier_weights: np.ndarray,
    reference_count: int,
    config: AlignmentConfig,
    *,
    subset: np.ndarray | None = None,
    particle_fourier: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """Accumulate aligned particle DFTs and invert once per reference."""
    from ._fourier_native import transform_fourier_cpu

    if particle_fourier is None:
        particle_fourier = np.fft.fft2(images, axes=(-2, -1)).astype(np.complex64)
    fourier = np.asarray(particle_fourier, dtype=np.complex64)
    if fourier.shape != images.shape:
        raise ValueError("particle_fourier must match the image stack shape")

    size = images.shape[1]
    sums = np.zeros((reference_count, size, size), dtype=np.complex128)
    weights = np.zeros(reference_count, dtype=np.float64)
    allowed = np.ones(len(images), dtype=np.bool_) if subset is None else subset
    mstep_offsets = candidate_values.get("_mstep_particle_offsets")
    field_prefix = "_mstep_" if mstep_offsets is not None else ""
    for particle in range(len(images)):
        if not allowed[particle]:
            continue
        if mstep_offsets is None:
            candidates = range(candidate_values["posterior"].shape[1])
        else:
            candidates = range(
                int(mstep_offsets[particle]), int(mstep_offsets[particle + 1])
            )
        for candidate in candidates:
            index = (particle, candidate) if mstep_offsets is None else candidate
            reference_index = int(
                candidate_values[f"{field_prefix}reference_index"][index]
            )
            weight = float(candidate_values[f"{field_prefix}posterior"][index]) * float(
                inlier_weights[particle]
            )
            if weight <= 1e-12:
                continue
            aligned = transform_fourier_cpu(
                fourier[particle],
                angle_deg=float(candidate_values[f"{field_prefix}angle_deg"][index]),
                shift_y_px=float(candidate_values[f"{field_prefix}shift_y_px"][index]),
                shift_x_px=float(candidate_values[f"{field_prefix}shift_x_px"][index]),
                mirror=bool(candidate_values[f"{field_prefix}mirror"][index]),
            )
            sums[reference_index] += weight * aligned
            weights[reference_index] += weight

    references, center_shifts = _finalize_fourier_reference_sums(
        sums, weights, config
    )
    return references, weights.astype(np.float32), center_shifts


def _validate_halfset_membership(
    first_half: np.ndarray, particle_count: int
) -> np.ndarray:
    values = np.asarray(first_half, dtype=np.bool_)
    if values.shape != (particle_count,):
        raise ValueError(f"first_half must have shape {(particle_count,)}")
    return values


def _update_references_shared(
    images: np.ndarray,
    candidate_values: dict[str, np.ndarray],
    inlier_weights: np.ndarray,
    reference_count: int,
    config: AlignmentConfig,
    *,
    first_half: np.ndarray,
    particle_fourier: np.ndarray | None = None,
) -> _SharedReferenceUpdate:
    """Transform each spatial candidate once and retain unnormalized half sums."""
    size = images.shape[1]
    membership = _validate_halfset_membership(first_half, len(images))
    half_sums = np.zeros((2, reference_count, size, size), dtype=np.float64)
    half_weights = np.zeros((2, reference_count), dtype=np.float64)
    mstep_offsets = candidate_values.get("_mstep_particle_offsets")
    field_prefix = "_mstep_" if mstep_offsets is not None else ""
    transformed_count = 0
    for particle in range(len(images)):
        half_index = 0 if membership[particle] else 1
        candidates = (
            range(candidate_values["posterior"].shape[1])
            if mstep_offsets is None
            else range(int(mstep_offsets[particle]), int(mstep_offsets[particle + 1]))
        )
        for candidate in candidates:
            index = (particle, candidate) if mstep_offsets is None else candidate
            reference_index = int(
                candidate_values[f"{field_prefix}reference_index"][index]
            )
            weight = float(candidate_values[f"{field_prefix}posterior"][index]) * float(
                inlier_weights[particle]
            )
            if weight <= 1e-12:
                continue
            aligned = transform_image(
                images[particle],
                angle_deg=float(candidate_values[f"{field_prefix}angle_deg"][index]),
                shift_y_px=float(candidate_values[f"{field_prefix}shift_y_px"][index]),
                shift_x_px=float(candidate_values[f"{field_prefix}shift_x_px"][index]),
                mirror=bool(candidate_values[f"{field_prefix}mirror"][index]),
            )
            half_sums[half_index, reference_index] += weight * aligned
            half_weights[half_index, reference_index] += weight
            transformed_count += 1
    profile = current_profile()
    if profile is not None:
        profile.count("shared_mstep_candidate_transforms", transformed_count)

    full_weights = half_weights[0] + half_weights[1]
    references, center_shifts = _finalize_spatial_reference_sums(
        half_sums[0] + half_sums[1], full_weights, config
    )
    half_started = time.perf_counter()
    half_references = np.stack(
        [
            _finalize_spatial_reference_sums(
                half_sums[index],
                half_weights[index],
                config,
                pre_shifts=center_shifts,
            )[0]
            for index in range(2)
        ]
    )
    return _SharedReferenceUpdate(
        references=references,
        effective_weights=full_weights.astype(np.float32),
        center_shifts=center_shifts,
        half_references=half_references,
        half_weights=half_weights.astype(np.float32),
        halfset_finalize_seconds=time.perf_counter() - half_started,
    )


def _update_references_fourier_shared(
    images: np.ndarray,
    candidate_values: dict[str, np.ndarray],
    inlier_weights: np.ndarray,
    reference_count: int,
    config: AlignmentConfig,
    *,
    first_half: np.ndarray,
    particle_fourier: np.ndarray | None = None,
) -> _SharedReferenceUpdate:
    """Transform each Fourier candidate once and retain unnormalized half sums."""
    from ._fourier_native import transform_fourier_cpu

    if particle_fourier is None:
        particle_fourier = np.fft.fft2(images, axes=(-2, -1)).astype(np.complex64)
    fourier = np.asarray(particle_fourier, dtype=np.complex64)
    if fourier.shape != images.shape:
        raise ValueError("particle_fourier must match the image stack shape")

    size = images.shape[1]
    membership = _validate_halfset_membership(first_half, len(images))
    half_sums = np.zeros((2, reference_count, size, size), dtype=np.complex128)
    half_weights = np.zeros((2, reference_count), dtype=np.float64)
    mstep_offsets = candidate_values.get("_mstep_particle_offsets")
    field_prefix = "_mstep_" if mstep_offsets is not None else ""
    transformed_count = 0
    for particle in range(len(images)):
        half_index = 0 if membership[particle] else 1
        candidates = (
            range(candidate_values["posterior"].shape[1])
            if mstep_offsets is None
            else range(int(mstep_offsets[particle]), int(mstep_offsets[particle + 1]))
        )
        for candidate in candidates:
            index = (particle, candidate) if mstep_offsets is None else candidate
            reference_index = int(
                candidate_values[f"{field_prefix}reference_index"][index]
            )
            weight = float(candidate_values[f"{field_prefix}posterior"][index]) * float(
                inlier_weights[particle]
            )
            if weight <= 1e-12:
                continue
            aligned = transform_fourier_cpu(
                fourier[particle],
                angle_deg=float(candidate_values[f"{field_prefix}angle_deg"][index]),
                shift_y_px=float(candidate_values[f"{field_prefix}shift_y_px"][index]),
                shift_x_px=float(candidate_values[f"{field_prefix}shift_x_px"][index]),
                mirror=bool(candidate_values[f"{field_prefix}mirror"][index]),
            )
            half_sums[half_index, reference_index] += weight * aligned
            half_weights[half_index, reference_index] += weight
            transformed_count += 1
    profile = current_profile()
    if profile is not None:
        profile.count("shared_mstep_candidate_transforms", transformed_count)

    full_weights = half_weights[0] + half_weights[1]
    references, center_shifts = _finalize_fourier_reference_sums(
        half_sums[0] + half_sums[1], full_weights, config
    )
    half_started = time.perf_counter()
    half_references = np.stack(
        [
            _finalize_fourier_reference_sums(
                half_sums[index],
                half_weights[index],
                config,
                pre_shifts=center_shifts,
            )[0]
            for index in range(2)
        ]
    )
    return _SharedReferenceUpdate(
        references=references,
        effective_weights=full_weights.astype(np.float32),
        center_shifts=center_shifts,
        half_references=half_references,
        half_weights=half_weights.astype(np.float32),
        halfset_finalize_seconds=time.perf_counter() - half_started,
    )


def _apply_reference_center_shifts(
    candidate_values: dict[str, np.ndarray], shifts: list[tuple[float, float]]
) -> None:
    for reference_index, (shift_y, shift_x) in enumerate(shifts):
        selected = candidate_values["reference_index"] == reference_index
        candidate_values["shift_y_px"][selected] += np.float32(shift_y)
        candidate_values["shift_x_px"][selected] += np.float32(shift_x)
        if "_mstep_reference_index" in candidate_values:
            selected_mstep = (
                candidate_values["_mstep_reference_index"] == reference_index
            )
            candidate_values["_mstep_shift_y_px"][selected_mstep] += np.float32(shift_y)
            candidate_values["_mstep_shift_x_px"][selected_mstep] += np.float32(shift_x)


def _stable_frc_cutoff(
    curve: np.ndarray,
    size: int,
    *,
    threshold: float = 0.143,
    persistence: int = 3,
) -> float:
    """Return the first smoothed threshold crossing sustained for several rings."""
    values = np.asarray(curve, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("FRC curve must be a non-empty one-dimensional array.")
    if size <= 0 or persistence < 1:
        raise ValueError("size and persistence must be positive.")
    padded = np.pad(values, (1, 1), mode="edge")
    smoothed = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    for ring in range(1, len(smoothed) - persistence + 1):
        if np.all(smoothed[ring : ring + persistence] < threshold):
            previous = smoothed[ring - 1]
            current = smoothed[ring]
            if previous > threshold and current != previous:
                crossing = (ring - 1) + (previous - threshold) / (previous - current)
            else:
                crossing = float(ring)
            return float(crossing / size)
    return float((len(smoothed) - 1) / size)


@profile_stage("frc")
def _frc(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float, float]:
    size = a.shape[0]
    fa = np.fft.fftshift(np.fft.fft2(a))
    fb = np.fft.fftshift(np.fft.fft2(b))
    center = integer_center(size)
    y, x = np.indices(a.shape, dtype=np.float32)
    radius = np.floor(np.sqrt((y - center) ** 2 + (x - center) ** 2)).astype(np.int32)
    curve = np.zeros(size // 2, dtype=np.float32)
    for ring in range(len(curve)):
        selected = radius == ring
        numerator = np.sum(fa[selected] * np.conj(fb[selected]))
        denominator = np.sqrt(
            np.sum(np.abs(fa[selected]) ** 2) * np.sum(np.abs(fb[selected]) ** 2)
        )
        curve[ring] = float(np.real(numerator) / max(float(denominator), 1e-12))
    above = np.flatnonzero(curve >= 0.143)
    cutoff = float(above[-1] / size) if len(above) else 0.0
    stable_cutoff = _stable_frc_cutoff(curve, size)
    return curve, cutoff, stable_cutoff


def _maximum_reference_correlation(references: np.ndarray) -> float:
    if len(references) < 2:
        return 0.0
    flattened = references.reshape(len(references), -1).astype(np.float64)
    flattened -= flattened.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(flattened, axis=1, keepdims=True)
    normalized = flattened / np.maximum(norms, 1e-12)
    correlations = normalized @ normalized.T
    correlations[np.diag_indices_from(correlations)] = -np.inf
    return float(np.max(correlations))


@profile_stage("halfset_diagnostics")
def _halfset_diagnostics(
    images: np.ndarray,
    candidate_values: dict[str, np.ndarray],
    inlier_weights: np.ndarray,
    reference_count: int,
    config: AlignmentConfig,
    first_half: np.ndarray,
    reference_updater,
    particle_fourier: np.ndarray,
) -> dict[str, Any]:
    first, first_weights, _ = profile_call(
        "half_a_update", reference_updater,
        images,
        candidate_values,
        inlier_weights,
        reference_count,
        config,
        subset=first_half,
        particle_fourier=particle_fourier,
    )
    second, second_weights, _ = profile_call(
        "half_b_update", reference_updater,
        images,
        candidate_values,
        inlier_weights,
        reference_count,
        config,
        subset=~first_half,
        particle_fourier=particle_fourier,
    )
    cutoffs = []
    stable_cutoffs = []
    curves = []
    for index in range(reference_count):
        curve, cutoff, stable_cutoff = _frc(first[index], second[index])
        curves.append(curve)
        cutoffs.append(cutoff)
        stable_cutoffs.append(stable_cutoff)
    return {
        "halfset_effective_weight": np.stack((first_weights, second_weights)),
        "frc": np.stack(curves),
        "frc_0143_cutoff_cyc_per_px": np.asarray(cutoffs, dtype=np.float32),
        "frc_0143_stable_cutoff_cyc_per_px": np.asarray(
            stable_cutoffs, dtype=np.float32
        ),
    }


@profile_stage("halfset_diagnostics")
def _shared_halfset_diagnostics(
    shared: _SharedReferenceUpdate,
) -> dict[str, Any]:
    first, second = shared.half_references
    cutoffs = []
    stable_cutoffs = []
    curves = []
    for index in range(first.shape[0]):
        curve, cutoff, stable_cutoff = _frc(first[index], second[index])
        curves.append(curve)
        cutoffs.append(cutoff)
        stable_cutoffs.append(stable_cutoff)
    return {
        "halfset_effective_weight": shared.half_weights,
        "frc": np.stack(curves),
        "frc_0143_cutoff_cyc_per_px": np.asarray(cutoffs, dtype=np.float32),
        "frc_0143_stable_cutoff_cyc_per_px": np.asarray(
            stable_cutoffs, dtype=np.float32
        ),
    }


@profile_stage("alignment_engine")
def run_soft_alignment_cpu(
    images: np.ndarray,
    references: np.ndarray,
    *,
    config: AlignmentConfig,
    class_priors: np.ndarray | None,
    initial_poses: PoseSet | None,
    workflow: str,
    _candidate_inference=None,
    _reference_updater=None,
    _shared_reference_updater=None,
    _backend_name: str = "cpu",
) -> AlignmentResult:
    images = np.asarray(images, dtype=np.float32)
    references = np.asarray(references, dtype=np.float32)
    if references.ndim == 2:
        references = references[None]
    if references.ndim != 3 or references.shape[0] == 0:
        raise ValueError("references must have shape (K, H, W) or (H, W).")
    if images.ndim != 3 or images.shape[1:] != references.shape[1:]:
        raise ValueError("images and references must have matching image shapes.")
    validate_even_square(tuple(images.shape[1:]))
    if initial_poses is not None and len(initial_poses) != len(images):
        raise ValueError(
            "initial_poses and images must contain the same number of items."
        )

    config = config.normalized(workflow=workflow)
    if (
        config.search_strategy in {"adaptive_posterior", "quadratic_refine"}
        and initial_poses is None
    ):
        raise ValueError(f"{config.search_strategy} search requires initial_poses")
    if _candidate_inference is None:

        def candidate_inference(*args, rescue_mask=None):
            if config.search_strategy == "adaptive_posterior":
                return infer_adaptive_candidates_cpu(
                    *args,
                    rescue_mask=rescue_mask,
                )
            if config.search_strategy == "quadratic_refine":
                return infer_quadratic_candidates_cpu(
                    *args,
                    rescue_mask=rescue_mask,
                )
            return infer_top_candidates(*args)
    else:
        candidate_inference = _candidate_inference
    reference_updater = _reference_updater or (
        _update_references_fourier
        if config.reference_update == "fourier"
        else _update_references
    )
    if _shared_reference_updater is not None:
        shared_reference_updater = _shared_reference_updater
    elif _reference_updater is None:
        shared_reference_updater = (
            _update_references_fourier_shared
            if config.reference_update == "fourier"
            else _update_references_shared
        )
    else:
        shared_reference_updater = None
    priors = validate_class_priors(class_priors, len(images), len(references))
    fixed_reference_groups = (
        np.argmax(priors, axis=1).astype(np.int32)
        if workflow == "refine"
        and config.search_strategy == "quadratic_refine"
        and np.all(np.count_nonzero(priors > 0.0, axis=1) == 1)
        else None
    )
    prepared_particles = profile_call("prepare_particles", prepare_stack, images, config)
    update_fourier = (
        profile_call("raw_particle_fft", np.fft.fft2, images, axes=(-2, -1)).astype(np.complex64)
        if config.reference_update == "fourier"
        else prepared_particles.fourier
    )
    current_references = references.copy()
    history = [current_references.copy()] if config.store_history else []
    diagnostics: list[dict[str, Any]] = []
    pose_prior = initial_poses
    candidate_values: dict[str, np.ndarray] | None = None
    inlier_weights = np.ones(len(images), dtype=np.float32)
    rescue_mask = np.zeros(len(images), dtype=np.bool_)
    permutation = np.random.default_rng(config.random_seed).permutation(len(images))
    first_half = np.zeros(len(images), dtype=np.bool_)
    first_half[permutation[: (len(images) + 1) // 2]] = True

    for iteration in range(config.max_iterations):
        started = time.perf_counter()
        prepared_references = profile_call(
            "prepare_references", prepare_stack, current_references, config
        )
        temperature = _temperature(config, iteration)
        candidate_values = profile_call(
            "candidate_inference", candidate_inference,
            prepared_particles,
            prepared_references,
            config,
            priors,
            temperature,
            pose_prior,
            rescue_mask=rescue_mask,
        )
        next_rescue_mask = np.asarray(
            candidate_values.get("_rescue_next", np.zeros(len(images), dtype=np.bool_)),
            dtype=np.bool_,
        )
        # Reserve the final iteration for local confirmation. A newly scheduled
        # global proposal therefore needs at least two remaining iterations.
        if iteration >= config.max_iterations - 2:
            next_rescue_mask = np.zeros(len(images), dtype=np.bool_)
            candidate_values["_rescue_next"] = next_rescue_mask
        best_scores = np.max(candidate_values["score"], axis=1)
        inlier_weights = _inlier_weights(
            best_scores, config, groups=fixed_reference_groups
        )
        responsibilities = _responsibilities(candidate_values, len(current_references))
        shared_update = None
        if config.halfset_diagnostics and shared_reference_updater is not None:
            shared_update = profile_call(
                "shared_reference_update",
                shared_reference_updater,
                images,
                candidate_values,
                inlier_weights,
                len(current_references),
                config,
                first_half=first_half,
                particle_fourier=update_fourier,
            )
            updated = shared_update.references
            effective_weights = shared_update.effective_weights
            center_shifts = shared_update.center_shifts
        else:
            updated, effective_weights, center_shifts = profile_call(
                "full_reference_update", reference_updater,
                images,
                candidate_values,
                inlier_weights,
                len(current_references),
                config,
                particle_fourier=update_fourier,
            )
        minimum_effective_weight = max(
            1.0, 0.01 * len(images) / len(current_references)
        )
        empty = np.flatnonzero(effective_weights < minimum_effective_weight)
        if len(empty):
            entropy = -np.sum(
                responsibilities * np.log(np.maximum(responsibilities, 1e-30)), axis=1
            )
            reseed_order = np.argsort(-entropy, kind="stable")
            for offset, reference_index in enumerate(empty):
                updated[reference_index] = prepared_particles.spatial[
                    reseed_order[offset % len(reseed_order)]
                ]
        _apply_reference_center_shifts(candidate_values, center_shifts)
        # Global and reference-free workflows must continue to explore the full
        # pose space as references evolve. Only an explicit refine workflow is
        # allowed to turn the previous estimate into a local pose prior.
        pose_prior = _best_poses(candidate_values) if workflow == "refine" else None
        rescue_mask = next_rescue_mask
        relative_change = float(
            np.linalg.norm(updated - current_references)
            / max(float(np.linalg.norm(current_references)), 1e-8)
        )
        posterior = candidate_values["posterior"]
        scores = candidate_values["score"]
        expected_scores = np.where(
            posterior > 0.0,
            posterior * np.where(np.isfinite(scores), scores, 0.0),
            0.0,
        )
        if "_mstep_particle_offsets" in candidate_values:
            mstep_posterior = candidate_values["_mstep_posterior"]
            mstep_scores = candidate_values["_mstep_score"]
            weighted_scores = mstep_posterior * mstep_scores
            expected_score_mean = float(
                np.mean(
                    np.add.reduceat(
                        weighted_scores,
                        candidate_values["_mstep_particle_offsets"][:-1],
                    )
                )
            )
        else:
            expected_score_mean = float(np.mean(np.sum(expected_scores, axis=1)))
        pose_entropy_values = candidate_values.get("_full_pose_entropy")
        if pose_entropy_values is None:
            pose_entropy_values = -np.sum(
                np.where(
                    posterior > 0.0,
                    posterior * np.log(np.maximum(posterior, 1e-30)),
                    0.0,
                ),
                axis=1,
            )
        map_posterior_values = candidate_values.get(
            "_full_map_posterior", np.max(posterior, axis=1)
        )
        iteration_diagnostics: dict[str, Any] = {
            "iteration": iteration,
            "temperature": temperature,
            "seconds": time.perf_counter()
            - started
            - (
                shared_update.halfset_finalize_seconds
                if shared_update is not None
                else 0.0
            ),
            "reference_relative_change": relative_change,
            "effective_component_weight": effective_weights,
            "component_weight_cv": float(
                np.std(effective_weights)
                / max(float(np.mean(effective_weights)), 1e-12)
            ),
            "minimum_component_weight_to_mean": float(
                np.min(effective_weights)
                / max(float(np.mean(effective_weights)), 1e-12)
            ),
            "maximum_offdiagonal_reference_correlation": (
                _maximum_reference_correlation(updated)
            ),
            "mean_max_responsibility": float(np.mean(np.max(responsibilities, axis=1))),
            "mean_expected_fourier_ncc": float(expected_score_mean),
            "mean_inlier_weight": float(np.mean(inlier_weights)),
            "mean_pose_posterior_entropy": float(np.mean(pose_entropy_values)),
            "mean_map_pose_posterior": float(np.mean(map_posterior_values)),
            "minimum_effective_component_weight": minimum_effective_weight,
            "reseeded_components": empty.astype(np.int32),
        }
        if config.search_strategy == "adaptive_posterior":
            iteration_diagnostics.update(
                {
                    "mean_coarse_cell_count": float(
                        np.mean(candidate_values["_coarse_cell_count"])
                    ),
                    "mean_selected_coarse_cell_count": float(
                        np.mean(candidate_values["_selected_coarse_cell_count"])
                    ),
                    "mean_selected_coarse_mass": float(
                        np.mean(candidate_values["_selected_coarse_mass"])
                    ),
                    "mean_fine_candidate_count": float(
                        np.mean(candidate_values["_fine_candidate_count"])
                    ),
                    "mean_retained_fine_mass": float(
                        np.mean(candidate_values["_retained_fine_mass"])
                    ),
                    "adaptive_cap_hit_count": int(
                        np.sum(candidate_values["_adaptive_cap_hit"])
                    ),
                    "boundary_hit_count": int(
                        np.sum(candidate_values["_boundary_hit"])
                    ),
                    "mean_normalized_pose_entropy": float(
                        np.mean(candidate_values["_normalized_pose_entropy"])
                    ),
                    "rescue_entropy_trigger_count": int(
                        np.sum(candidate_values["_rescue_entropy_trigger"])
                    ),
                    "rescue_map_trigger_count": int(
                        np.sum(candidate_values["_rescue_map_trigger"])
                    ),
                    "rescue_particle_count": int(
                        np.sum(candidate_values["_rescue_used"])
                    ),
                    "rescue_accepted_count": int(
                        np.sum(candidate_values["_rescue_accepted"])
                    ),
                    "rescue_rejected_count": int(
                        np.sum(
                            candidate_values["_rescue_used"]
                            & ~candidate_values["_rescue_accepted"]
                        )
                    ),
                    "mean_rescue_score_gain": float(
                        np.nanmean(candidate_values["_rescue_score_gain"])
                        if np.any(candidate_values["_rescue_used"])
                        else 0.0
                    ),
                    "rescue_scheduled_count": int(np.sum(next_rescue_mask)),
                }
            )
        elif config.search_strategy == "quadratic_refine":
            iteration_diagnostics.update(
                {
                    "mean_screened_angle_count": float(
                        np.mean(candidate_values["_quadratic_screened_angle_count"])
                    ),
                    "mean_active_reference_count": float(
                        np.mean(candidate_values["_quadratic_active_reference_count"])
                    ),
                    "mean_angular_mode_count": float(
                        np.mean(candidate_values["_quadratic_angular_mode_count"])
                    ),
                    "mean_posterior_support_count": float(
                        np.mean(
                            candidate_values[
                                "_quadratic_posterior_support_count"
                            ]
                        )
                    ),
                    "translation_fit_attempt_count": int(
                        np.sum(candidate_values["_quadratic_translation_fit_attempt_count"])
                    ),
                    "translation_fit_accept_count": int(
                        np.sum(candidate_values["_quadratic_translation_fit_accept_count"])
                    ),
                    "angle_fit_attempt_count": int(
                        np.sum(candidate_values["_quadratic_angle_fit_attempt_count"])
                    ),
                    "angle_fit_accept_count": int(
                        np.sum(candidate_values["_quadratic_angle_fit_accept_count"])
                    ),
                    "quadratic_boundary_hit_count": int(
                        np.sum(candidate_values["_quadratic_boundary_hit_count"])
                    ),
                    "quadratic_flat_or_convex_reject_count": int(
                        np.sum(candidate_values["_quadratic_flat_or_convex_count"])
                    ),
                    "quadratic_out_of_bounds_reject_count": int(
                        np.sum(candidate_values["_quadratic_out_of_bounds_count"])
                    ),
                    "quadratic_exact_reject_count": int(
                        np.sum(candidate_values["_quadratic_exact_reject_count"])
                    ),
                    "mean_quadratic_objective_gain": float(
                        np.mean(candidate_values["_quadratic_objective_gain"])
                    ),
                    "correlation_map_ifft_count": int(
                        np.sum(candidate_values["_quadratic_ifft_count"])
                    ),
                }
            )
        if config.halfset_diagnostics:
            if shared_update is not None:
                iteration_diagnostics.update(
                    _shared_halfset_diagnostics(shared_update)
                )
            else:
                iteration_diagnostics.update(
                    _halfset_diagnostics(
                        images,
                        candidate_values,
                        inlier_weights,
                        len(current_references),
                        config,
                        first_half,
                        reference_updater,
                        update_fourier,
                    )
                )
        diagnostics.append(iteration_diagnostics)
        current_references = updated
        if config.store_history:
            history.append(current_references.copy())
        profile = current_profile()
        if profile is not None:
            profile.iterations.append({
                "iteration": iteration,
                "wall_seconds_including_diagnostics": time.perf_counter() - started,
                "legacy_seconds_excluding_halfsets": iteration_diagnostics["seconds"],
            })

    assert candidate_values is not None
    responsibilities = _responsibilities(candidate_values, len(current_references))
    assignments = np.argmax(responsibilities, axis=1).astype(np.int32)
    candidates = CandidateSet(
        **{name: candidate_values[name] for name in STANDARD_CANDIDATE_FIELDS}
    )
    poses = _best_poses(candidate_values)
    active_frequency = prepared_particles.frequency_mask > 0.0
    active_ring_count = np.bincount(
        prepared_particles.score_weight_bins[active_frequency].ravel(),
        minlength=prepared_particles.score_weight_profiles.shape[1],
    )
    active_rings = active_ring_count > 0
    active_profiles = prepared_particles.score_weight_profiles[:, active_rings]
    profile_means = (
        active_profiles @ active_ring_count[active_rings]
    ) / np.sum(active_ring_count)
    return AlignmentResult(
        references=current_references,
        poses=poses,
        reference_assignments=assignments,
        responsibilities=responsibilities,
        candidates=candidates,
        inlier_weights=inlier_weights,
        diagnostics=diagnostics,
        reference_history=history,
        metadata={
            "engine": "alignimg-soft-fourier-cpu",
            "backend": _backend_name,
            "workflow": workflow,
            "config": asdict(config),
            "transform_convention": (
                "periodic-x-mirror-ccw_rotate-shift; integer-origin; wrap"
            ),
            "center_convention": CENTER_CONVENTION,
            "candidate_scoring": config.candidate_scoring,
            "score_model": config.score_model,
            "reference_update": config.reference_update,
            "fft_policy": (
                "normalized scoring FFT and raw update FFT cached once; Fourier-native "
                "candidate scoring and Fourier-domain soft M-step; one output IFFT per "
                "reference plus half-set diagnostic IFFTs when enabled"
                if config.candidate_scoring == "fourier"
                and config.reference_update == "fourier"
                else "particle FFT/polar descriptors cached; Fourier-native rotation, "
                "translation, and NCC candidate scoring; real-space soft M-step"
                if config.candidate_scoring == "fourier"
                else "particle FFT/polar descriptors cached; polar angular proposal and "
                f"Cartesian Fourier reranking; {config.reference_update}-domain soft M-step"
                if config.search_strategy == "proposal"
                else "particle FFT cached; prior-centered correlation-map translation and "
                f"continuous quadratic pose modes; {config.reference_update}-domain soft M-step"
                if config.search_strategy == "quadratic_refine"
                else "particle FFT cached; prior-centered coarse Fourier-NCC scoring; "
                f"posterior-mass fine oversampling; {config.reference_update}-domain soft M-step"
            ),
            "score_weighting": (
                "per-particle inverse three-point-smoothed radial empirical power"
                if config.score_model == "whitened_fourier_ncc"
                else "uniform within the configured frequency band"
            ),
            "score_weight_summary": {
                "profile_count": int(len(active_profiles)),
                "active_frequency_count": int(np.count_nonzero(active_frequency)),
                "active_minimum": float(np.min(active_profiles)),
                "active_mean": float(np.mean(profile_means)),
                "active_maximum": float(np.max(active_profiles)),
            },
            "search_strategy": config.search_strategy,
            "robust_weighting_scope": (
                "fixed_reference"
                if fixed_reference_groups is not None
                else "global"
            ),
            "relion_like_not_full_relion_likelihood": (
                config.search_strategy in {"adaptive_posterior", "quadratic_refine"}
            ),
            "random_seed": config.random_seed,
            "halfset_membership": first_half,
            "halfset_update_policy": (
                "shared_unnormalized_accumulation"
                if config.halfset_diagnostics and shared_reference_updater is not None
                else "separate_reference_updates"
                if config.halfset_diagnostics
                else "disabled"
            ),
        },
        _pose_entropy_values=candidate_values.get("_full_pose_entropy"),
        _map_posterior_values=candidate_values.get("_full_map_posterior"),
    )


@profile_stage("spectral_bootstrap")
def spectral_bootstrap(
    images: np.ndarray,
    n_components: int,
    config: AlignmentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic polar-harmonic k-means++/medoid initialization."""
    images = np.asarray(images, dtype=np.float32)
    if not 1 <= int(n_components) <= len(images):
        raise ValueError("n_components must be in the range 1..N.")
    prepared = prepare_stack(images, config)
    harmonics = np.abs(np.fft.rfft(prepared.polar, axis=1))
    harmonic_count = min(9, harmonics.shape[1])
    descriptors = harmonics[:, 1:harmonic_count].reshape(len(images), -1)
    descriptors -= descriptors.mean(axis=0, keepdims=True)
    scale = descriptors.std(axis=0, keepdims=True)
    descriptors /= np.maximum(scale, 1e-6)
    rng = np.random.default_rng(config.random_seed)
    centers = [int(rng.integers(len(images)))]
    distance_sq = np.sum((descriptors - descriptors[centers[0]]) ** 2, axis=1)
    for _ in range(1, int(n_components)):
        total = float(distance_sq.sum())
        if total <= 1e-12:
            remaining = next(
                index for index in range(len(images)) if index not in centers
            )
        else:
            remaining = int(rng.choice(len(images), p=distance_sq / total))
            if remaining in centers:
                remaining = next(
                    index for index in range(len(images)) if index not in centers
                )
        centers.append(remaining)
        distance_sq = np.minimum(
            distance_sq,
            np.sum((descriptors - descriptors[remaining]) ** 2, axis=1),
        )
    labels = np.zeros(len(images), dtype=np.int32)
    for _ in range(20):
        distances = np.stack(
            [
                np.sum((descriptors - descriptors[index]) ** 2, axis=1)
                for index in centers
            ],
            axis=1,
        )
        new_labels = np.argmin(distances, axis=1).astype(np.int32)
        new_centers = []
        for component in range(int(n_components)):
            members = np.flatnonzero(new_labels == component)
            if not len(members):
                members = np.asarray([int(np.argmax(np.min(distances, axis=1)))])
            centroid = descriptors[members].mean(axis=0)
            medoid = int(
                members[
                    np.argmin(np.sum((descriptors[members] - centroid) ** 2, axis=1))
                ]
            )
            new_centers.append(medoid)
        if np.array_equal(new_labels, labels) and new_centers == centers:
            labels = new_labels
            break
        labels, centers = new_labels, new_centers
    references = np.empty(
        (int(n_components), images.shape[1], images.shape[2]), dtype=np.float32
    )
    bootstrap_sigma = config.lowpass_sigma if config.lowpass_sigma > 0 else 2.0
    for component, medoid in enumerate(centers):
        members = np.flatnonzero(labels == component)
        aligned_members = []
        for member in members:
            angle = _candidate_angles(
                prepared.polar[member], prepared.polar[medoid], 1
            )[0]
            aligned_members.append(
                transform_image(prepared.spatial[member], angle_deg=angle)
            )
        reference = np.mean(aligned_members, axis=0, dtype=np.float32)
        references[component] = cv2.GaussianBlur(reference, (0, 0), bootstrap_sigma)
    return references.astype(np.float32), labels
