"""Continuous local pose refinement using quadratic profile fits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ._adaptive import (
    STANDARD_CANDIDATE_FIELDS,
    _empty_candidate_values,
    _wrap_angle,
    symmetric_grid_offsets,
)
from ._fourier import (
    PreparedStack,
    _angle_difference,
    _fourier_ncc,
    _shift_fourier,
    reference_norms_for_particle,
    score_weights_for_particle,
)
from ._fourier_native import transform_fourier_cpu
from ._profiling import count as profile_count, profile_call, profile_stage
from .models import AlignmentConfig, PoseSet


@dataclass(frozen=True)
class TranslationProfiles:
    """Translation-optimized results for an ordered array of angles."""

    angle_deg: np.ndarray
    shift_y_px: np.ndarray
    shift_x_px: np.ndarray
    score: np.ndarray
    objective: np.ndarray
    fit_attempt_count: int
    fit_accept_count: int
    boundary_hit_count: int
    flat_or_convex_count: int
    out_of_bounds_count: int
    exact_reject_count: int
    objective_gain: float
    ifft_count: int


TranslationProfiler = Callable[
    [
        PreparedStack,
        PreparedStack,
        int,
        int,
        np.ndarray,
        bool,
        AlignmentConfig,
        float,
        PoseSet,
    ],
    TranslationProfiles,
]


def quadratic_vertex_offset(
    left: float,
    center: float,
    right: float,
    *,
    step: float = 1.0,
) -> tuple[float, str]:
    """Fit a three-point parabola and return its bounded vertex offset."""
    values = np.asarray((left, center, right, step), dtype=np.float64)
    if not np.all(np.isfinite(values)) or float(step) <= 0.0:
        return 0.0, "nonfinite"
    denominator = float(left) - 2.0 * float(center) + float(right)
    scale = max(abs(float(left)), abs(float(center)), abs(float(right)), 1.0)
    if denominator >= -32.0 * np.finfo(np.float64).eps * scale:
        return 0.0, "flat_or_convex"
    offset = 0.5 * (float(left) - float(right)) / denominator * float(step)
    if not np.isfinite(offset):
        return 0.0, "nonfinite"
    if abs(offset) > 0.5 * float(step) + 1e-12:
        return 0.0, "out_of_bounds"
    return float(offset), "accepted"


def angular_local_maxima(values: np.ndarray) -> np.ndarray:
    """Return one deterministic index for each local maximum plateau."""
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("angular profile must be a non-empty finite vector")
    maxima: list[int] = []
    start = 0
    while start < len(scores):
        stop = start
        while stop + 1 < len(scores) and scores[stop + 1] == scores[start]:
            stop += 1
        left = scores[start - 1] if start > 0 else -np.inf
        right = scores[stop + 1] if stop + 1 < len(scores) else -np.inf
        if scores[start] > left and scores[start] > right:
            maxima.append(start)
        start = stop + 1
    if not maxima:
        maxima.append(int(np.argmax(scores)))
    return np.asarray(maxima, dtype=np.int64)


def _integer_window(center: float, radius: float) -> np.ndarray:
    lower = int(np.ceil(float(center) - float(radius) - 1e-9))
    upper = int(np.floor(float(center) + float(radius) + 1e-9))
    if lower > upper:
        return np.asarray([int(np.rint(center))], dtype=np.int32)
    return np.arange(lower, upper + 1, dtype=np.int32)


def _translation_objective(
    score: float,
    shift_y: float,
    shift_x: float,
    *,
    center_y: float,
    center_x: float,
    temperature: float,
    sigma: float,
) -> float:
    return float(score) / float(temperature) - 0.5 * (
        ((float(shift_y) - center_y) / sigma) ** 2
        + ((float(shift_x) - center_x) / sigma) ** 2
    )


@profile_stage("quadratic_translation_profiles_cpu")
def optimize_translation_profiles_cpu(
    particles: PreparedStack,
    references: PreparedStack,
    particle_index: int,
    reference_index: int,
    angles: np.ndarray,
    mirror: bool,
    config: AlignmentConfig,
    temperature: float,
    pose_prior: PoseSet,
) -> TranslationProfiles:
    """Optimize integer and subpixel translations for a batch of CPU angles."""
    angle_values = np.asarray(angles, dtype=np.float64)
    center_y = float(pose_prior.shift_y_px[particle_index])
    center_x = float(pose_prior.shift_x_px[particle_index])
    shifts_y = _integer_window(center_y, config.local_shift_range)
    shifts_x = _integer_window(center_x, config.local_shift_range)
    weights = score_weights_for_particle(particles, particle_index)
    reference = references.fourier[reference_index]
    reference_norm = reference_norms_for_particle(
        particles, references, particle_index
    )[reference_index]
    size = int(reference.shape[0])

    output_y = np.empty(len(angle_values), dtype=np.float64)
    output_x = np.empty(len(angle_values), dtype=np.float64)
    output_score = np.empty(len(angle_values), dtype=np.float64)
    output_objective = np.empty(len(angle_values), dtype=np.float64)
    fit_attempt_count = 0
    fit_accept_count = 0
    boundary_hit_count = 0
    flat_or_convex_count = 0
    out_of_bounds_count = 0
    exact_reject_count = 0
    objective_gain = 0.0

    for index, angle in enumerate(angle_values):
        rotated = transform_fourier_cpu(
            particles.fourier[particle_index],
            angle_deg=float(angle),
            mirror=bool(mirror),
        )
        cross_power = weights * rotated * np.conj(reference)
        correlation = profile_call(
            "correlation_map_ifft", np.fft.ifft2, cross_power
        ).real * (size * size)
        rotated_norm = float(np.sum(weights * np.abs(rotated) ** 2))
        denominator = max(np.sqrt(rotated_norm * float(reference_norm)), 1e-12)
        score_surface = correlation[
            np.ix_((-shifts_y) % size, (-shifts_x) % size)
        ] / denominator
        objective_surface = score_surface / float(temperature)
        objective_surface -= 0.5 * (
            ((shifts_y.astype(np.float64) - center_y) / config.pose_shift_sigma)
            ** 2
        )[:, None]
        objective_surface -= 0.5 * (
            ((shifts_x.astype(np.float64) - center_x) / config.pose_shift_sigma)
            ** 2
        )[None, :]
        flat_index = int(np.argmax(objective_surface))
        row, column = np.unravel_index(flat_index, objective_surface.shape)
        integer_y = float(shifts_y[row])
        integer_x = float(shifts_x[column])
        integer_score = float(score_surface[row, column])
        integer_objective = float(objective_surface[row, column])

        fit_y = integer_y
        fit_x = integer_x
        proposed_axes = 0
        for axis, position, length in (
            ("y", row, len(shifts_y)),
            ("x", column, len(shifts_x)),
        ):
            if position == 0 or position == length - 1:
                boundary_hit_count += 1
                continue
            fit_attempt_count += 1
            if axis == "y":
                values = objective_surface[row - 1 : row + 2, column]
            else:
                values = objective_surface[row, column - 1 : column + 2]
            offset, reason = quadratic_vertex_offset(*values)
            if reason == "accepted":
                proposed_axes += 1
                if axis == "y":
                    fit_y += offset
                else:
                    fit_x += offset
            elif reason == "out_of_bounds":
                out_of_bounds_count += 1
            else:
                flat_or_convex_count += 1

        accepted_score = integer_score
        accepted_objective = integer_objective
        if proposed_axes:
            shifted = _shift_fourier(rotated, fit_y, fit_x)
            fitted_score = _fourier_ncc(
                shifted, reference, weights, float(reference_norm)
            )
            fitted_objective = _translation_objective(
                fitted_score,
                fit_y,
                fit_x,
                center_y=center_y,
                center_x=center_x,
                temperature=temperature,
                sigma=float(config.pose_shift_sigma),
            )
            if fitted_objective > integer_objective + 1e-12:
                accepted_score = fitted_score
                accepted_objective = fitted_objective
                fit_accept_count += proposed_axes
                objective_gain += fitted_objective - integer_objective
            else:
                fit_y = integer_y
                fit_x = integer_x
                exact_reject_count += proposed_axes

        output_y[index] = fit_y
        output_x[index] = fit_x
        output_score[index] = accepted_score
        output_objective[index] = accepted_objective

    profile_count("quadratic_correlation_map_iffts", len(angle_values))
    profile_count("quadratic_screened_angles", len(angle_values))
    return TranslationProfiles(
        angle_deg=angle_values,
        shift_y_px=output_y,
        shift_x_px=output_x,
        score=output_score,
        objective=output_objective,
        fit_attempt_count=fit_attempt_count,
        fit_accept_count=fit_accept_count,
        boundary_hit_count=boundary_hit_count,
        flat_or_convex_count=flat_or_convex_count,
        out_of_bounds_count=out_of_bounds_count,
        exact_reject_count=exact_reject_count,
        objective_gain=objective_gain,
        ifft_count=len(angle_values),
    )


def _add_profile_stats(target: dict[str, float], values: TranslationProfiles) -> None:
    target["translation_fit_attempt_count"] += values.fit_attempt_count
    target["translation_fit_accept_count"] += values.fit_accept_count
    target["boundary_hit_count"] += values.boundary_hit_count
    target["flat_or_convex_count"] += values.flat_or_convex_count
    target["out_of_bounds_count"] += values.out_of_bounds_count
    target["exact_reject_count"] += values.exact_reject_count
    target["objective_gain"] += values.objective_gain
    target["ifft_count"] += values.ifft_count


def _normalized_posterior(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    probabilities = np.exp(values - np.max(values))
    probabilities /= np.sum(probabilities)
    return probabilities


@profile_stage("quadratic_controller")
def infer_quadratic_candidates(
    particles: PreparedStack,
    references: PreparedStack,
    config: AlignmentConfig,
    class_priors: np.ndarray,
    temperature: float,
    initial_poses: PoseSet | None,
    *,
    translation_profiler: TranslationProfiler = optimize_translation_profiles_cpu,
    rescue_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Infer continuous local modes and retain their full soft posterior."""
    if initial_poses is None:
        raise ValueError("quadratic_refine search requires initial_poses")
    particle_count = len(particles.spatial)
    if len(initial_poses) != particle_count:
        raise ValueError("initial_poses and particles must contain the same count")
    if rescue_mask is not None and np.any(np.asarray(rescue_mask, dtype=np.bool_)):
        raise ValueError("quadratic_refine does not support rescue searches")

    result = _empty_candidate_values(particle_count, config.top_l)
    diagnostics = {
        name: np.zeros(particle_count, dtype=dtype)
        for name, dtype in (
            ("screened_angle_count", np.int32),
            ("active_reference_count", np.int32),
            ("angular_mode_count", np.int32),
            ("posterior_support_count", np.int32),
            ("translation_fit_attempt_count", np.int32),
            ("translation_fit_accept_count", np.int32),
            ("angle_fit_attempt_count", np.int32),
            ("angle_fit_accept_count", np.int32),
            ("boundary_hit_count", np.int32),
            ("flat_or_convex_count", np.int32),
            ("out_of_bounds_count", np.int32),
            ("exact_reject_count", np.int32),
            ("ifft_count", np.int32),
            ("objective_gain", np.float32),
        )
    }
    mstep_values: dict[str, list[np.ndarray]] = {
        name: [] for name in STANDARD_CANDIDATE_FIELDS
    }
    mstep_offsets = [0]
    full_pose_entropy = np.zeros(particle_count, dtype=np.float32)
    full_map_posterior = np.zeros(particle_count, dtype=np.float32)
    angle_offsets = symmetric_grid_offsets(
        config.local_angle_range, config.coarse_angle_step
    )

    for particle_index in range(particle_count):
        center_angle = float(initial_poses.angle_deg[particle_index])
        active_references = np.flatnonzero(class_priors[particle_index] > 0.0)
        if len(active_references) == 0:
            raise ValueError(f"particle {particle_index} has no active reference")
        diagnostics["active_reference_count"][particle_index] = len(
            active_references
        )
        mirrors = (
            (False, True)
            if config.mirror_search
            else (bool(initial_poses.mirror[particle_index]),)
        )
        modes: list[tuple[float, float, int, float, float, float, bool, int]] = []
        stable_id = 0
        stats = {name: 0.0 for name in (
            "translation_fit_attempt_count",
            "translation_fit_accept_count",
            "boundary_hit_count",
            "flat_or_convex_count",
            "out_of_bounds_count",
            "exact_reject_count",
            "objective_gain",
            "ifft_count",
        )}
        for mirrored in mirrors:
            for reference_index_value in active_references:
                reference_index = int(reference_index_value)
                angles = np.asarray(
                    [_wrap_angle(center_angle + value) for value in angle_offsets],
                    dtype=np.float64,
                )
                profiles = profile_call(
                    "translation_profiles",
                    translation_profiler,
                    particles,
                    references,
                    particle_index,
                    reference_index,
                    angles,
                    bool(mirrored),
                    config,
                    temperature,
                    initial_poses,
                )
                _add_profile_stats(stats, profiles)
                diagnostics["screened_angle_count"][particle_index] += len(angles)
                angle_delta = np.asarray(
                    [_angle_difference(value, center_angle) for value in angles]
                )
                logits = profiles.objective.copy()
                logits -= 0.5 * (angle_delta / config.pose_angle_sigma) ** 2
                logits += np.log(
                    max(
                        float(class_priors[particle_index, reference_index]),
                        1e-30,
                    )
                )
                local_maxima = angular_local_maxima(logits)
                diagnostics["angular_mode_count"][particle_index] += len(
                    local_maxima
                )
                accepted_angles = angles.copy()
                accepted_y = profiles.shift_y_px.copy()
                accepted_x = profiles.shift_x_px.copy()
                accepted_scores = profiles.score.copy()
                accepted_objectives = logits.copy()
                for maximum_index_value in local_maxima:
                    maximum_index = int(maximum_index_value)
                    accepted_angle = float(accepted_angles[maximum_index])
                    accepted_shift_y = float(accepted_y[maximum_index])
                    accepted_shift_x = float(accepted_x[maximum_index])
                    accepted_score = float(accepted_scores[maximum_index])
                    accepted_objective = float(
                        accepted_objectives[maximum_index]
                    )
                    if maximum_index == 0 or maximum_index == len(angles) - 1:
                        stats["boundary_hit_count"] += 1
                    else:
                        diagnostics["angle_fit_attempt_count"][particle_index] += 1
                        offset, reason = quadratic_vertex_offset(
                            logits[maximum_index - 1],
                            logits[maximum_index],
                            logits[maximum_index + 1],
                            step=config.coarse_angle_step,
                        )
                        if reason == "accepted":
                            fitted_angle = _wrap_angle(accepted_angle + offset)
                            fitted = profile_call(
                                "angle_exact_rescoring",
                                translation_profiler,
                                particles,
                                references,
                                particle_index,
                                reference_index,
                                np.asarray([fitted_angle], dtype=np.float64),
                                bool(mirrored),
                                config,
                                temperature,
                                initial_poses,
                            )
                            _add_profile_stats(stats, fitted)
                            fitted_objective = float(fitted.objective[0])
                            fitted_objective -= 0.5 * (
                                _angle_difference(fitted_angle, center_angle)
                                / config.pose_angle_sigma
                            ) ** 2
                            fitted_objective += np.log(
                                max(
                                    float(
                                        class_priors[
                                            particle_index, reference_index
                                        ]
                                    ),
                                    1e-30,
                                )
                            )
                            if fitted_objective > accepted_objective + 1e-12:
                                stats["objective_gain"] += (
                                    fitted_objective - accepted_objective
                                )
                                accepted_angle = fitted_angle
                                accepted_shift_y = float(fitted.shift_y_px[0])
                                accepted_shift_x = float(fitted.shift_x_px[0])
                                accepted_score = float(fitted.score[0])
                                accepted_objective = fitted_objective
                                diagnostics["angle_fit_accept_count"][particle_index] += 1
                            else:
                                stats["exact_reject_count"] += 1
                        elif reason == "out_of_bounds":
                            stats["out_of_bounds_count"] += 1
                        else:
                            stats["flat_or_convex_count"] += 1
                    accepted_angles[maximum_index] = accepted_angle
                    accepted_y[maximum_index] = accepted_shift_y
                    accepted_x[maximum_index] = accepted_shift_x
                    accepted_scores[maximum_index] = accepted_score
                    accepted_objectives[maximum_index] = accepted_objective

                # Retain the complete angle profile for the internal soft
                # posterior.  A continuous fitted pose replaces its grid
                # maximum, while the remaining samples preserve uncertainty
                # for responsibilities and the Fourier M-step.
                for support_index in range(len(angles)):
                    modes.append(
                        (
                            float(accepted_objectives[support_index]),
                            float(accepted_scores[support_index]),
                            reference_index,
                            float(accepted_angles[support_index]),
                            float(accepted_y[support_index]),
                            float(accepted_x[support_index]),
                            bool(mirrored),
                            stable_id,
                        )
                    )
                    stable_id += 1

        if not modes:
            raise ValueError(f"particle {particle_index} produced no pose modes")
        modes.sort(key=lambda item: (-item[0], item[7]))
        logits = np.asarray([item[0] for item in modes], dtype=np.float64)
        posterior = _normalized_posterior(logits)
        full_pose_entropy[particle_index] = float(
            -np.sum(posterior * np.log(np.maximum(posterior, 1e-30)))
        )
        full_map_posterior[particle_index] = float(np.max(posterior))
        diagnostics["posterior_support_count"][particle_index] = len(modes)
        for name in stats:
            diagnostics[name][particle_index] = stats[name]

        fields = {
            "reference_index": np.asarray([item[2] for item in modes], dtype=np.int32),
            "angle_deg": np.asarray([item[3] for item in modes], dtype=np.float32),
            "shift_y_px": np.asarray([item[4] for item in modes], dtype=np.float32),
            "shift_x_px": np.asarray([item[5] for item in modes], dtype=np.float32),
            "mirror": np.asarray([item[6] for item in modes], dtype=np.bool_),
            "score": np.asarray([item[1] for item in modes], dtype=np.float32),
            "posterior": posterior.astype(np.float32),
        }
        for name in STANDARD_CANDIDATE_FIELDS:
            mstep_values[name].append(fields[name])
        mstep_offsets.append(mstep_offsets[-1] + len(modes))

        retained_count = min(config.top_l, len(modes))
        retained_mass = float(np.sum(posterior[:retained_count]))
        for name in STANDARD_CANDIDATE_FIELDS:
            result[name][particle_index, :retained_count] = fields[name][
                :retained_count
            ]
        result["posterior"][particle_index, :retained_count] = (
            posterior[:retained_count] / retained_mass
        )
        for name in (
            "reference_index",
            "angle_deg",
            "shift_y_px",
            "shift_x_px",
            "mirror",
        ):
            if retained_count < config.top_l:
                result[name][particle_index, retained_count:] = fields[name][0]

    result["_mstep_particle_offsets"] = np.asarray(mstep_offsets, dtype=np.int64)
    for name, chunks in mstep_values.items():
        result[f"_mstep_{name}"] = np.concatenate(chunks)
    for name, values in diagnostics.items():
        result[f"_quadratic_{name}"] = values
    result["_full_pose_entropy"] = full_pose_entropy
    result["_full_map_posterior"] = full_map_posterior
    result["_rescue_next"] = np.zeros(particle_count, dtype=np.bool_)
    return result


def infer_quadratic_candidates_cpu(*args, **kwargs) -> dict[str, np.ndarray]:
    return infer_quadratic_candidates(
        *args,
        **kwargs,
        translation_profiler=optimize_translation_profiles_cpu,
    )
