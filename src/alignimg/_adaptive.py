"""RELION-like adaptive posterior pose search using AlignImg's NCC model."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._fourier import (
    PreparedStack,
    _angle_difference,
    _fourier_ncc,
    _shift_fourier,
    infer_top_candidates,
    score_weights_for_particle,
)
from ._transform import transform_image
from ._profiling import count as profile_count, profile_call, profile_stage
from .models import AlignmentConfig, PoseSet


CELL_DTYPE = np.dtype(
    [
        ("reference_index", np.int32),
        ("angle_deg", np.float64),
        ("shift_y_px", np.float64),
        ("shift_x_px", np.float64),
        ("mirror", np.bool_),
    ]
)

STANDARD_CANDIDATE_FIELDS = (
    "reference_index",
    "angle_deg",
    "shift_y_px",
    "shift_x_px",
    "mirror",
    "score",
    "posterior",
)

CandidateScorer = Callable[[PreparedStack, PreparedStack, int, np.ndarray], np.ndarray]
CandidateBatchScorer = Callable[
    [PreparedStack, PreparedStack, np.ndarray, list[np.ndarray]],
    list[np.ndarray],
]

MAX_ADAPTIVE_PARTICLES_PER_BATCH = 32


def _wrap_angle(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def symmetric_grid_offsets(radius: float, step: float) -> np.ndarray:
    """Return a uniform center-containing grid that does not exceed radius."""
    radius = float(radius)
    step = float(step)
    if radius < 0.0 or step <= 0.0:
        raise ValueError("grid radius must be non-negative and step must be positive")
    count = int(np.floor(radius / step + 1e-9))
    return (np.arange(-count, count + 1, dtype=np.float64) * step).astype(np.float64)


def fine_offsets(coarse_step: float, oversampling_order: int) -> np.ndarray:
    """Return center-inclusive offsets spanning one coarse cell."""
    coarse_step = float(coarse_step)
    order = int(oversampling_order)
    if coarse_step <= 0.0 or order < 0:
        raise ValueError(
            "coarse_step must be positive and oversampling_order non-negative"
        )
    if order == 0:
        return np.asarray([0.0], dtype=np.float64)
    factor = 2**order
    fine_step = coarse_step / factor
    return np.arange(-factor // 2, factor // 2 + 1, dtype=np.float64) * fine_step


def select_posterior_mass(
    posterior: np.ndarray,
    *,
    fraction: float,
    max_cells: int | None,
) -> tuple[np.ndarray, float, bool]:
    """Select the smallest stable-ranked set covering a posterior fraction."""
    values = np.asarray(posterior, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("posterior must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("posterior must contain finite non-negative values")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("posterior must contain positive mass")
    normalized = values / total
    order = np.argsort(-normalized, kind="stable")
    cumulative = np.cumsum(normalized[order])
    required = int(np.searchsorted(cumulative, float(fraction), side="left")) + 1
    required = min(max(1, required), len(order))
    capped = False
    if max_cells is not None and required > int(max_cells):
        required = int(max_cells)
        capped = True
    selected = order[:required].astype(np.int64)
    return selected, float(normalized[selected].sum()), capped


def schedule_uncertain_rescue(
    *,
    boundary_hit: np.ndarray,
    normalized_entropy: np.ndarray,
    map_posterior: np.ndarray,
    eligible: np.ndarray,
    group_index: np.ndarray,
    config: AlignmentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Schedule a bounded, deterministic set of uncertain particles per component."""
    boundary = np.asarray(boundary_hit, dtype=np.bool_)
    entropy = np.asarray(normalized_entropy, dtype=np.float64)
    map_probability = np.asarray(map_posterior, dtype=np.float64)
    allowed = np.asarray(eligible, dtype=np.bool_)
    groups = np.asarray(group_index)
    if (
        not (
            boundary.shape
            == entropy.shape
            == map_probability.shape
            == allowed.shape
            == groups.shape
        )
        or boundary.ndim != 1
    ):
        raise ValueError("rescue signals must be one-dimensional arrays of equal shape")
    if not np.issubdtype(groups.dtype, np.integer):
        raise ValueError("group_index must contain integers")
    if not np.all(np.isfinite(entropy)) or not np.all(np.isfinite(map_probability)):
        raise ValueError("rescue uncertainty signals must be finite")

    entropy_trigger = np.zeros(len(boundary), dtype=np.bool_)
    if config.rescue_normalized_entropy_threshold is not None:
        entropy_trigger = entropy >= float(config.rescue_normalized_entropy_threshold)
    map_trigger = np.zeros(len(boundary), dtype=np.bool_)
    if config.rescue_map_posterior_threshold is not None:
        map_trigger = map_probability <= float(config.rescue_map_posterior_threshold)

    scheduled = np.zeros(len(boundary), dtype=np.bool_)
    candidates = allowed & (boundary | entropy_trigger | map_trigger)
    for group in np.unique(groups):
        group_mask = groups == group
        selected = np.flatnonzero(candidates & group_mask)
        limit = int(
            np.ceil(float(config.rescue_max_fraction) * int(np.sum(group_mask)))
        )
        if len(selected) > limit:
            boundary_rank = boundary[selected].astype(np.int8)
            entropy_rank = entropy[selected]
            map_rank = map_probability[selected]
            order = np.lexsort((selected, map_rank, -entropy_rank, -boundary_rank))
            selected = selected[order[:limit]]
        scheduled[selected] = bool(config.rescue_uncertain_particles)
    return scheduled, entropy_trigger & allowed, map_trigger & allowed


def oversample_coarse_cells(
    cells: np.ndarray,
    *,
    center_angle_deg: float,
    center_shift_y_px: float,
    center_shift_x_px: float,
    coarse_angle_step: float,
    coarse_shift_step: float,
    local_angle_range: float,
    local_shift_range: float,
    oversampling_order: int,
) -> np.ndarray:
    """Expand selected coarse cells into a deduplicated center-inclusive fine grid."""
    values = np.asarray(cells, dtype=CELL_DTYPE)
    angle_offsets = fine_offsets(coarse_angle_step, oversampling_order)
    shift_offsets = fine_offsets(coarse_shift_step, oversampling_order)
    output: list[tuple[int, float, float, float, bool]] = []
    seen: set[tuple[int, float, float, float, bool]] = set()
    tolerance = 1e-6
    for cell in values:
        for angle_offset in angle_offsets:
            angle = _wrap_angle(float(cell["angle_deg"]) + angle_offset)
            if (
                abs(_angle_difference(angle, center_angle_deg))
                > float(local_angle_range) + tolerance
            ):
                continue
            for shift_y_offset in shift_offsets:
                shift_y = float(cell["shift_y_px"]) + shift_y_offset
                if (
                    abs(shift_y - float(center_shift_y_px))
                    > float(local_shift_range) + tolerance
                ):
                    continue
                for shift_x_offset in shift_offsets:
                    shift_x = float(cell["shift_x_px"]) + shift_x_offset
                    if (
                        abs(shift_x - float(center_shift_x_px))
                        > float(local_shift_range) + tolerance
                    ):
                        continue
                    key = (
                        int(cell["reference_index"]),
                        round(angle, 7),
                        round(shift_y, 7),
                        round(shift_x, 7),
                        bool(cell["mirror"]),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    output.append(
                        (
                            key[0],
                            angle,
                            shift_y,
                            shift_x,
                            key[4],
                        )
                    )
    if not output:
        raise ValueError("oversampling produced no fine pose candidates")
    return np.asarray(output, dtype=CELL_DTYPE)


def score_candidates_cpu(
    particles: PreparedStack,
    references: PreparedStack,
    particle_index: int,
    cells: np.ndarray,
) -> np.ndarray:
    """Authoritative Fourier-NCC scorer for one particle's flat pose buffer."""
    values = np.asarray(cells, dtype=CELL_DTYPE)
    score_weights = score_weights_for_particle(particles, particle_index)
    from ._fourier import reference_norms_for_particle

    reference_norms = reference_norms_for_particle(particles, references, particle_index)
    scores = np.empty(len(values), dtype=np.float32)
    rotated_cache: dict[tuple[float, bool], np.ndarray] = {}
    for index, cell in enumerate(values):
        key = (round(float(cell["angle_deg"]), 7), bool(cell["mirror"]))
        rotated = rotated_cache.get(key)
        if rotated is None:
            spatial = transform_image(
                particles.spatial[particle_index],
                angle_deg=float(cell["angle_deg"]),
                mirror=bool(cell["mirror"]),
            )
            rotated = np.fft.fft2(spatial).astype(np.complex64)
            rotated_cache[key] = rotated
        shifted = _shift_fourier(
            rotated,
            float(cell["shift_y_px"]),
            float(cell["shift_x_px"]),
        )
        scores[index] = _fourier_ncc(
            shifted,
            references.fourier[int(cell["reference_index"])],
            score_weights,
            reference_norms[int(cell["reference_index"])],
        )
    return scores


def score_candidates_fourier_cpu(
    particles: PreparedStack,
    references: PreparedStack,
    particle_index: int,
    cells: np.ndarray,
) -> np.ndarray:
    """Score adaptive pose cells directly from the cached particle DFT."""
    from ._fourier_native import score_fourier_candidates_cpu
    from ._fourier import reference_norms_for_particle

    return score_fourier_candidates_cpu(
        particles.fourier[particle_index],
        references.fourier,
        cells,
        score_weights_for_particle(particles, particle_index),
        reference_norms=reference_norms_for_particle(particles, references, particle_index),
    )


def _pose_log_posterior(
    scores: np.ndarray,
    cells: np.ndarray,
    *,
    class_prior: np.ndarray,
    pose_prior: PoseSet,
    particle_index: int,
    temperature: float,
    config: AlignmentConfig,
) -> np.ndarray:
    values = np.asarray(cells, dtype=CELL_DTYPE)
    reference_indices = values["reference_index"].astype(np.int64)
    result = np.asarray(scores, dtype=np.float64) / float(temperature)
    result += np.log(
        np.maximum(np.asarray(class_prior, dtype=np.float64)[reference_indices], 1e-30)
    )
    angle_delta = np.asarray(
        [
            _angle_difference(value, pose_prior.angle_deg[particle_index])
            for value in values["angle_deg"]
        ],
        dtype=np.float64,
    )
    shift_y_delta = values["shift_y_px"].astype(np.float64) - float(
        pose_prior.shift_y_px[particle_index]
    )
    shift_x_delta = values["shift_x_px"].astype(np.float64) - float(
        pose_prior.shift_x_px[particle_index]
    )
    result -= 0.5 * (angle_delta / float(config.pose_angle_sigma)) ** 2
    result -= 0.5 * (
        (shift_y_delta / float(config.pose_shift_sigma)) ** 2
        + (shift_x_delta / float(config.pose_shift_sigma)) ** 2
    )
    return result


def _normalized_posterior(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    probabilities = np.exp(values - np.max(values))
    probabilities /= np.sum(probabilities)
    return probabilities


def _coarse_cells_for_particle(
    references: PreparedStack,
    class_prior: np.ndarray,
    pose_prior: PoseSet,
    particle_index: int,
    config: AlignmentConfig,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    angle_offsets = symmetric_grid_offsets(
        config.local_angle_range, config.coarse_angle_step
    )
    shift_offsets = symmetric_grid_offsets(
        config.local_shift_range, config.coarse_shift_step
    )
    center_angle = float(pose_prior.angle_deg[particle_index])
    center_y = float(pose_prior.shift_y_px[particle_index])
    center_x = float(pose_prior.shift_x_px[particle_index])
    mirrors = (
        (False, True)
        if config.mirror_search
        else (bool(pose_prior.mirror[particle_index]),)
    )
    output = []
    for mirrored in mirrors:
        for reference_index in range(len(references.spatial)):
            if float(class_prior[reference_index]) <= 0.0:
                continue
            for angle_offset in angle_offsets:
                angle = _wrap_angle(center_angle + angle_offset)
                for shift_y_offset in shift_offsets:
                    for shift_x_offset in shift_offsets:
                        output.append(
                            (
                                reference_index,
                                angle,
                                center_y + shift_y_offset,
                                center_x + shift_x_offset,
                                mirrored,
                            )
                        )
    if not output:
        raise ValueError(f"particle {particle_index} has no allowed coarse cells")
    effective_boundary = (
        float(np.max(np.abs(angle_offsets))),
        float(np.max(np.abs(shift_offsets))),
        float(np.max(np.abs(shift_offsets))),
    )
    return np.asarray(output, dtype=CELL_DTYPE), effective_boundary


def _map_hits_boundary(
    cell: np.void,
    *,
    pose_prior: PoseSet,
    particle_index: int,
    effective_boundary: tuple[float, float, float],
) -> bool:
    angle_boundary, shift_y_boundary, shift_x_boundary = effective_boundary
    tolerance = 1e-5
    angle_hit = angle_boundary > 0.0 and (
        abs(_angle_difference(cell["angle_deg"], pose_prior.angle_deg[particle_index]))
        >= angle_boundary - tolerance
    )
    shift_y_hit = shift_y_boundary > 0.0 and (
        abs(float(cell["shift_y_px"]) - float(pose_prior.shift_y_px[particle_index]))
        >= shift_y_boundary - tolerance
    )
    shift_x_hit = shift_x_boundary > 0.0 and (
        abs(float(cell["shift_x_px"]) - float(pose_prior.shift_x_px[particle_index]))
        >= shift_x_boundary - tolerance
    )
    return bool(angle_hit or shift_y_hit or shift_x_hit)


def _empty_candidate_values(particle_count: int, top_l: int) -> dict[str, np.ndarray]:
    shape = (particle_count, top_l)
    return {
        "reference_index": np.zeros(shape, dtype=np.int32),
        "angle_deg": np.zeros(shape, dtype=np.float32),
        "shift_y_px": np.zeros(shape, dtype=np.float32),
        "shift_x_px": np.zeros(shape, dtype=np.float32),
        "mirror": np.zeros(shape, dtype=np.bool_),
        "score": np.full(shape, -np.inf, dtype=np.float32),
        "posterior": np.zeros(shape, dtype=np.float32),
        "_coarse_cell_count": np.zeros(particle_count, dtype=np.int32),
        "_selected_coarse_cell_count": np.zeros(particle_count, dtype=np.int32),
        "_selected_coarse_mass": np.zeros(particle_count, dtype=np.float32),
        "_fine_candidate_count": np.zeros(particle_count, dtype=np.int32),
        "_retained_fine_mass": np.zeros(particle_count, dtype=np.float32),
        "_adaptive_cap_hit": np.zeros(particle_count, dtype=np.bool_),
        "_boundary_hit": np.zeros(particle_count, dtype=np.bool_),
        "_normalized_pose_entropy": np.zeros(particle_count, dtype=np.float32),
        "_rescue_entropy_trigger": np.zeros(particle_count, dtype=np.bool_),
        "_rescue_map_trigger": np.zeros(particle_count, dtype=np.bool_),
        "_rescue_used": np.zeros(particle_count, dtype=np.bool_),
        "_rescue_accepted": np.zeros(particle_count, dtype=np.bool_),
        "_rescue_score_gain": np.full(particle_count, np.nan, dtype=np.float32),
        "_rescue_next": np.zeros(particle_count, dtype=np.bool_),
    }


@profile_stage("adaptive_controller")
def infer_adaptive_candidates(
    particles: PreparedStack,
    references: PreparedStack,
    config: AlignmentConfig,
    class_priors: np.ndarray,
    temperature: float,
    initial_poses: PoseSet | None,
    *,
    score_candidates: CandidateScorer = score_candidates_cpu,
    score_candidate_batches: CandidateBatchScorer | None = None,
    rescue_mask: np.ndarray | None = None,
    proposal_fallback=None,
) -> dict[str, np.ndarray]:
    """Infer adaptive coarse-to-fine candidates with a shared flat-buffer controller."""
    if initial_poses is None:
        raise ValueError("adaptive_posterior search requires initial_poses")
    particle_count = len(particles.spatial)
    if len(initial_poses) != particle_count:
        raise ValueError("initial_poses and particles must contain the same count")
    rescue = (
        np.zeros(particle_count, dtype=np.bool_)
        if rescue_mask is None
        else np.asarray(rescue_mask, dtype=np.bool_)
    )
    if rescue.shape != (particle_count,):
        raise ValueError("rescue_mask must have shape (N,)")
    fallback_values = None
    rescue_position = np.full(particle_count, -1, dtype=np.int64)
    if np.any(rescue):
        fallback = proposal_fallback or infer_top_candidates
        rescue_indices = np.flatnonzero(rescue)
        rescue_position[rescue_indices] = np.arange(len(rescue_indices))
        rescue_profiles = particles.score_weight_profiles
        if len(rescue_profiles) != 1:
            rescue_profiles = rescue_profiles[rescue_indices]
        rescue_particles = PreparedStack(
            particles.spatial[rescue_indices],
            particles.fourier[rescue_indices],
            particles.polar[rescue_indices],
            particles.mask,
            particles.frequency_mask,
            rescue_profiles,
            particles.score_weight_bins,
        )
        fallback_values = fallback(
            rescue_particles,
            references,
            config,
            class_priors[rescue_indices],
            temperature,
            None,
        )

    result = _empty_candidate_values(particle_count, config.top_l)
    mstep_values: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "reference_index",
            "angle_deg",
            "shift_y_px",
            "shift_x_px",
            "mirror",
            "score",
            "posterior",
        )
    }
    mstep_offsets = [0]
    full_pose_entropy = np.zeros(particle_count, dtype=np.float32)
    full_map_posterior = np.zeros(particle_count, dtype=np.float32)
    if score_candidate_batches is None:
        particle_batch_size = 1

        def score_batches(
            particle_values,
            reference_values,
            particle_indices,
            cell_batches,
        ):
            return [
                score_candidates(
                    particle_values,
                    reference_values,
                    int(particle_index),
                    cells,
                )
                for particle_index, cells in zip(
                    particle_indices, cell_batches, strict=True
                )
            ]

    else:
        particle_batch_size = min(
            particle_count,
            (
                MAX_ADAPTIVE_PARTICLES_PER_BATCH
                if config.batch_size is None
                else int(config.batch_size)
            ),
            MAX_ADAPTIVE_PARTICLES_PER_BATCH,
        )
        score_batches = score_candidate_batches

    for particle_start in range(0, particle_count, particle_batch_size):
        particle_indices = np.arange(
            particle_start,
            min(particle_start + particle_batch_size, particle_count),
            dtype=np.int32,
        )
        profile_count("adaptive_particle_batches")
        profile_count("adaptive_batched_particles", len(particle_indices))
        coarse_batches = []
        effective_boundaries = []
        for particle_index in particle_indices:
            coarse, effective_boundary = _coarse_cells_for_particle(
                references,
                class_priors[particle_index],
                initial_poses,
                int(particle_index),
                config,
            )
            profile_count("coarse_candidates", len(coarse))
            coarse_batches.append(coarse)
            effective_boundaries.append(effective_boundary)
        coarse_score_batches = profile_call(
            "coarse_scoring",
            score_batches,
            particles,
            references,
            particle_indices,
            coarse_batches,
        )
        if len(coarse_score_batches) != len(coarse_batches):
            raise ValueError("coarse batch scorer returned the wrong number of arrays")

        fine_batches = []
        for local_index, particle_index_value in enumerate(particle_indices):
            particle_index = int(particle_index_value)
            coarse = coarse_batches[local_index]
            coarse_scores = np.asarray(
                coarse_score_batches[local_index], dtype=np.float32
            )
            if coarse_scores.shape != (len(coarse),):
                raise ValueError("coarse batch scorer returned an invalid score shape")
            coarse_logits = _pose_log_posterior(
                coarse_scores,
                coarse,
                class_prior=class_priors[particle_index],
                pose_prior=initial_poses,
                particle_index=particle_index,
                temperature=temperature,
                config=config,
            )
            coarse_posterior = _normalized_posterior(coarse_logits)
            selected_indices, selected_mass, cap_hit = select_posterior_mass(
                coarse_posterior,
                fraction=config.adaptive_fraction,
                max_cells=config.max_adaptive_cells,
            )
            coarse_map = int(np.argmax(coarse_posterior))
            boundary_hit = _map_hits_boundary(
                coarse[coarse_map],
                pose_prior=initial_poses,
                particle_index=particle_index,
                effective_boundary=effective_boundaries[local_index],
            )
            fine = oversample_coarse_cells(
                coarse[selected_indices],
                center_angle_deg=float(initial_poses.angle_deg[particle_index]),
                center_shift_y_px=float(initial_poses.shift_y_px[particle_index]),
                center_shift_x_px=float(initial_poses.shift_x_px[particle_index]),
                coarse_angle_step=config.coarse_angle_step,
                coarse_shift_step=config.coarse_shift_step,
                local_angle_range=config.local_angle_range,
                local_shift_range=config.local_shift_range,
                oversampling_order=config.oversampling_order,
            )
            profile_count("fine_candidates", len(fine))
            fine_batches.append(fine)
            result["_coarse_cell_count"][particle_index] = len(coarse)
            result["_selected_coarse_cell_count"][particle_index] = len(
                selected_indices
            )
            result["_selected_coarse_mass"][particle_index] = selected_mass
            result["_fine_candidate_count"][particle_index] = len(fine)
            result["_adaptive_cap_hit"][particle_index] = cap_hit
            result["_boundary_hit"][particle_index] = boundary_hit

        fine_score_batches = profile_call(
            "fine_scoring",
            score_batches,
            particles,
            references,
            particle_indices,
            fine_batches,
        )
        if len(fine_score_batches) != len(fine_batches):
            raise ValueError("fine batch scorer returned the wrong number of arrays")

        for local_index, particle_index_value in enumerate(particle_indices):
            particle_index = int(particle_index_value)
            fine = fine_batches[local_index]
            fine_scores = np.asarray(fine_score_batches[local_index], dtype=np.float32)
            if fine_scores.shape != (len(fine),):
                raise ValueError("fine batch scorer returned an invalid score shape")
            fine_logits = _pose_log_posterior(
                fine_scores,
                fine,
                class_prior=class_priors[particle_index],
                pose_prior=initial_poses,
                particle_index=particle_index,
                temperature=temperature,
                config=config,
            )
            fine_posterior = _normalized_posterior(fine_logits)
            full_pose_entropy[particle_index] = float(
                -np.sum(
                    fine_posterior * np.log(np.maximum(fine_posterior, 1e-30))
                )
            )
            full_map_posterior[particle_index] = float(np.max(fine_posterior))

            if rescue[particle_index]:
                assert fallback_values is not None
                fallback_index = int(rescue_position[particle_index])
                keep = fallback_values["posterior"][fallback_index] > 0.0
                fallback_scores = np.asarray(
                    fallback_values["score"][fallback_index][keep], dtype=np.float64
                )
                score_gain = float(np.max(fallback_scores) - np.max(fine_scores))
                accepted = score_gain > float(config.rescue_min_score_improvement)
                result["_rescue_used"][particle_index] = True
                result["_rescue_score_gain"][particle_index] = score_gain
                result["_rescue_accepted"][particle_index] = accepted
                if accepted:
                    for name in STANDARD_CANDIDATE_FIELDS:
                        result[name][particle_index] = fallback_values[name][
                            fallback_index
                        ]
                    for name in mstep_values:
                        mstep_values[name].append(
                            np.asarray(
                                fallback_values[name][fallback_index][keep]
                            ).copy()
                        )
                    rescue_posterior = np.asarray(
                        fallback_values["posterior"][fallback_index][keep],
                        dtype=np.float64,
                    )
                    full_pose_entropy[particle_index] = float(
                        -np.sum(
                            rescue_posterior
                            * np.log(np.maximum(rescue_posterior, 1e-30))
                        )
                    )
                    full_map_posterior[particle_index] = float(
                        np.max(rescue_posterior)
                    )
                    result["_retained_fine_mass"][particle_index] = 1.0
                    result["_normalized_pose_entropy"][particle_index] = float(
                        np.clip(
                            full_pose_entropy[particle_index]
                            / np.log(len(rescue_posterior)),
                            0.0,
                            1.0,
                        )
                        if len(rescue_posterior) > 1
                        else 0.0
                    )
                    mstep_offsets.append(mstep_offsets[-1] + int(np.sum(keep)))
                    continue

            for name in (
                "reference_index",
                "angle_deg",
                "shift_y_px",
                "shift_x_px",
                "mirror",
            ):
                mstep_values[name].append(np.asarray(fine[name]).copy())
            mstep_values["score"].append(
                np.asarray(fine_scores, dtype=np.float32)
            )
            mstep_values["posterior"].append(
                np.asarray(fine_posterior, dtype=np.float32)
            )
            mstep_offsets.append(mstep_offsets[-1] + len(fine))
            order = np.argsort(-fine_posterior, kind="stable")
            retained = order[: min(config.top_l, len(order))]
            retained_mass = float(fine_posterior[retained].sum())
            normalized_retained = fine_posterior[retained] / retained_mass
            chosen = fine[retained]
            count = len(retained)
            for name in (
                "reference_index",
                "angle_deg",
                "shift_y_px",
                "shift_x_px",
                "mirror",
            ):
                result[name][particle_index, :count] = chosen[name]
                if count < config.top_l:
                    result[name][particle_index, count:] = chosen[name][0]
            result["score"][particle_index, :count] = fine_scores[retained]
            result["posterior"][particle_index, :count] = normalized_retained
            result["_retained_fine_mass"][particle_index] = retained_mass
            result["_normalized_pose_entropy"][particle_index] = float(
                np.clip(
                    full_pose_entropy[particle_index] / np.log(len(fine)),
                    0.0,
                    1.0,
                )
                if len(fine) > 1
                else 0.0
            )
    (
        result["_rescue_next"],
        result["_rescue_entropy_trigger"],
        result["_rescue_map_trigger"],
    ) = schedule_uncertain_rescue(
        boundary_hit=result["_boundary_hit"],
        normalized_entropy=result["_normalized_pose_entropy"],
        map_posterior=full_map_posterior,
        eligible=~rescue,
        group_index=result["reference_index"][:, 0],
        config=config,
    )
    result["_mstep_particle_offsets"] = np.asarray(mstep_offsets, dtype=np.int64)
    for name, chunks in mstep_values.items():
        result[f"_mstep_{name}"] = np.concatenate(chunks)
    result["_full_pose_entropy"] = full_pose_entropy
    result["_full_map_posterior"] = full_map_posterior
    return result


def infer_adaptive_candidates_cpu(
    particles: PreparedStack,
    references: PreparedStack,
    config: AlignmentConfig,
    class_priors: np.ndarray,
    temperature: float,
    initial_poses: PoseSet | None,
    **kwargs,
) -> dict[str, np.ndarray]:
    scorer = (
        score_candidates_fourier_cpu
        if config.candidate_scoring == "fourier"
        else score_candidates_cpu
    )
    return infer_adaptive_candidates(
        particles,
        references,
        config,
        class_priors,
        temperature,
        initial_poses,
        **kwargs,
        score_candidates=scorer,
        proposal_fallback=infer_top_candidates,
    )
