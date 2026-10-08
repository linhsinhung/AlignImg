"""Allocation inventory for the resident/streamed polar batch solver."""

from dataclasses import dataclass, replace

from .memory import MemoryPlan, plan_batch_size


@dataclass(frozen=True)
class PolarMemoryPlan(MemoryPlan):
    particle_storage_policy: str
    target_particle_batch: int
    live_cache_bytes: int
    workflow_remaining_bytes: int
    current_available_bytes: int
    spatial_cache_bytes: int
    fixed_components: dict
    item_components: dict

    @property
    def estimated_peak_bytes(self):
        """New polar allocations only; live workflow caches are separate."""
        return self.fixed_bytes + self.batch_size * self.bytes_per_item

    def asdict(self):
        return {**super().asdict(), "estimated_peak_bytes": self.estimated_peak_bytes}


def polar_allocation_inventory(size, references, angles, radii, centers, mirrors):
    """Conservative sum of named allocations, not a multiplier on result size.

    Rings/curves are float32, FFTs complex64, peak fitting/priors float64.
    Both open and fixed MRA fit this envelope: the latter gathers a reference
    per particle, but expands winners back to K slots for deterministic ordering.
    Include layout materialization even when CuPy can sometimes use a view.

    cuFFT's documented upper bound is 8 * batch * transform_size complex
    elements (CUDA 12.8 cuFFT, section 2.2.1). Reserve reference, full-batch and
    tail plans: CuPy's plan cache can keep all three alive. These are bounds,
    NOT measured memory peaks or claims that cuFFT normally uses this much.
    https://docs.nvidia.com/cuda/archive/12.8.1/cufft/index.html
    """
    k, a, r, c, m = map(int, (references, angles, radii, centers, mirrors))
    f = a // 2 + 1
    combos, curves = c * m, c * m * k
    fixed = {
        "reference_images_fp32": k * size * size * 4,
        "polar_offsets_fp64": 2 * a * r * 8,
        "reference_sampling_vectors": k * (4 + 8 + 8 + 1),
        "reference_rings_fp32": k * a * r * 4,
        "reference_normalization_scratch": k * (a * r + r + 3) * 4 + r * 12,
        "reference_fft_input_layout_fp32": k * a * r * 4,
        "reference_fft_complex64": k * f * r * 8,
        "reference_conjugate_and_einsum_layout": 2 * k * f * r * 8,
        "reference_cufft_workspace_bound": 8 * k * r * a * 8,
        "small_index_vectors": k * 4 + m + 4 * 8,
    }
    item = {
        "translation_grids_fp64_and_valid": c * (8 + 8 + 1),
        "source_rows_lengths_fixed_reference": 4 + 4 + 8 + 4,
        "priors_active_log_and_log_scratch": k * (8 + 1 + 8 + 8),
        "broadcast_sampling_indices_centers_mirrors": combos * (4 + 8 + 8 + 1),
        "rings_fp32": combos * a * r * 4,
        "normalization_scratch_fp32": combos * (a * r + r + 3) * 4,
        "subject_fft_input_layout_fp32": combos * a * r * 4,
        "subject_fft_complex64": combos * f * r * 8,
        "subject_einsum_layout_complex64": combos * f * r * 8,
        "fixed_reference_gather_conjugate_layout": 3 * f * r * 8 + 4,
        "cross_spectrum_and_ifft_input_complex64": 2 * curves * f * 8,
        "correlation_curves_fp32": curves * a * 4,
        "peak_selector_copy_fp64": curves * a * 8,
        # Neighbors, denominator, offsets, score and their expression scratch.
        "peak_fit_fp64_vectors": curves * 12 * 8,
        "peak_fit_indices_and_predicates": curves * (6 * 4 + 2 * 8 + 8),
        "active_and_expanded_peak_outputs": 2 * curves * (4 + 8 + 8 + 1),
        # Objective, masked objective, ordered copy, runner-up copy; columns
        # and comparison masks for argmax. No full correlation-map download.
        "objective_ordering_and_runner_up": curves * (4 * 8 + 8 + 3),
        "fixed_reference_active_slots": k,
        "winner_state": 8 * 8 + 2 * 4 + 2,
        # Winner/runner-up center, mirror, reference, flat-id arithmetic and
        # gathered vectors (includes temporary integer divisions/products).
        "winner_index_vectors": 32 * 8 + 12 * 4,
        "winner_objective_angle_shift_vectors": 32 * 8 + 8,
        "winner_merge_four_column_pools": 8 * 4 * 8 + 3 * 4,
        "compact_result_and_casts_fp64": (11 + 4) * 8,
        # Full and tail R2C + C2R plans may remain cached simultaneously.
        "subject_and_inverse_cufft_workspace_bounds": 2
        * 8
        * (combos * r + curves)
        * a
        * 8,
    }
    return fixed, item


def plan_polar_storage(
    config,
    size,
    particle_count,
    reference_count,
    angle_samples,
    radial_bins,
    maximum_centers,
    mirror_count,
    *,
    free_bytes,
    total_bytes,
    workflow_budget_bytes=None,
    live_cache_bytes=0,
    force_streaming=False,
    particle_limit=None,
    allow_spatial_cache=False,
    spatial_cache_bytes=0,
):
    fixed, item = polar_allocation_inventory(
        size, reference_count, angle_samples, radial_bins, maximum_centers, mirror_count
    )
    base = plan_batch_size(
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        memory_fraction=config.memory_fraction,
        fixed_bytes=0,
        bytes_per_item=1,
        requested_batch_size=config.batch_size,
    )
    current = base.budget_bytes  # free VRAM already excludes live caches
    remaining = (
        max(0, workflow_budget_bytes - live_cache_bytes)
        if workflow_budget_bytes is not None
        else current
    )
    budget = min(current, remaining)
    target = min(particle_count, config.batch_size or base.automatic_batch_soft_cap)
    batch_cap = min(target, particle_limit) if particle_limit is not None else target
    spatial_bytes = particle_count * size * size * 4
    if spatial_cache_bytes not in (0, spatial_bytes):
        raise ValueError("polar spatial cache size differs from the particle source")
    # Source residency and solver batch size are independent. A retained source
    # is already included in live_cache_bytes/free VRAM; only a first upload is
    # a new fixed allocation. The cuFFT/sampler inventory is unchanged.
    cached_fixed = dict(fixed)
    if not spatial_cache_bytes:
        cached_fixed["new_spatial_cache_fp32"] = spatial_bytes
    cached_fits = sum(cached_fixed.values()) + sum(item.values()) <= budget
    resident_fixed = {
        **fixed,
        "resident_particles_fp32": spatial_bytes,
    }
    resident_fits = sum(resident_fixed.values()) + target * sum(item.values()) <= budget
    if allow_spatial_cache and not force_streaming and cached_fits:
        policy = "cached"
        fixed = cached_fixed
        batch = min(batch_cap, (budget - sum(fixed.values())) // sum(item.values()))
    elif not force_streaming and resident_fits:
        policy = "resident"
        fixed = resident_fixed
        batch = batch_cap
    else:
        policy = "streaming"
        item["streamed_particle_images_fp32"] = size * size * 4
        batch = max(
            1, min(batch_cap, max(0, budget - sum(fixed.values())) // sum(item.values()))
        )
    plan = replace(
        base,
        budget_bytes=budget,
        fixed_bytes=sum(fixed.values()),
        bytes_per_item=sum(item.values()),
        batch_size=batch,
    )
    return PolarMemoryPlan(
        **{key: value for key, value in plan.asdict().items() if key != "fits_minimum"},
        particle_storage_policy=policy,
        target_particle_batch=target,
        live_cache_bytes=live_cache_bytes,
        workflow_remaining_bytes=remaining,
        current_available_bytes=current,
        spatial_cache_bytes=spatial_cache_bytes,
        fixed_components=fixed,
        item_components=item,
    )
