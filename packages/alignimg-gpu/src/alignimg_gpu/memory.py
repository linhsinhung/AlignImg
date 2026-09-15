"""Deterministic VRAM budgeting shared by the CUDA and CuPy engines."""

from __future__ import annotations

from dataclasses import asdict, dataclass


_MINIMUM_RESERVE_BYTES = 256 * 1024**2
_AUTOMATIC_BATCH_SOFT_CAP = 256


@dataclass(frozen=True)
class MemoryPlan:
    free_bytes: int
    total_bytes: int
    budget_bytes: int
    reserve_bytes: int
    fixed_bytes: int
    bytes_per_item: int
    requested_batch_size: int | None
    automatic_batch_soft_cap: int
    batch_size: int

    def asdict(self) -> dict[str, int | None]:
        return asdict(self)


def plan_batch_size(
    *,
    free_bytes: int,
    total_bytes: int,
    memory_fraction: float,
    fixed_bytes: int,
    bytes_per_item: int,
    requested_batch_size: int | None,
) -> MemoryPlan:
    """Choose a batch cap without assuming that all currently free VRAM is ours."""
    if free_bytes < 0 or total_bytes <= 0 or free_bytes > total_bytes:
        raise ValueError("invalid CUDA memory values")
    if not 0 < memory_fraction <= 1:
        raise ValueError("memory_fraction must be in (0, 1]")
    if fixed_bytes < 0 or bytes_per_item <= 0:
        raise ValueError("memory estimates must be non-negative with a positive item size")
    if requested_batch_size is not None and requested_batch_size < 1:
        raise ValueError("requested_batch_size must be positive")

    reserve_bytes = min(
        free_bytes,
        max(_MINIMUM_RESERVE_BYTES, int(total_bytes * (1.0 - memory_fraction))),
    )
    budget_bytes = max(0, free_bytes - reserve_bytes)
    available_for_items = max(0, budget_bytes - fixed_bytes)
    automatic = max(1, available_for_items // bytes_per_item)
    batch_size = min(int(automatic), _AUTOMATIC_BATCH_SOFT_CAP)
    if requested_batch_size is not None:
        batch_size = min(int(automatic), int(requested_batch_size))
    return MemoryPlan(
        free_bytes=int(free_bytes),
        total_bytes=int(total_bytes),
        budget_bytes=int(budget_bytes),
        reserve_bytes=int(reserve_bytes),
        fixed_bytes=int(fixed_bytes),
        bytes_per_item=int(bytes_per_item),
        requested_batch_size=requested_batch_size,
        automatic_batch_soft_cap=_AUTOMATIC_BATCH_SOFT_CAP,
        batch_size=max(1, batch_size),
    )
