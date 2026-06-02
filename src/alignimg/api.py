"""Public MAP-EM API for AlignImg."""

from __future__ import annotations

from typing import Any

from . import _utils as au

MAPEMConfig = au.MAPEMConfig

_BACKEND_INFO = {
    "single": {
        "implemented": True,
        "description": "Single-process CPU MAP-EM backend.",
    },
    "multicore": {
        "implemented": True,
        "description": "Process-parallel CPU MAP-EM backend.",
    },
}


def available_backends() -> dict[str, dict[str, Any]]:
    """Return implemented backend metadata."""
    return {name: dict(info) for name, info in _BACKEND_INFO.items()}


def _normalize_backend(name: str | None) -> str:
    if name is None:
        return "single"
    key = str(name).strip().lower()
    if key in _BACKEND_INFO:
        return key
    raise ValueError("backend must be 'single' or 'multicore'.")


def make_mapem_config(
    phase: int = 3,
    weight_mode: str = "sigmoid",
    keep_fraction: float = 0.75,
    weight_temperature: float = 0.08,
    score_threshold=None,
    min_weight: float = 0.0,
    lambda_shift: float = 0.01,
    sigma_shift_y: float = 8.0,
    sigma_shift_x: float = 8.0,
    lambda_angle: float = 0.0,
    sigma_angle: float = 8.0,
    global_step: float = 10.0,
    mid_range: float = 12.0,
    mid_step: float = 2.0,
    fine_range: float = 2.0,
    fine_step: float = 0.5,
    topk: int = 3,
    use_batched_scan: bool = False,
    normalize_reference: bool = False,
    mask_soft_edge: int = 5,
    diagnostics_n: int = 0,
) -> MAPEMConfig:
    """Create a MAP-EM configuration with package defaults."""
    return MAPEMConfig(
        phase=phase,
        weight_mode=weight_mode,
        keep_fraction=keep_fraction,
        score_threshold=score_threshold,
        weight_temperature=weight_temperature,
        min_weight=min_weight,
        lambda_shift=lambda_shift,
        sigma_shift_y=sigma_shift_y,
        sigma_shift_x=sigma_shift_x,
        lambda_angle=lambda_angle,
        sigma_angle=sigma_angle,
        global_step=global_step,
        mid_range=mid_range,
        mid_step=mid_step,
        fine_range=fine_range,
        fine_step=fine_step,
        topk=topk,
        use_batched_scan=use_batched_scan,
        normalize_reference=normalize_reference,
        mask_soft_edge=mask_soft_edge,
        diagnostics_n=diagnostics_n,
    )


def run_alignment(
    X,
    initial_ref,
    *,
    num_iterations: int = 4,
    mask_diameter=None,
    backend: str = "single",
    config: MAPEMConfig | None = None,
    verbose: bool = True,
    n_jobs=None,
    chunksize=1,
    initial_params=None,
    search_mode: str | None = None,
    use_shared_memory: bool = False,
):
    """Run MAP-EM alignment through the selected CPU backend.

    Parameters use the canonical pose convention:
        params = [angle, dy, dx, score]

    `initial_params` is the only warm-start input. Passing it with
    `search_mode="refine"` skips the cold-start global angle sweep.
    """
    backend = _normalize_backend(backend)
    cfg = config if config is not None else make_mapem_config()

    if backend == "multicore":
        from . import _multicore as amc

        return amc.run_alignment_mapem_multicore(
            X,
            initial_ref,
            num_iterations=num_iterations,
            mask_diameter=mask_diameter,
            config=cfg,
            verbose=verbose,
            n_jobs=n_jobs,
            chunksize=chunksize,
            initial_params=initial_params,
            search_mode=search_mode,
            use_shared_memory=use_shared_memory,
        )

    return au.run_alignment_mapem_cpu(
        X,
        initial_ref,
        num_iterations=num_iterations,
        mask_diameter=mask_diameter,
        config=cfg,
        verbose=verbose,
        initial_params=initial_params,
        search_mode=search_mode,
    )


def run_transform(
    X,
    params,
    *,
    backend: str = "single",
    n_jobs=None,
):
    """Apply MAP-EM transform parameters to an image stack."""
    backend = _normalize_backend(backend)
    if backend == "multicore":
        from . import _multicore as amc

        return amc.run_transform_multicore(X, params, n_jobs=n_jobs)
    return au.run_transform_single_cpu(X, params)
