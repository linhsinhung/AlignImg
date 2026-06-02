#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public backend dispatcher for the clean alignment package.

Formal architecture:
    alignimg.utils     : CPU primitives + CPU single-process backends
    alignimg.multicore : multi-core CPU MAP-EM backend
    alignimg.api       : public interface / backend dispatcher
    align_utils_gpu.py : future GPU backend

Default algorithm:
    backend="single", algorithm="mapem", phase=3

Recommended Phase-3 baseline:
    robust MAP-EM with translation prior only:
        lambda_shift=0.01
        sigma_shift_y=8.0
        sigma_shift_x=8.0
        lambda_angle=0.0

For repeated CPU calls, the recommended production path is:
    backend="multicore", algorithm="mapem", phase=3,
    search_mode="refine" with initial_params from the previous call.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from . import utils as au

HAS_GPU = False

_BACKEND_INFO = {
    "single": {
        "implemented": True,
        "description": "Single-process CPU backend using clean alignimg.utils primitives.",
    },
    "multicore": {
        "implemented": True,
        "description": "Multi-core CPU backend using particle-level parallel MAP-EM E-step.",
    },
    "gpu": {
        "implemented": False,
        "description": "Future GPU backend using align_utils_gpu primitives.",
    },
}


def available_backends() -> Dict[str, Dict[str, Any]]:
    """Return available backend information."""
    return {k: dict(v) for k, v in _BACKEND_INFO.items()}


def normalize_backend_name(name: Optional[str]) -> str:
    """Normalize backend aliases."""
    if name is None:
        return "single"

    key = str(name).strip().lower()
    alias_map = {
        "single": "single",
        "serial": "single",
        "cpu-single": "single",
        "cpu-serial": "single",
        "align-single-clean": "single",
        "align-single-mapem": "single",
        "mapem": "single",
        "multicore": "multicore",
        "multi-core": "multicore",
        "cpu-multicore": "multicore",
        "cpu-parallel": "multicore",
        "parallel": "multicore",
        "gpu": "gpu",
        "cuda": "gpu",
    }
    if key in alias_map:
        return alias_map[key]
    raise ValueError(f"Unknown backend '{name}'. Available: {list(_BACKEND_INFO.keys())}")


def normalize_algorithm_name(name: Optional[str]) -> str:
    """Normalize algorithm aliases."""
    if name is None:
        return "mapem"

    key = str(name).strip().lower()
    alias_map = {
        "mapem": "mapem",
        "map-em": "mapem",
        "robust-mapem": "mapem",
        "phase3": "mapem",
        "classic": "classic",
        "hard": "classic",
        "hard-map": "classic",
        "align-single-clean": "classic",
    }
    if key in alias_map:
        return alias_map[key]
    raise ValueError("Unknown algorithm '{}'. Use 'mapem' or 'classic'.".format(name))


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
) -> au.MAPEMConfig:
    """Create a MAPEMConfig with the current recommended defaults."""
    return au.MAPEMConfig(
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


def _resolve_initial_params(initial_params=None, previous_params=None, warm_start_params=None):
    provided = [
        name
        for name, value in (
            ("initial_params", initial_params),
            ("previous_params", previous_params),
            ("warm_start_params", warm_start_params),
        )
        if value is not None
    ]
    if len(provided) > 1:
        raise ValueError(
            "Use only one pose warm-start argument: "
            "initial_params, previous_params, or warm_start_params."
        )
    if previous_params is not None:
        return previous_params
    if warm_start_params is not None:
        return warm_start_params
    return initial_params


def run_alignment(
    X,
    initial_ref,
    num_iterations: int = 4,
    mask_diameter=None,
    backend: str = "single",
    algorithm: str = "mapem",
    verbose: bool = True,
    config: Optional[au.MAPEMConfig] = None,
    n_jobs=None,
    chunksize=1,
    initial_params=None,
    previous_params=None,
    warm_start_params=None,
    search_mode: Optional[str] = None,
    use_shared_memory: bool = False,
    **kwargs,
):
    """Run alignment through the public backend dispatcher.

    Parameters
    ----------
    backend:
        Currently implemented: "single" and "multicore" for MAP-EM.
        Reserved stub: "gpu".

    algorithm:
        "mapem"  -> robust MAP-EM backend; default recommended algorithm.
        "classic" -> old clean hard-MAP/unweighted backend.

    config:
        Optional au.MAPEMConfig. If provided, it overrides MAP-EM kwargs.

    Common MAP-EM kwargs include:
        phase, weight_mode, keep_fraction, lambda_shift,
        sigma_shift_y, sigma_shift_x, lambda_angle, sigma_angle.

    Pose warm starts:
        Pass initial_params, previous_params, or warm_start_params with shape
        (N, 3) or (N, 4). The canonical convention is preserved:
        params = [angle, dy, dx, score]. Warm-started MAP-EM skips the initial
        global angle sweep and runs local refinement from these angles.

    search_mode:
        "auto" preserves existing behavior, "refine" forces local warm-start
        refinement from initial_params, and "global" uses the cold-start
        schedule.

    use_shared_memory:
        Optional multicore MAP-EM mode that shares input arrays between worker
        processes to reduce pickle/copy overhead. It is ignored by the single
        backend and defaults to False.
    """
    backend = normalize_backend_name(backend)
    algorithm = normalize_algorithm_name(algorithm)
    pose_seed = _resolve_initial_params(
        initial_params=initial_params,
        previous_params=previous_params,
        warm_start_params=warm_start_params,
    )

    if backend == "multicore":
        if algorithm == "classic":
            raise NotImplementedError(
                "multicore classic backend is not implemented; "
                "use algorithm='mapem' or backend='single'."
            )
        from . import multicore as amc

        cfg = config if config is not None else make_mapem_config(**kwargs)
        return amc.run_alignment_mapem_multicore(
            X,
            initial_ref,
            num_iterations=num_iterations,
            mask_diameter=mask_diameter,
            config=cfg,
            verbose=verbose,
            n_jobs=n_jobs,
            chunksize=chunksize,
            initial_params=pose_seed,
            search_mode=search_mode,
            use_shared_memory=use_shared_memory,
        )

    if backend == "gpu":
        raise NotImplementedError(
            "The GPU backend is reserved for future implementation. "
            "Use backend='single' for the current implementation."
        )

    if backend != "single":
        raise ValueError(f"Unsupported backend: {backend}")

    if algorithm == "classic":
        if pose_seed is not None:
            raise NotImplementedError("Pose warm starts are currently implemented for algorithm='mapem' only.")
        return au.run_alignment_single_cpu(
            X,
            initial_ref,
            num_iterations=num_iterations,
            mask_diameter=mask_diameter,
            verbose=verbose,
        )

    if algorithm == "mapem":
        cfg = config if config is not None else make_mapem_config(**kwargs)
        return au.run_alignment_mapem_cpu(
            X,
            initial_ref,
            num_iterations=num_iterations,
            mask_diameter=mask_diameter,
            config=cfg,
            verbose=verbose,
            initial_params=pose_seed,
            search_mode=search_mode,
        )

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def run_transform(
    X,
    params,
    backend: str = "single",
    engine: Optional[str] = None,
    algorithm: Optional[str] = None,
    n_jobs=None,
    **kwargs,
):
    """Apply final transform parameters.

    The transform convention is identical for classic and MAP-EM:
        params = [angle, dy, dx, score]
    """
    del kwargs

    if engine is not None and algorithm is None:
        e = str(engine).strip().lower()
        if e in {"align-single-mapem", "align-single-v2-mapem", "single-v2-mapem"}:
            algorithm = "mapem"
        elif e in {"align-single-clean", "single"}:
            algorithm = "classic"

    backend = normalize_backend_name(backend)
    if backend == "multicore":
        from . import multicore as amc

        return amc.run_transform_multicore(X, params, n_jobs=n_jobs)
    if backend == "gpu":
        raise NotImplementedError(f"{backend} backend is not implemented yet")
    if backend != "single":
        raise ValueError(f"Unsupported backend: {backend}")

    # Both algorithms use the same final transform convention.
    return au.run_transform_single_cpu(X, params)


if __name__ == "__main__":
    print("[alignimg.api] available backends:")
    print(available_backends())
    print("\nDefault run_alignment uses algorithm='mapem', phase=3.")
