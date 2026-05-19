#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean public API dispatcher for image alignment backends.

Current implementation status:
- single: implemented via align_single.py
- multicore: reserved for future work
- gpu: reserved for future work
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import align_single

# Temporary compatibility flag for existing tests/callers.
HAS_GPU = False

_BACKEND_INFO = {
    "single": {
        "implemented": True,
        "description": "Clean single-process CPU reference backend.",
    },
    "multicore": {
        "implemented": False,
        "description": "Future multi-core CPU backend following the clean single-process convention.",
    },
    "gpu": {
        "implemented": False,
        "description": "Future GPU backend following the clean single-process convention.",
    },
}


def normalize_backend_name(name: Optional[str]) -> str:
    """Normalize backend aliases to canonical backend names."""
    if name is None:
        return "single"

    key = str(name).strip().lower()
    alias_map = {
        "single": "single",
        "cpu-single": "single",
        "align-single-clean": "single",
        "multicore": "multicore",
        "multi-core": "multicore",
        "cpu-multicore": "multicore",
        "cpu-parallel": "multicore",
        "gpu": "gpu",
        "cuda": "gpu",
    }
    if key in alias_map:
        return alias_map[key]

    valid = ", ".join(sorted(_BACKEND_INFO.keys()))
    raise ValueError(f"Unknown backend '{name}'. Valid backends: {valid}")


def available_backends() -> Dict[str, Dict[str, Any]]:
    """Return implemented status and description for all known backends."""
    return {k: dict(v) for k, v in _BACKEND_INFO.items()}


def run_alignment(
    X,
    initial_ref,
    num_iterations: int = 4,
    mask_diameter=None,
    backend: str = "single",
    verbose: bool = True,
    **kwargs,
):
    """Run alignment using the selected backend dispatcher."""
    del kwargs  # reserved for future backend-specific parameters

    backend_name = normalize_backend_name(backend)

    if backend_name == "single":
        final_ref, history_refs, params = align_single.run_alignment_single(
            X,
            initial_ref,
            num_iterations=num_iterations,
            mask_diameter=mask_diameter,
            verbose=verbose,
        )
        meta = {
            "backend": "single",
            "engine": "align-single-clean",
            "num_iterations": num_iterations,
            "mask_diameter": mask_diameter,
            "implemented": True,
        }
        return final_ref, history_refs, params, meta

    if backend_name == "multicore":
        raise NotImplementedError(
            "The multicore backend is reserved for future implementation. "
            "Use backend='single' for the current clean reference implementation."
        )

    if backend_name == "gpu":
        raise NotImplementedError(
            "The GPU backend is reserved for future implementation. "
            "Use backend='single' for the current clean reference implementation."
        )

    # Defensive fallback: normalize_backend_name already guards this.
    raise ValueError(f"Unsupported backend: {backend_name}")


def run_transform(
    X,
    params,
    backend: str = "single",
    engine: Optional[str] = None,
    **kwargs,
):
    """Apply final transforms using the selected backend."""
    del kwargs  # reserved for future backend-specific parameters

    backend_name = backend
    if engine is not None and backend == "single":
        # Compatibility path: old callers may pass engine instead of backend.
        if str(engine).strip().lower() in {"align-single-clean", "single"}:
            backend_name = "single"

    backend_name = normalize_backend_name(backend_name)

    if backend_name == "single":
        return align_single.run_transform_single(X, params)

    if backend_name == "multicore":
        raise NotImplementedError(
            "The multicore backend is reserved for future implementation. "
            "Use backend='single' for the current clean reference implementation."
        )

    if backend_name == "gpu":
        raise NotImplementedError(
            "The GPU backend is reserved for future implementation. "
            "Use backend='single' for the current clean reference implementation."
        )

    raise ValueError(f"Unsupported backend: {backend_name}")


if __name__ == "__main__":
    print("[alignimg_api] available backends:")
    print(available_backends())

    if hasattr(align_single, "_demo_synthetic_stack"):
        X_demo, ref_demo = align_single._demo_synthetic_stack(n=8, size=64, seed=0)
    else:
        raise RuntimeError("align_single._demo_synthetic_stack is required for smoke test.")

    final_ref, history_refs, params, meta = run_alignment(
        X_demo,
        ref_demo,
        num_iterations=2,
        backend="single",
        verbose=False,
    )
    X_corrected = run_transform(X_demo, params, backend="single")

    print("[alignimg_api] final_ref shape:", final_ref.shape)
    print("[alignimg_api] params shape:", params.shape)
    print("[alignimg_api] corrected stack shape:", X_corrected.shape)
    print("[alignimg_api] meta:", meta)
