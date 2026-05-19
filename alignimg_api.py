#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional

import align_utils as au

HAS_GPU = False

_BACKEND_INFO = {
    "single": {"implemented": True, "description": "Clean single-process CPU reference backend."},
    "multicore": {"implemented": False, "description": "Future multi-core CPU backend."},
    "gpu": {"implemented": False, "description": "Future GPU backend."},
}


def available_backends() -> Dict[str, Dict[str, Any]]:
    return {k: dict(v) for k, v in _BACKEND_INFO.items()}


def normalize_backend_name(name: Optional[str]) -> str:
    if name is None:
        return "single"
    key = str(name).strip().lower()
    alias_map = {
        "single": "single",
        "align-single-clean": "single",
        "cpu-single": "single",
        "multicore": "multicore",
        "multi-core": "multicore",
        "gpu": "gpu",
        "cuda": "gpu",
    }
    if key in alias_map:
        return alias_map[key]
    raise ValueError(f"Unknown backend '{name}'")


def run_alignment(X, initial_ref, num_iterations: int = 4, mask_diameter=None, backend: str = "single", verbose: bool = True, **kwargs):
    del kwargs
    backend = normalize_backend_name(backend)
    if backend == "single":
        return au.run_alignment_single_cpu(X, initial_ref, num_iterations=num_iterations, mask_diameter=mask_diameter, verbose=verbose)
    if backend == "multicore":
        raise NotImplementedError("multicore backend is not implemented yet")
    if backend == "gpu":
        raise NotImplementedError("gpu backend is not implemented yet")
    raise ValueError(f"Unsupported backend: {backend}")


def run_transform(X, params, backend: str = "single", engine: Optional[str] = None, **kwargs):
    del kwargs
    if engine is not None:
        e = str(engine).strip().lower()
        if e in {"align-single-clean", "single"}:
            backend = "single"
    backend = normalize_backend_name(backend)
    if backend == "single":
        return au.run_transform_single_cpu(X, params)
    if backend in {"multicore", "gpu"}:
        raise NotImplementedError(f"{backend} backend is not implemented yet")
    raise ValueError(f"Unsupported backend: {backend}")


if __name__ == "__main__":
    print("[alignimg_api] available backends:")
    print(available_backends())
