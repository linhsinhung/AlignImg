#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight public API smoke tests with synthetic data only."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import alignimg.api as api
import alignimg.utils as au


def make_synthetic_stack(n=8, size=64):
    """Create a deterministic asymmetric stack for API smoke tests."""
    geo = au.get_geometry_context((size, size))
    y, x = np.indices((size, size), dtype=np.float32)

    base = np.exp(-(((y - size * 0.38) ** 2) + ((x - size * 0.57) ** 2)) / (2.0 * 5.0 ** 2))
    base += 0.70 * np.exp(
        -(((y - size * 0.66) ** 2) + ((x - size * 0.34) ** 2)) / (2.0 * 3.5 ** 2)
    )
    base += 0.35 * np.exp(
        -(((y - size * 0.48) ** 2) + ((x - size * 0.42) ** 2)) / (2.0 * 2.0 ** 2)
    )
    base = base.astype(np.float32)

    transforms = [
        (-6.0, 1.0, -1.0),
        (3.0, -1.0, 2.0),
        (8.0, 2.0, 1.0),
        (-2.0, 0.0, -2.0),
    ]
    X = np.empty((n, size, size), dtype=np.float32)
    for i in range(n):
        angle, dy, dx = transforms[i % len(transforms)]
        X[i] = au.transform_final_image(base, geo, angle=angle, dy=dy, dx=dx)

    return X, base


def make_fast_config(use_batched_scan=False):
    return api.make_mapem_config(
        global_step=60.0,
        mid_range=4.0,
        mid_step=4.0,
        fine_range=1.0,
        fine_step=1.0,
        topk=1,
        mask_soft_edge=3,
        use_batched_scan=use_batched_scan,
    )


def assert_alignment_output(final_ref, history, params, meta, n, size):
    assert final_ref.shape == (size, size)
    assert len(history) >= 2
    assert params.shape == (n, 4)
    assert meta["last_weights"].shape == (n,)
    assert meta["last_image_scores"].shape == (n,)
    assert meta["last_posterior_scores"].shape == (n,)


def main():
    n = 8
    size = 64
    X, initial_ref = make_synthetic_stack(n=n, size=size)

    backends = api.available_backends()
    assert backends["single"]["implemented"] is True
    assert backends["multicore"]["implemented"] is True

    cfg = make_fast_config()

    final_ref_single, history_single, params_single, meta_single = api.run_alignment(
        X,
        initial_ref,
        num_iterations=1,
        mask_diameter=56,
        backend="single",
        algorithm="mapem",
        config=cfg,
        verbose=False,
    )
    assert_alignment_output(final_ref_single, history_single, params_single, meta_single, n, size)

    final_ref_multi, history_multi, params_multi, meta_multi = api.run_alignment(
        X,
        initial_ref,
        num_iterations=1,
        mask_diameter=56,
        backend="multicore",
        algorithm="mapem",
        config=cfg,
        verbose=False,
        n_jobs=2,
        chunksize=1,
    )
    assert_alignment_output(final_ref_multi, history_multi, params_multi, meta_multi, n, size)
    assert meta_multi["use_shared_memory"] is False

    final_ref_shared, history_shared, params_shared, meta_shared = api.run_alignment(
        X,
        initial_ref,
        num_iterations=1,
        mask_diameter=56,
        backend="multicore",
        algorithm="mapem",
        config=cfg,
        verbose=False,
        n_jobs=2,
        chunksize=1,
        use_shared_memory=True,
    )
    assert_alignment_output(final_ref_shared, history_shared, params_shared, meta_shared, n, size)
    assert meta_shared["use_shared_memory"] is True
    assert np.allclose(final_ref_shared, final_ref_multi)
    assert np.allclose(params_shared, params_multi)

    final_ref_warm, history_warm, params_warm, meta_warm = api.run_alignment(
        X,
        final_ref_multi,
        num_iterations=1,
        mask_diameter=56,
        backend="multicore",
        algorithm="mapem",
        config=cfg,
        verbose=False,
        n_jobs=2,
        chunksize=1,
        previous_params=params_multi,
        search_mode="refine",
    )
    assert_alignment_output(final_ref_warm, history_warm, params_warm, meta_warm, n, size)
    assert meta_warm["search_mode"] == "refine"

    corrected = api.run_transform(X, params_warm, backend="single")
    assert corrected.shape == X.shape

    batched_cfg = make_fast_config(use_batched_scan=True)
    assert batched_cfg.use_batched_scan is True
    final_ref_batched, history_batched, params_batched, meta_batched = api.run_alignment(
        X,
        final_ref_single,
        num_iterations=1,
        mask_diameter=56,
        backend="single",
        algorithm="mapem",
        config=batched_cfg,
        verbose=False,
        previous_params=params_single[:, :3],
        search_mode="refine",
    )
    assert_alignment_output(final_ref_batched, history_batched, params_batched, meta_batched, n, size)

    print("test_api_smoke.py: ok")


def test_api_smoke():
    main()


if __name__ == "__main__":
    main()
