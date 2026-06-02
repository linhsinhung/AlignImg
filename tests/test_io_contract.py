#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public API input/output contract checks with synthetic data only."""

from __future__ import annotations

import numpy as np

import alignimg.api as api
from alignimg import _utils as au


def make_synthetic_stack(n=8, size=64):
    geo = au.get_geometry_context((size, size))
    y, x = np.indices((size, size), dtype=np.float32)

    base = np.exp(-(((y - size * 0.40) ** 2) + ((x - size * 0.56) ** 2)) / (2.0 * 5.0 ** 2))
    base += 0.65 * np.exp(
        -(((y - size * 0.67) ** 2) + ((x - size * 0.35) ** 2)) / (2.0 * 3.0 ** 2)
    )
    base += 0.30 * np.exp(
        -(((y - size * 0.50) ** 2) + ((x - size * 0.43) ** 2)) / (2.0 * 2.0 ** 2)
    )
    base = base.astype(np.float32)

    transforms = [
        (-5.0, 1.0, -1.0),
        (4.0, -1.0, 2.0),
        (7.0, 2.0, 1.0),
        (-3.0, 0.0, -2.0),
    ]
    X = np.empty((n, size, size), dtype=np.float32)
    for i in range(n):
        angle, dy, dx = transforms[i % len(transforms)]
        X[i] = au.transform_final_image(base, geo, angle=angle, dy=dy, dx=dx)

    return X, base


def make_fast_config():
    return api.make_mapem_config(
        global_step=60.0,
        mid_range=4.0,
        mid_step=4.0,
        fine_range=1.0,
        fine_step=1.0,
        topk=1,
        mask_soft_edge=3,
    )


def assert_contract(final_ref, history, params, meta, X, num_iterations, search_mode, has_initial_params):
    n, h, w = X.shape

    assert final_ref.shape == (h, w)
    assert len(history) == num_iterations + 1
    assert params.shape == (n, 4)

    assert meta["last_weights"].shape == (n,)
    assert meta["last_image_scores"].shape == (n,)
    assert meta["last_posterior_scores"].shape == (n,)
    assert meta["search_mode"] == search_mode
    assert meta["has_initial_params"] is has_initial_params


def main():
    n = 8
    size = 64
    num_iterations = 1
    X, initial_ref = make_synthetic_stack(n=n, size=size)
    cfg = make_fast_config()

    final_ref, history, params, meta = api.run_alignment(
        X,
        initial_ref,
        num_iterations=num_iterations,
        mask_diameter=56,
        backend="single",
        config=cfg,
        verbose=False,
        search_mode="global",
    )
    assert_contract(
        final_ref,
        history,
        params,
        meta,
        X,
        num_iterations=num_iterations,
        search_mode="global",
        has_initial_params=False,
    )

    corrected = api.run_transform(X, params, backend="single")
    assert corrected.shape == X.shape

    warm_ref, warm_history, warm_params, warm_meta = api.run_alignment(
        X,
        final_ref,
        num_iterations=num_iterations,
        mask_diameter=56,
        backend="single",
        config=cfg,
        verbose=False,
        initial_params=params,
        search_mode="refine",
    )
    assert_contract(
        warm_ref,
        warm_history,
        warm_params,
        warm_meta,
        X,
        num_iterations=num_iterations,
        search_mode="refine",
        has_initial_params=True,
    )

    warm_corrected = api.run_transform(X, warm_params, backend="single")
    assert warm_corrected.shape == X.shape

    try:
        api.run_alignment(
            X,
            final_ref,
            num_iterations=num_iterations,
            mask_diameter=56,
            backend="single",
            config=cfg,
            verbose=False,
            previous_params=params,
            search_mode="refine",
        )
    except TypeError:
        pass
    else:
        raise AssertionError("previous_params should not be accepted")

    print("test_io_contract.py: ok")


def test_io_contract():
    main()


if __name__ == "__main__":
    main()
