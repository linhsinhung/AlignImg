#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean single-process reference alignment implementation."""

from __future__ import annotations

import time
import numpy as np

from align_utils import (
    get_geometry_context,
    apply_circular_mask,
    apply_lowpass_filter,
    transform_final_image,
    coarse_to_fine_joint_search,
    local_joint_search,
)


def single_iter_schedule(it: int, num_iterations: int) -> dict:
    """Return per-iteration search schedule."""
    if it == 0:
        return {"mode": "global", "lp_sigma": 3.0}
    if it == 1:
        return {"mode": "local", "angle_range": 8.0, "angle_step": 2.0, "lp_sigma": 1.0}
    if it == 2:
        return {"mode": "local", "angle_range": 4.0, "angle_step": 1.0, "lp_sigma": 0.0}
    return {"mode": "local", "angle_range": 2.0, "angle_step": 0.5, "lp_sigma": 0.0}



def align_one_single(img, ref_match, geo, prev_angle, schedule):
    """Align one particle and return final transform params."""
    if schedule["mode"] == "global":
        best = coarse_to_fine_joint_search(img, ref_match, geo)
    else:
        best = local_joint_search(
            img,
            ref_match,
            geo,
            center_angle=float(prev_angle),
            angle_range=float(schedule["angle_range"]),
            angle_step=float(schedule["angle_step"]),
        )

    angle = float(best["angle"])
    dy = float(best["dy"])
    dx = float(best["dx"])
    score = float(best["score"])
    aligned_img = transform_final_image(img, geo, angle, dy, dx)
    # Convention: params = [angle, dy, dx, score] and are directly reusable in transform_final_image.
    params = np.array([angle, dy, dx, score], dtype=np.float32)
    return params, aligned_img


def run_alignment_single(X, initial_ref, num_iterations=4, mask_diameter=None, verbose=True):
    """Serial CPU reference-alignment loop."""
    X = np.asarray(X, dtype=np.float32)
    initial_ref = np.asarray(initial_ref, dtype=np.float32)
    n = X.shape[0]
    geo = get_geometry_context(X.shape)

    params = np.zeros((n, 4), dtype=np.float32)
    history_refs = []
    current_ref = apply_circular_mask(initial_ref, geo, diameter=mask_diameter)

    for it in range(int(num_iterations)):
        schedule = single_iter_schedule(it, int(num_iterations))
        if verbose:
            print(f"[iter {it + 1}/{num_iterations}] mode={schedule['mode']} lp_sigma={schedule['lp_sigma']}")

        ref_match = apply_lowpass_filter(current_ref, sigma=float(schedule["lp_sigma"]))

        accum = np.zeros_like(current_ref, dtype=np.float32)
        for i in range(n):
            p, aligned = align_one_single(X[i], ref_match, geo, prev_angle=float(params[i, 0]), schedule=schedule)
            params[i] = p
            accum += aligned

        new_ref = accum / max(n, 1)
        new_ref = apply_circular_mask(new_ref, geo, diameter=mask_diameter)
        current_ref = new_ref.astype(np.float32, copy=False)
        history_refs.append(current_ref.copy())

    meta = {"num_iterations": int(num_iterations), "num_particles": int(n)}
    return current_ref, history_refs, params, meta


def run_transform_single(X, params):
    """Apply final parameters directly to all particles."""
    X = np.asarray(X, dtype=np.float32)
    params = np.asarray(params, dtype=np.float32)
    geo = get_geometry_context(X.shape)
    out = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        angle, dy, dx = float(params[i, 0]), float(params[i, 1]), float(params[i, 2])
        out[i] = transform_final_image(X[i], geo, angle, dy, dx)
    return out


def _demo_synthetic_stack(n=16, size=96, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.indices((size, size), dtype=np.float32)
    cy = (size - 1) / 2.0
    cx = (size - 1) / 2.0

    base = np.zeros((size, size), dtype=np.float32)
    for _ in range(8):
        ay = rng.uniform(-18, 18)
        ax = rng.uniform(-18, 18)
        s = rng.uniform(2.5, 6.0)
        amp = rng.uniform(0.6, 1.2)
        base += amp * np.exp(-(((y - (cy + ay)) ** 2 + (x - (cx + ax)) ** 2) / (2 * s * s)))

    geo = get_geometry_context((size, size))
    X = np.zeros((n, size, size), dtype=np.float32)
    for i in range(n):
        angle = rng.uniform(-25.0, 25.0)
        dy = rng.uniform(-6.0, 6.0)
        dx = rng.uniform(-6.0, 6.0)
        noisy = base + rng.normal(0.0, 0.08, size=(size, size)).astype(np.float32)
        X[i] = transform_final_image(noisy.astype(np.float32), geo, angle, dy, dx)

    initial_ref = np.mean(X, axis=0).astype(np.float32)
    return X, initial_ref


if __name__ == "__main__":
    X, initial_ref = _demo_synthetic_stack()
    t0 = time.perf_counter()
    final_ref, history_refs, params, meta = run_alignment_single(X, initial_ref, num_iterations=4, verbose=True)
    elapsed = time.perf_counter() - t0

    print(f"final reference mean/std: {float(np.mean(final_ref)):.6f} / {float(np.std(final_ref)):.6f}")
    print(f"params shape: {params.shape}")
    print(f"elapsed time: {elapsed:.3f}s")
