#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np

import align_utils as au
from align_single import run_alignment_single, run_transform_single


def angle_diff_deg(a, b):
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def normalized_corr(a, b, mask=None):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if mask is not None:
        keep = mask > 0.05
        a = a[keep]
        b = b[keep]
    a = a - np.mean(a)
    b = b - np.mean(b)
    return float(np.sum(a * b) / (np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-8))


def make_template(size=96, seed=7):
    rng = np.random.default_rng(seed)
    geo = au.get_geometry_context((size, size))
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    xn = (x - geo.cx) / size
    yn = (y - geo.cy) / size

    template = np.zeros((size, size), dtype=np.float32)
    blobs = [
        (-0.18, -0.12, 0.060, 1.25),
        (0.13, -0.08, 0.045, 0.95),
        (0.04, 0.17, 0.075, 0.75),
        (-0.23, 0.16, 0.040, -0.55),
        (0.20, 0.19, 0.030, 0.45),
    ]
    for cx, cy, sigma, amp in blobs:
        template += amp * np.exp(-(((xn - cx) ** 2 + (yn - cy) ** 2) / (2 * sigma**2)))

    texture = rng.normal(0.0, 0.04, size=(size, size)).astype(np.float32)
    template += au.apply_lowpass_filter(texture, sigma=2.0)
    template = au.apply_circular_mask(template, geo, diameter=size * 0.82)
    template = (template - np.mean(template)) / (np.std(template) + 1e-8)
    return template.astype(np.float32)


def generate_stack(n=80, size=96, noise=0.35, max_shift=4.5, max_angle=170.0, seed=7):
    rng = np.random.default_rng(seed)
    geo = au.get_geometry_context((size, size))
    gt = make_template(size, seed + 101)

    applied_angles = rng.uniform(-max_angle, max_angle, size=n).astype(np.float32)
    applied_dys = rng.uniform(-max_shift, max_shift, size=n).astype(np.float32)
    applied_dxs = rng.uniform(-max_shift, max_shift, size=n).astype(np.float32)

    X = np.empty((n, size, size), dtype=np.float32)

    for i, (ang, dy, dx) in enumerate(zip(applied_angles, applied_dys, applied_dxs)):
        img = au.rotate_image(gt, geo, float(ang))
        img = au.shift_image(img, geo, float(dy), float(dx))
        img += rng.normal(0.0, noise, size=(size, size)).astype(np.float32)
        X[i] = img

    ideal_angles = angle_diff_deg(-applied_angles, 0.0).astype(np.float32)

    return gt, X, ideal_angles


def main():
    n = 200
    size = 96
    seed = 21
    mask_diameter = size * 0.82

    gt, X, ideal_angles = generate_stack(n=n, size=size, seed=seed)
    init_ref = np.mean(X, axis=0).astype(np.float32)
    geo = au.get_geometry_context((size, size))
    mask = geo.get_circular_mask(diameter=mask_diameter)

    t0 = time.perf_counter()
    final_ref, history, params, meta = run_alignment_single(
        X,
        init_ref,
        num_iterations=4,
        mask_diameter=mask_diameter,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0

    corrected = run_transform_single(X, params)
    corrected_mean = np.mean(corrected, axis=0)

    raw_mean_corr = normalized_corr(corrected_mean, gt, mask)

    # Estimate global gauge correction from final_ref to GT.
    global_best = au.coarse_to_fine_joint_search(final_ref, gt, geo)
    global_angle = float(global_best["angle"])
    global_dy = float(global_best["dy"])
    global_dx = float(global_best["dx"])

    final_ref_gauge = au.transform_final_image(final_ref, geo, global_angle, global_dy, global_dx)
    corrected_mean_gauge = au.transform_final_image(corrected_mean, geo, global_angle, global_dy, global_dx)

    gauge_ref_corr = normalized_corr(final_ref_gauge, gt, mask)
    gauge_mean_corr = normalized_corr(corrected_mean_gauge, gt, mask)

    combined_angle = angle_diff_deg(params[:, 0] + global_angle, 0.0)
    angle_err_gauge = angle_diff_deg(combined_angle, ideal_angles)

    print("\n=== align_single benchmark ===")
    print(f"elapsed: {elapsed:.3f} s")
    print(f"params shape: {params.shape}")
    print(f"raw corrected mean corr: {raw_mean_corr:.4f}")
    print(f"global gauge angle: {global_angle:.3f} deg")
    print(f"gauge-corrected final_ref corr: {gauge_ref_corr:.4f}")
    print(f"gauge-corrected corrected_mean corr: {gauge_mean_corr:.4f}")
    print(f"gauge-corrected angle MAE: {np.mean(np.abs(angle_err_gauge)):.3f} deg")
    print(f"gauge-corrected angle RMSE: {np.sqrt(np.mean(angle_err_gauge**2)):.3f} deg")


if __name__ == "__main__":
    main()
