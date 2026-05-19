#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean next-generation CPU alignment utilities.

Convention sanity:
- params = [angle, dy, dx, score]
- aligned = transform_final_image(img, geo, angle, dy, dx)

Notes:
- get_best_translation_fft returns *observed* displacement.
- Joint search negates observed displacement to obtain correction shift.
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy.fft import fft2, ifft2


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-180, 180)."""
    return (float(angle) + 180.0) % 360.0 - 180.0


class GeometryContext:
    """Geometry/cache helper for 2D image alignment."""

    def __init__(self, shape):
        if len(shape) == 3:
            self.H, self.W = int(shape[1]), int(shape[2])
        elif len(shape) == 2:
            self.H, self.W = int(shape[0]), int(shape[1])
        else:
            raise ValueError(f"Unsupported shape: {shape}")

        self.cy = self.H / 2.0
        self.cx = self.W / 2.0
        self.cy_int = self.H // 2
        self.cx_int = self.W // 2
        self.max_radius = min(self.H, self.W) / 2.0

        y, x = np.indices((self.H, self.W), dtype=np.float32)
        self.grid_y = y
        self.grid_x = x
        self.dist_sq = (y - self.cy) ** 2 + (x - self.cx) ** 2

        self._mask_cache: dict[tuple[float | None, int], np.ndarray] = {}

    def get_circular_mask(self, diameter=None, soft_edge=5) -> np.ndarray:
        key = (None if diameter is None else float(diameter), int(soft_edge))
        if key in self._mask_cache:
            return self._mask_cache[key]

        radius = (self.max_radius - 1.0) if diameter is None else (float(diameter) / 2.0)
        soft_edge = int(soft_edge)

        if soft_edge <= 0:
            mask = (self.dist_sq <= radius * radius).astype(np.float32)
            self._mask_cache[key] = mask
            return mask

        inner = max(0.0, radius - float(soft_edge))
        mask = np.zeros((self.H, self.W), dtype=np.float32)
        inside = self.dist_sq <= inner * inner
        edge = (self.dist_sq > inner * inner) & (self.dist_sq <= radius * radius)
        mask[inside] = 1.0
        if np.any(edge):
            d = np.sqrt(self.dist_sq[edge])
            t = (d - inner) / float(soft_edge)
            mask[edge] = 0.5 * (1.0 + np.cos(np.pi * t))

        self._mask_cache[key] = mask
        return mask


def get_geometry_context(shape) -> GeometryContext:
    return GeometryContext(shape)


def apply_circular_mask(img: np.ndarray, geo: GeometryContext, diameter=None, soft_edge=5) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) * geo.get_circular_mask(diameter=diameter, soft_edge=soft_edge)


def apply_lowpass_filter(img: np.ndarray, sigma=0.0) -> np.ndarray:
    if float(sigma) <= 0.0:
        return img
    return cv2.GaussianBlur(np.asarray(img, dtype=np.float32), (0, 0), float(sigma))


def shift_image(img: np.ndarray, geo: GeometryContext, dy: float, dx: float) -> np.ndarray:
    m = np.float32([[1, 0, float(dx)], [0, 1, float(dy)]])
    return cv2.warpAffine(np.asarray(img, dtype=np.float32), m, (geo.W, geo.H), flags=cv2.INTER_LINEAR)


def rotate_image(img: np.ndarray, geo: GeometryContext, angle: float) -> np.ndarray:
    m = cv2.getRotationMatrix2D((geo.cx, geo.cy), float(angle), 1.0)
    return cv2.warpAffine(np.asarray(img, dtype=np.float32), m, (geo.W, geo.H), flags=cv2.INTER_LINEAR)


def transform_final_image(img: np.ndarray, geo: GeometryContext, angle: float, dy: float, dx: float) -> np.ndarray:
    """Canonical transform: rotate first, then shift using params [angle, dy, dx]."""
    rotated = rotate_image(img, geo, normalize_angle(angle))
    return shift_image(rotated, geo, dy=float(dy), dx=float(dx))


def normalized_cross_correlation(a, b, mask=None) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if mask is not None:
        m = np.asarray(mask, dtype=np.float32)
        wsum = float(np.sum(m))
        if wsum <= 1e-8:
            return 0.0
        ma = float(np.sum(a * m) / wsum)
        mb = float(np.sum(b * m) / wsum)
        da = (a - ma) * m
        db = (b - mb) * m
    else:
        ma = float(np.mean(a))
        mb = float(np.mean(b))
        da = a - ma
        db = b - mb

    num = float(np.sum(da * db))
    den = float(np.sqrt(np.sum(da * da) * np.sum(db * db)))
    if den <= 1e-12:
        return 0.0
    return num / den


def get_best_translation_fft(img, ref, geo: GeometryContext):
    """Estimate translation via FFT cross-correlation.

    Returns:
        obs_dy, obs_dx, max_score where dy/dx are *observed displacement*.

    IMPORTANT:
        The returned dy/dx are observed displacement.
        To correct the image, use correction shift = -obs_dy, -obs_dx.
    """
    img = np.asarray(img, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    f_img = fft2(img)
    f_ref = fft2(ref)
    cc = np.real(ifft2(f_img * np.conj(f_ref)))

    iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
    obs_dy = int(iy) if iy <= geo.H // 2 else int(iy - geo.H)
    obs_dx = int(ix) if ix <= geo.W // 2 else int(ix - geo.W)
    return float(obs_dy), float(obs_dx), float(cc[iy, ix])


def joint_angle_translation_score(img, ref, geo, angle, mask=None):
    angle = normalize_angle(float(angle))
    rotated = rotate_image(img, geo, angle)
    obs_dy, obs_dx, fft_score = get_best_translation_fft(rotated, ref, geo)
    dy = -float(obs_dy)
    dx = -float(obs_dx)
    corrected = transform_final_image(img, geo, angle, dy, dx)
    score = normalized_cross_correlation(corrected, ref, mask=mask)
    return {"angle": angle, "dy": dy, "dx": dx, "score": float(score), "fft_score": float(fft_score)}


def scan_joint_angles(img, ref, geo, angles, mask=None):
    uniq = []
    seen = set()
    for a in np.asarray(angles, dtype=np.float32).ravel():
        an = normalize_angle(float(a))
        k = round(an, 6)
        if k not in seen:
            seen.add(k)
            uniq.append(an)

    results = [joint_angle_translation_score(img, ref, geo, a, mask=mask) for a in uniq]
    results.sort(key=lambda d: d["score"], reverse=True)
    return results


def coarse_to_fine_joint_search(img, ref, geo, global_step=10.0, mid_range=12.0, mid_step=2.0, fine_range=2.0, fine_step=0.5, topk=3, mask=None):
    stage1_angles = np.arange(-180.0, 180.0, float(global_step), dtype=np.float32)
    stage1 = scan_joint_angles(img, ref, geo, stage1_angles, mask=mask)

    top = stage1[: max(1, int(topk))]
    stage2_angles = []
    for c in top:
        c0 = c["angle"]
        stage2_angles.extend(np.arange(c0 - mid_range, c0 + mid_range + 0.5 * mid_step, mid_step, dtype=np.float32).tolist())
    stage2 = scan_joint_angles(img, ref, geo, stage2_angles, mask=mask)

    center = stage2[0]["angle"] if stage2 else (stage1[0]["angle"] if stage1 else 0.0)
    stage3_angles = np.arange(center - fine_range, center + fine_range + 0.5 * fine_step, fine_step, dtype=np.float32)
    stage3 = scan_joint_angles(img, ref, geo, stage3_angles, mask=mask)

    all_candidates = stage1 + stage2 + stage3
    best = stage3[0] if stage3 else (stage2[0] if stage2 else (stage1[0] if stage1 else {"angle": 0.0, "dy": 0.0, "dx": 0.0, "score": 0.0, "fft_score": 0.0}))
    out = dict(best)
    out["all_candidates"] = all_candidates
    out["n_evaluations"] = int(len(stage1) + len(stage2) + len(stage3))
    return out


def local_joint_search(img, ref, geo, center_angle, angle_range=5.0, angle_step=1.0, mask=None):
    angles = np.arange(float(center_angle) - float(angle_range), float(center_angle) + float(angle_range) + 0.5 * float(angle_step), float(angle_step), dtype=np.float32)
    results = scan_joint_angles(img, ref, geo, angles, mask=mask)
    if results:
        return results[0]
    return {"angle": 0.0, "dy": 0.0, "dx": 0.0, "score": 0.0, "fft_score": 0.0}



def default_iter_schedule(it: int, num_iterations: int) -> dict:
    if it == 0:
        return {"mode": "global", "lp_sigma": 3.0}
    if it == 1:
        return {"mode": "local", "angle_range": 8.0, "angle_step": 2.0, "lp_sigma": 1.0}
    if it == 2:
        return {"mode": "local", "angle_range": 4.0, "angle_step": 1.0, "lp_sigma": 0.0}
    return {"mode": "local", "angle_range": 2.0, "angle_step": 0.5, "lp_sigma": 0.0}


def align_one_single_cpu(img, ref_match, geo, prev_angle, schedule):
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
    params = np.array([angle, dy, dx, score], dtype=np.float32)
    return params, aligned_img


def run_alignment_single_cpu(X, initial_ref, num_iterations=4, mask_diameter=None, verbose=True):
    X = np.asarray(X, dtype=np.float32)
    initial_ref = np.asarray(initial_ref, dtype=np.float32)
    n = int(X.shape[0])
    geo = get_geometry_context(X.shape)
    params = np.zeros((n, 4), dtype=np.float32)
    current_ref = apply_circular_mask(initial_ref, geo, diameter=mask_diameter)
    history_refs = [current_ref.copy()]

    for it in range(int(num_iterations)):
        schedule = default_iter_schedule(it, int(num_iterations))
        if verbose:
            print(f"[iter {it + 1}/{num_iterations}] mode={schedule['mode']} lp_sigma={schedule['lp_sigma']}")
        ref_match = apply_lowpass_filter(current_ref, schedule["lp_sigma"])
        accum = np.zeros_like(current_ref, dtype=np.float32)
        for i in range(n):
            params_i, aligned_i = align_one_single_cpu(X[i], ref_match, geo, prev_angle=float(params[i, 0]), schedule=schedule)
            params[i] = params_i
            accum += aligned_i
        current_ref = accum / max(n, 1)
        current_ref = apply_circular_mask(current_ref, geo, diameter=mask_diameter)
        history_refs.append(current_ref.copy())

    meta = {
        "backend": "single",
        "engine": "align-single-clean",
        "num_iterations": int(num_iterations),
        "mask_diameter": mask_diameter,
        "num_particles": n,
        "implemented": True,
    }
    return current_ref, history_refs, params, meta


def run_transform_single_cpu(X, params):
    X = np.asarray(X, dtype=np.float32)
    params = np.asarray(params, dtype=np.float32)
    geo = get_geometry_context(X.shape)
    out = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        out[i] = transform_final_image(X[i], geo, params[i, 0], params[i, 1], params[i, 2])
    return out
