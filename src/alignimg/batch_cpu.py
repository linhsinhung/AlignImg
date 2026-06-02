#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experimental batched CPU angle-scan helpers for AlignImg.

This module is intentionally optional. The production MAP-EM path keeps using
the scalar scan unless MAPEMConfig.use_batched_scan is explicitly enabled.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import fft2, ifft2

from . import utils as au


def _unique_normalized_angles(angles) -> list[float]:
    uniq = []
    seen = set()
    for a in np.asarray(angles, dtype=np.float32).ravel():
        an = au.normalize_angle(float(a))
        key = round(an, 6)
        if key not in seen:
            seen.add(key)
            uniq.append(an)
    return uniq


def scan_joint_angles_batched_cpu(img, ref, geo, angles, mask=None):
    """Evaluate one image against many angle candidates using batched FFT.

    The returned candidate dictionaries match alignimg.utils.scan_joint_angles().
    Rotation still uses OpenCV one angle at a time; the experimental speedup is
    limited to batched translation FFTs and shared candidate bookkeeping.
    """
    uniq = _unique_normalized_angles(angles)
    if not uniq:
        return []

    img = np.asarray(img, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)

    rotated_stack = np.empty((len(uniq), geo.H, geo.W), dtype=np.float32)
    for k, angle in enumerate(uniq):
        rotated_stack[k] = au.rotate_image(img, geo, angle)

    f_rot = fft2(rotated_stack, axes=(-2, -1))
    f_ref = fft2(ref)
    cc_stack = np.real(ifft2(f_rot * np.conj(f_ref)[None, :, :], axes=(-2, -1)))

    flat = cc_stack.reshape((len(uniq), -1))
    best_flat = np.argmax(flat, axis=1)
    iy, ix = np.unravel_index(best_flat, (geo.H, geo.W))

    results = []
    for k, angle in enumerate(uniq):
        obs_dy = int(iy[k]) if int(iy[k]) <= geo.H // 2 else int(iy[k] - geo.H)
        obs_dx = int(ix[k]) if int(ix[k]) <= geo.W // 2 else int(ix[k] - geo.W)
        dy = -float(obs_dy)
        dx = -float(obs_dx)
        corrected = au.shift_image(rotated_stack[k], geo, dy=dy, dx=dx)
        score = au.normalized_cross_correlation(corrected, ref, mask=mask)
        results.append({
            "angle": float(angle),
            "dy": dy,
            "dx": dx,
            "score": float(score),
            "fft_score": float(cc_stack[k, int(iy[k]), int(ix[k])]),
        })

    results.sort(key=lambda d: d["score"], reverse=True)
    return results
