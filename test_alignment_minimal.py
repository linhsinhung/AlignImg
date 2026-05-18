#!/usr/bin/env python3
"""Minimal staged synthetic alignment benchmark.

This script intentionally avoids the production reference-building pipeline. It
uses a single asymmetric synthetic template and then validates transform and
estimator conventions one stage at a time.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def _install_minimal_scipy_fallback() -> None:
    """Install the tiny scipy surface align_utils needs when scipy is absent."""
    if importlib.util.find_spec("scipy") is not None:
        return

    scipy_mod = types.ModuleType("scipy")
    fft_mod = types.ModuleType("scipy.fft")
    ndimage_mod = types.ModuleType("scipy.ndimage")
    fft_mod.fft2 = np.fft.fft2
    fft_mod.ifft2 = np.fft.ifft2
    fft_mod.fftshift = np.fft.fftshift

    def center_of_mass(arr):
        data = np.asarray(arr, dtype=np.float64)
        total = float(np.sum(data))
        if abs(total) <= 1e-12:
            return tuple((np.array(data.shape) - 1.0) / 2.0)
        coords = np.indices(data.shape, dtype=np.float64)
        return tuple(float(np.sum(coords[i] * data) / total) for i in range(data.ndim))

    ndimage_mod.center_of_mass = center_of_mass
    scipy_mod.fft = fft_mod
    scipy_mod.ndimage = ndimage_mod
    sys.modules.setdefault("scipy", scipy_mod)
    sys.modules.setdefault("scipy.fft", fft_mod)
    sys.modules.setdefault("scipy.ndimage", ndimage_mod)


def _install_minimal_cv2_fallback() -> None:
    """Install a small cv2-compatible shim sufficient for this benchmark."""
    if importlib.util.find_spec("cv2") is not None:
        return

    cv2_mod = types.ModuleType("cv2")
    cv2_mod.INTER_LINEAR = 1
    cv2_mod.LINE_AA = 16
    cv2_mod.WARP_FILL_OUTLIERS = 8

    def _bilinear_sample(img, x, y):
        h, w = img.shape
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = x0 + 1
        y1 = y0 + 1
        valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
        x0c = np.clip(x0, 0, w - 1)
        x1c = np.clip(x1, 0, w - 1)
        y0c = np.clip(y0, 0, h - 1)
        y1c = np.clip(y1, 0, h - 1)
        wx = x - x0
        wy = y - y0
        out = (
            (1 - wx) * (1 - wy) * img[y0c, x0c]
            + wx * (1 - wy) * img[y0c, x1c]
            + (1 - wx) * wy * img[y1c, x0c]
            + wx * wy * img[y1c, x1c]
        )
        return np.where(valid, out, 0).astype(img.dtype, copy=False)

    def getRotationMatrix2D(center, angle, scale):
        cx, cy = center
        alpha = float(scale) * math.cos(math.radians(angle))
        beta = float(scale) * math.sin(math.radians(angle))
        return np.array(
            [[alpha, beta, (1 - alpha) * cx - beta * cy], [-beta, alpha, beta * cx + (1 - alpha) * cy]],
            dtype=np.float32,
        )

    def warpAffine(img, M, dsize, flags=1):
        width, height = dsize
        yy, xx = np.indices((height, width), dtype=np.float32)
        homog = np.vstack([M, [0, 0, 1]]).astype(np.float64)
        inv = np.linalg.inv(homog)[:2]
        src_x = inv[0, 0] * xx + inv[0, 1] * yy + inv[0, 2]
        src_y = inv[1, 0] * xx + inv[1, 1] * yy + inv[1, 2]
        return _bilinear_sample(np.asarray(img), src_x, src_y)

    def GaussianBlur(img, ksize, sigma):
        if sigma <= 0:
            return img
        radius = max(1, int(math.ceil(3.0 * float(sigma))))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-0.5 * (x / float(sigma)) ** 2)
        kernel /= np.sum(kernel)
        tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, np.asarray(img, dtype=np.float32))
        return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, tmp).astype(np.asarray(img).dtype, copy=False)

    def line(img, pt1, pt2, color, thickness=1, lineType=16):
        x1, y1 = pt1
        x2, y2 = pt2
        n = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
        rr = max(0, int(math.ceil(thickness / 2)))
        for t in np.linspace(0.0, 1.0, n):
            x = int(round(x1 + t * (x2 - x1)))
            y = int(round(y1 + t * (y2 - y1)))
            y0, y1c = max(0, y - rr), min(img.shape[0], y + rr + 1)
            x0, x1c = max(0, x - rr), min(img.shape[1], x + rr + 1)
            img[y0:y1c, x0:x1c] = color
        return img

    def linearPolar(src, center, maxRadius, flags=8):
        h, w = src.shape
        cy, cx = float(center[1]), float(center[0])
        out_y, out_x = np.indices((h, w), dtype=np.float32)
        theta = out_y / float(h) * 2.0 * math.pi
        radius = out_x / max(float(w - 1), 1.0) * float(maxRadius)
        src_x = cx + radius * np.cos(theta)
        src_y = cy + radius * np.sin(theta)
        return _bilinear_sample(np.asarray(src), src_x, src_y)

    def imwrite(path, img):
        # Dependency-free binary PGM payload. The extension may be .png, but the
        # overview remains readable by tools that inspect the file signature.
        arr = np.asarray(img, dtype=np.uint8)
        with open(path, "wb") as f:
            f.write(f"P5\n{arr.shape[1]} {arr.shape[0]}\n255\n".encode("ascii"))
            f.write(arr.tobytes())
        return True

    cv2_mod.getRotationMatrix2D = getRotationMatrix2D
    cv2_mod.warpAffine = warpAffine
    cv2_mod.GaussianBlur = GaussianBlur
    cv2_mod.line = line
    cv2_mod.linearPolar = linearPolar
    cv2_mod.imwrite = imwrite
    sys.modules.setdefault("cv2", cv2_mod)


_install_minimal_scipy_fallback()
_install_minimal_cv2_fallback()
cv2 = importlib.import_module("cv2")

from align_utils import (
    apply_circular_mask,
    apply_lowpass_filter,
    coarse_to_fine_joint_search,
    _angle_from_fm_profile_index,
    _valid_fm_lag_indices,
    get_best_translation_fft,
    get_coarse_angle_fourier_mellin_profile,
    get_geometry_context,
    normalize_angle,
    rotate_image,
    shift_image,
    transform_final_image,
)


@dataclass
class ResultRow:
    test_name: str
    applied_angle: float
    applied_dy: float
    applied_dx: float
    ideal_angle: float
    ideal_dy: float
    ideal_dx: float
    estimated_angle: float
    estimated_dy: float
    estimated_dx: float
    angle_error_deg: float
    dy_error_px: float
    dx_error_px: float
    corr: float
    nrmse: float
    passed: bool
    notes: str
    raw_fft_dy: float = float("nan")
    raw_fft_dx: float = float("nan")
    negated_fft_dy: float = float("nan")
    negated_fft_dx: float = float("nan")
    fm_angle: float = float("nan")
    bruteforce_angle: float = float("nan")
    fm_vs_bruteforce_error: float = float("nan")
    fm_vs_ideal_error: float = float("nan")
    fm_peak_index: int = -1
    fm_lag: int = 0
    top5_fm_angles: str = ""
    top5_fm_scores: str = ""
    fm_axis0_best_angle: float = float("nan")
    fm_axis0_top5_angles: str = ""
    fm_axis1_best_angle: float = float("nan")
    fm_axis1_top5_angles: str = ""
    fm_zm_best_angle: float = float("nan")
    fm_zm_top5_angles: str = ""
    fm_highpass_best_angle: float = float("nan")
    fm_highpass_top5_angles: str = ""
    fm_radial_weight_best_angle: float = float("nan")
    fm_radial_weight_top5_angles: str = ""
    fm_polar2d_best_angle: float = float("nan")
    fm_polar2d_top5_angles: str = ""


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64) - float(np.mean(a))
    bb = np.asarray(b, dtype=np.float64) - float(np.mean(b))
    denom = math.sqrt(float(np.sum(aa * aa)) * float(np.sum(bb * bb)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(aa * bb) / denom)


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    rmse = math.sqrt(float(np.mean((aa - bb) ** 2)))
    scale = float(np.max(bb) - np.min(bb))
    if scale <= 1e-12:
        scale = float(np.std(bb)) + 1e-12
    return float(rmse / scale)


def angle_error(est: float, ideal: float) -> float:
    return abs(normalize_angle(float(est) - float(ideal)))


def normalize_template(img: np.ndarray, geo) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    img = apply_circular_mask(img, geo, diameter=geo.max_radius * 1.75, soft_edge=6)
    img -= float(np.mean(img))
    std = float(np.std(img))
    if std > 1e-12:
        img /= std
    return img.astype(np.float32, copy=False)


def add_gaussian(img: np.ndarray, cy: float, cx: float, sy: float, sx: float, amp: float) -> None:
    y, x = np.indices(img.shape, dtype=np.float32)
    img += amp * np.exp(-0.5 * (((y - cy) / sy) ** 2 + ((x - cx) / sx) ** 2)).astype(np.float32)


def make_asymmetric_template(size: int) -> np.ndarray:
    geo = get_geometry_context((size, size))
    img = np.zeros((size, size), dtype=np.float32)

    # Mixed-polarity, intentionally non-symmetric landmarks.
    add_gaussian(img, 0.30 * size, 0.36 * size, 5.0, 7.0, 2.8)  # large positive blob
    add_gaussian(img, 0.72 * size, 0.69 * size, 2.2, 2.8, 1.8)  # small positive blob
    add_gaussian(img, 0.60 * size, 0.26 * size, 4.0, 3.0, -2.2)  # negative blob

    # Off-center elongated diagonal bar.
    bar = np.zeros_like(img)
    p1 = (int(0.54 * size), int(0.18 * size))
    p2 = (int(0.85 * size), int(0.42 * size))
    cv2.line(bar, p1, p2, color=1.6, thickness=max(3, size // 24), lineType=cv2.LINE_AA)
    img += apply_lowpass_filter(bar, sigma=0.8)

    # L-shaped feature with unequal arms.
    lshape = np.zeros_like(img)
    cv2.line(
        lshape,
        (int(0.16 * size), int(0.78 * size)),
        (int(0.42 * size), int(0.78 * size)),
        color=1.2,
        thickness=max(2, size // 32),
        lineType=cv2.LINE_AA,
    )
    cv2.line(
        lshape,
        (int(0.16 * size), int(0.78 * size)),
        (int(0.16 * size), int(0.58 * size)),
        color=1.2,
        thickness=max(2, size // 32),
        lineType=cv2.LINE_AA,
    )
    img += apply_lowpass_filter(lshape, sigma=0.6)

    # A small notch/dot pair to break residual accidental symmetries.
    img[int(0.22 * size) : int(0.27 * size), int(0.76 * size) : int(0.81 * size)] -= 1.4
    img[int(0.84 * size), int(0.57 * size)] += 2.0

    return normalize_template(img, geo)


def add_noise(img: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    if noise <= 0:
        return img.astype(np.float32, copy=True)
    noisy = img + rng.normal(0.0, noise * float(np.std(img)), img.shape).astype(np.float32)
    return noisy.astype(np.float32, copy=False)


def ideal_correction_for_rotate_then_shift(applied_angle: float, applied_dy: float, applied_dx: float) -> tuple[float, float, float]:
    """Inverse for transformed = T(shift) after R(angle), applied as R(inv_angle) then T(inv_shift').

    transform_final_image applies rotation first and then image-space shift. To
    invert T(s) * R(a), the equivalent R(-a) then T(s') has s' = -R(-a) s.
    """
    ideal_angle = normalize_angle(-applied_angle)
    theta = math.radians(ideal_angle)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    # OpenCV/image coordinates: x' = cos*x + sin*y, y' = -sin*x + cos*y for positive angle.
    rot_dx = cos_t * applied_dx + sin_t * applied_dy
    rot_dy = -sin_t * applied_dx + cos_t * applied_dy
    return ideal_angle, -rot_dy, -rot_dx


def evaluate(test_name: str, gt: np.ndarray, corrected: np.ndarray, applied_angle: float, applied_dy: float, applied_dx: float,
             ideal_angle: float, ideal_dy: float, ideal_dx: float, estimated_angle: float, estimated_dy: float, estimated_dx: float,
             pass_condition: bool, notes: str, corr_threshold: float = 0.98, **diagnostics) -> ResultRow:
    corr = normalized_correlation(corrected, gt)
    err = nrmse(corrected, gt)
    return ResultRow(
        test_name=test_name,
        applied_angle=float(applied_angle),
        applied_dy=float(applied_dy),
        applied_dx=float(applied_dx),
        ideal_angle=float(ideal_angle),
        ideal_dy=float(ideal_dy),
        ideal_dx=float(ideal_dx),
        estimated_angle=float(estimated_angle),
        estimated_dy=float(estimated_dy),
        estimated_dx=float(estimated_dx),
        angle_error_deg=float(angle_error(estimated_angle, ideal_angle)),
        dy_error_px=float(estimated_dy - ideal_dy),
        dx_error_px=float(estimated_dx - ideal_dx),
        corr=float(corr),
        nrmse=float(err),
        passed=bool(pass_condition and corr > corr_threshold),
        notes=notes,
        **diagnostics,
    )


def write_failure_png(path: Path, gt: np.ndarray, transformed: np.ndarray, corrected: np.ndarray) -> None:
    def scale(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        lo, hi = np.percentile(x, [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        return np.clip((x - lo) / (hi - lo), 0, 1)

    diff = corrected - gt
    panels = [scale(gt), scale(transformed), scale(corrected), scale(np.abs(diff))]
    overview = np.concatenate(panels, axis=1)
    cv2.imwrite(str(path), (overview * 255).astype(np.uint8))


def brute_force_rotation(img: np.ndarray, ref: np.ndarray, geo, step: float = 2.0) -> tuple[float, float]:
    best_angle = 0.0
    best_corr = -float("inf")
    for cand in np.arange(-180.0, 180.0, step):
        rotated = rotate_image(img, geo, float(cand))
        corr = normalized_correlation(rotated, ref)
        if corr > best_corr:
            best_corr = corr
            best_angle = float(cand)
    return normalize_angle(best_angle), float(best_corr)

def brute_force_rotation_translation(img, ref, geo, step=2.0):
    best = {
        "angle": 0.0,
        "dy": 0.0,
        "dx": 0.0,
        "corr": -np.inf,
        "fft_score": -np.inf,
    }

    for cand in np.arange(-180.0, 180.0, step):
        # Apply candidate correction angle first
        rotated = rotate_image(img, geo, float(cand))

        # FFT returns observed displacement, so correction is negative
        obs_dy, obs_dx, fft_score = get_best_translation_fft(rotated, ref, geo)
        corr_dy = -obs_dy
        corr_dx = -obs_dx

        corrected = transform_final_image(img, geo, float(cand), corr_dy, corr_dx)
        corr = normalized_correlation(corrected, ref)

        if corr > best["corr"]:
            best = {
                "angle": normalize_angle(float(cand)),
                "dy": float(corr_dy),
                "dx": float(corr_dx),
                "corr": float(corr),
                "fft_score": float(fft_score),
            }

    return best

def fourier_mellin_diagnostics(img: np.ndarray, ref: np.ndarray, geo, top_k: int = 5) -> dict:
    """Return Fourier-Mellin peak diagnostics without changing estimator behavior."""
    fm_angle, profile = get_coarse_angle_fourier_mellin_profile(img, ref, geo)
    valid_idx = _valid_fm_lag_indices(geo, profile_size=profile.size)
    valid_idx = valid_idx[(valid_idx >= 0) & (valid_idx < profile.size)]
    valid_idx = valid_idx[np.isfinite(profile[valid_idx])]
    if valid_idx.size == 0:
        return {
            "fm_angle": normalize_angle(float(fm_angle)),
            "fm_peak_index": -1,
            "fm_lag": 0,
            "top5_fm_angles": "",
            "top5_fm_scores": "",
        }

    ordered = valid_idx[np.argsort(profile[valid_idx])[::-1]]
    peak_index = int(ordered[0])
    fft_len = int(profile.size)
    fm_lag = int(peak_index - fft_len if peak_index > fft_len // 2 else peak_index)
    top = ordered[:top_k]
    top_angles = [normalize_angle(_angle_from_fm_profile_index(int(idx), geo)) for idx in top]
    top_scores = [float(profile[int(idx)]) for idx in top]
    return {
        "fm_angle": normalize_angle(float(fm_angle)),
        "fm_peak_index": peak_index,
        "fm_lag": fm_lag,
        "top5_fm_angles": ";".join(f"{angle:.3f}" for angle in top_angles),
        "top5_fm_scores": ";".join(f"{score:.6g}" for score in top_scores),
    }


def _fm_profile_correlation(prof_img: np.ndarray, prof_ref: np.ndarray, geo) -> np.ndarray:
    """Compute the same padded 1D FFT correlation used by the FM estimator."""
    prof_img = np.asarray(prof_img, dtype=np.float32).ravel()
    prof_ref = np.asarray(prof_ref, dtype=np.float32).ravel()
    if prof_img.size != prof_ref.size:
        raise ValueError(f"Profile lengths differ: {prof_img.size} != {prof_ref.size}")

    # Match the production estimator for the square minimal benchmark while
    # allowing the axis diagnostic to stay meaningful if W != H in the future.
    pad_len = int(prof_img.size)
    fft_len = int(2 ** np.ceil(np.log2(2 * prof_img.size - 1)))
    if prof_img.size == int(geo.H):
        pad_len = int(geo.prof_pad_len)
        fft_len = int(geo.prof_fft_len)

    prof_img_pad = np.pad(prof_img, (0, pad_len))
    prof_ref_pad = np.pad(prof_ref, (0, pad_len))
    fp_img = np.fft.fft(prof_img_pad, n=fft_len)
    fp_ref = np.fft.fft(prof_ref_pad, n=fft_len)
    return np.real(np.fft.ifft(fp_img * np.conj(fp_ref))).astype(np.float32, copy=False)


def _fm_profile_peak_diagnostics(profile: np.ndarray, geo, top_k: int = 5) -> dict:
    """Return best lag/angle and top-K valid peaks for a 1D FM score profile."""
    profile = np.asarray(profile, dtype=np.float32).ravel()
    valid_idx = _valid_fm_lag_indices(geo, profile_size=profile.size)
    valid_idx = valid_idx[(valid_idx >= 0) & (valid_idx < profile.size)]
    valid_idx = valid_idx[np.isfinite(profile[valid_idx])]
    if valid_idx.size == 0:
        return {
            "best_angle": float("nan"),
            "best_lag": 0,
            "top5_angles": "",
            "top5_scores": "",
        }

    ordered = valid_idx[np.argsort(profile[valid_idx])[::-1]]
    peak_index = int(ordered[0])
    fft_len = int(profile.size)
    best_lag = int(peak_index - fft_len if peak_index > fft_len // 2 else peak_index)
    top = ordered[:top_k]
    top_angles = [normalize_angle(_angle_from_fm_profile_index(int(idx), geo)) for idx in top]
    top_scores = [float(profile[int(idx)]) for idx in top]
    return {
        "best_angle": normalize_angle(_angle_from_fm_profile_index(peak_index, geo)),
        "best_lag": best_lag,
        "top5_angles": ";".join(f"{angle:.3f}" for angle in top_angles),
        "top5_scores": ";".join(f"{score:.6g}" for score in top_scores),
    }


def _standardize_profile(profile: np.ndarray) -> np.ndarray:
    """Return a zero-mean/unit-std profile for diagnostic-only FM variants."""
    profile = np.asarray(profile, dtype=np.float32).ravel()
    centered = profile - float(np.mean(profile))
    std = float(np.std(centered))
    if std > 1e-12:
        centered = centered / std
    return centered.astype(np.float32, copy=False)


def _fourier_mellin_polar_images(img: np.ndarray, ref: np.ndarray, geo) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce current Fourier-Mellin preprocessing through linearPolar."""
    h, w = img.shape
    if h != geo.H or w != geo.W:
        raise ValueError(f"Image shape {img.shape} does not match geo ({geo.H}, {geo.W})")

    window = geo.window
    F_img = np.abs(np.fft.fftshift(np.fft.fft2(img * window)))
    F_ref = np.abs(np.fft.fftshift(np.fft.fft2(ref * window)))
    F_img = np.log(F_img + 1)
    F_ref = np.log(F_ref + 1)

    center = (geo.cx, geo.cy)
    max_radius = geo.max_radius
    polar_img = cv2.linearPolar(F_img, center, max_radius, cv2.WARP_FILL_OUTLIERS)
    polar_ref = cv2.linearPolar(F_ref, center, max_radius, cv2.WARP_FILL_OUTLIERS)
    return polar_img, polar_ref


def _lag_from_fm_profile_index(index: int, profile_size: int) -> int:
    """Convert an FFT-correlation index to a signed linear-correlation lag."""
    return int(index) - int(profile_size) if int(index) > int(profile_size) // 2 else int(index)


def _polar2d_ncc_profile(polar_img: np.ndarray, polar_ref: np.ndarray, geo) -> np.ndarray:
    """Score each valid angular lag by 2D polar normalized correlation."""
    fft_len = int(geo.prof_fft_len)
    profile = np.full(fft_len, -np.inf, dtype=np.float32)
    valid_idx = _valid_fm_lag_indices(geo, profile_size=fft_len)
    valid_idx = valid_idx[(valid_idx >= 0) & (valid_idx < fft_len)]
    for idx in valid_idx:
        lag = _lag_from_fm_profile_index(int(idx), fft_len)
        shifted = np.roll(polar_img, -lag, axis=0)
        profile[int(idx)] = normalized_correlation(shifted, polar_ref)
    return profile


def fourier_mellin_variant_diagnostics(img: np.ndarray, ref: np.ndarray, geo, top_k: int = 5) -> dict:
    """Run diagnostic-only Fourier-Mellin variants without touching production code."""
    polar_img, polar_ref = _fourier_mellin_polar_images(img, ref, geo)

    prof_img = np.sum(polar_img, axis=1)
    prof_ref = np.sum(polar_ref, axis=1)
    zero_mean_profile = _fm_profile_correlation(
        _standardize_profile(prof_img),
        _standardize_profile(prof_ref),
        geo,
    )

    polar_img_highpass = polar_img.copy()
    polar_ref_highpass = polar_ref.copy()
    r_min = int(0.15 * polar_img_highpass.shape[1])
    polar_img_highpass[:, :r_min] = 0
    polar_ref_highpass[:, :r_min] = 0
    highpass_profile = _fm_profile_correlation(
        np.sum(polar_img_highpass, axis=1),
        np.sum(polar_ref_highpass, axis=1),
        geo,
    )

    weights = np.linspace(0.0, 1.0, polar_img.shape[1], dtype=np.float32)
    radial_weight_profile = _fm_profile_correlation(
        np.sum(polar_img * weights[None, :], axis=1),
        np.sum(polar_ref * weights[None, :], axis=1),
        geo,
    )

    polar2d_profile = _polar2d_ncc_profile(polar_img, polar_ref, geo)

    return {
        "zero_mean": _fm_profile_peak_diagnostics(zero_mean_profile, geo, top_k=top_k),
        "highpass": _fm_profile_peak_diagnostics(highpass_profile, geo, top_k=top_k),
        "radial_weight": _fm_profile_peak_diagnostics(radial_weight_profile, geo, top_k=top_k),
        "polar2d": _fm_profile_peak_diagnostics(polar2d_profile, geo, top_k=top_k),
    }


def fourier_mellin_axis_diagnostics(img: np.ndarray, ref: np.ndarray, geo, top_k: int = 5) -> dict:
    """Compare both polar summation axes for Fourier-Mellin angle profiles."""
    polar_img, polar_ref = _fourier_mellin_polar_images(img, ref, geo)

    axis0_profile = _fm_profile_correlation(
        np.sum(polar_img, axis=0),
        np.sum(polar_ref, axis=0),
        geo,
    )
    axis1_profile = _fm_profile_correlation(
        np.sum(polar_img, axis=1),
        np.sum(polar_ref, axis=1),
        geo,
    )

    return {
        "axis0": _fm_profile_peak_diagnostics(axis0_profile, geo, top_k=top_k),
        "axis1": _fm_profile_peak_diagnostics(axis1_profile, geo, top_k=top_k),
    }

def append_result(rows: list[ResultRow], row: ResultRow, output_dir: Path, gt: np.ndarray, transformed: np.ndarray, corrected: np.ndarray) -> None:
    rows.append(row)
    if not row.passed:
        safe = f"{len(rows):03d}_{row.test_name}_a{row.applied_angle:g}_dy{row.applied_dy:g}_dx{row.applied_dx:g}.png".replace("-", "m")
        write_failure_png(output_dir / safe, gt, transformed, corrected)
        print(
            f"FAIL {row.test_name}: angle={row.applied_angle:g}, dy={row.applied_dy:g}, dx={row.applied_dx:g}, "
            f"corr={row.corr:.4f}, nrmse={row.nrmse:.4f}, notes={row.notes}"
        )


def run_transform_inverse(gt: np.ndarray, geo, rows: list[ResultRow], output_dir: Path) -> None:
    cases = [(0, 0, 0), (15, 0, 0), (-37, 0, 0), (0, 5, -7), (23, 4, -6), (-71, -5, 8), (135, 7, 3)]
    for angle, dy, dx in cases:
        transformed = shift_image(rotate_image(gt, geo, angle), geo, dy, dx)
        ideal_angle, ideal_dy, ideal_dx = ideal_correction_for_rotate_then_shift(angle, dy, dx)
        corrected = transform_final_image(transformed, geo, ideal_angle, ideal_dy, ideal_dx)
        row = evaluate(
            "transform_inverse_test", gt, corrected, angle, dy, dx,
            ideal_angle, ideal_dy, ideal_dx, ideal_angle, ideal_dy, ideal_dx,
            True, "analytic inverse via transform_final_image",
        )
        append_result(rows, row, output_dir, gt, transformed, corrected)


def run_translation_sign_diagnostic(gt: np.ndarray, geo, rows: list[ResultRow], output_dir: Path) -> None:
    for dy, dx in [(5, -7), (-9, 4), (11, 12)]:
        transformed = shift_image(gt, geo, dy, dx)
        raw_fft_dy, raw_fft_dx, score = get_best_translation_fft(transformed, gt, geo)
        negated_fft_dy = -raw_fft_dy
        negated_fft_dx = -raw_fft_dx
        ideal_angle, ideal_dy, ideal_dx = 0.0, -dy, -dx
        corrected = transform_final_image(transformed, geo, 0.0, negated_fft_dy, negated_fft_dx)
        print(
            "translation_sign_diagnostic_test: "
            f"applied=({dy:g}, {dx:g}), "
            f"ideal_correction=({ideal_dy:g}, {ideal_dx:g}), "
            f"raw_fft=({raw_fft_dy:g}, {raw_fft_dx:g}), "
            f"negated_fft=({negated_fft_dy:g}, {negated_fft_dx:g})"
        )
        row = evaluate(
            "translation_sign_diagnostic_test", gt, corrected, 0, dy, dx,
            ideal_angle, ideal_dy, ideal_dx, 0.0, negated_fft_dy, negated_fft_dx,
            abs(negated_fft_dy - ideal_dy) <= 1 and abs(negated_fft_dx - ideal_dx) <= 1,
            f"fft_score={score:.6g}; raw FFT displacement is stored separately",
            raw_fft_dy=raw_fft_dy, raw_fft_dx=raw_fft_dx,
            negated_fft_dy=negated_fft_dy, negated_fft_dx=negated_fft_dx,
        )
        append_result(rows, row, output_dir, gt, transformed, corrected)


def run_translation_only(gt: np.ndarray, geo, rows: list[ResultRow], output_dir: Path) -> None:
    for dy, dx in [(0, 0), (5, -7), (-9, 4), (11, 12)]:
        transformed = shift_image(gt, geo, dy, dx)
        obs_dy, obs_dx, score = get_best_translation_fft(transformed, gt, geo)
        correction_dy = -obs_dy
        correction_dx = -obs_dx
        ideal_angle, ideal_dy, ideal_dx = 0.0, -dy, -dx
        corrected = transform_final_image(transformed, geo, 0.0, correction_dy, correction_dx)
        row = evaluate(
            "translation_only_oracle_test", gt, corrected, 0, dy, dx,
            ideal_angle, ideal_dy, ideal_dx, 0.0, correction_dy, correction_dx,
            abs(correction_dy - ideal_dy) <= 1 and abs(correction_dx - ideal_dx) <= 1,
            f"fft_score={score:.6g}; raw FFT displacement negated for correction",
            raw_fft_dy=obs_dy, raw_fft_dx=obs_dx,
            negated_fft_dy=correction_dy, negated_fft_dx=correction_dx,
        )
        append_result(rows, row, output_dir, gt, transformed, corrected)


def run_rotation_only_bruteforce(gt: np.ndarray, geo, rows: list[ResultRow], output_dir: Path, step: float = 2.0) -> None:
    for angle in [0, 12, -34, 77, -123, 179]:
        transformed = rotate_image(gt, geo, angle)
        best_angle, best_corr = brute_force_rotation(transformed, gt, geo, step=step)
        ideal_angle = normalize_angle(-angle)
        corrected = transform_final_image(transformed, geo, best_angle, 0.0, 0.0)
        row = evaluate(
            "rotation_only_bruteforce_test", gt, corrected, angle, 0, 0,
            ideal_angle, 0, 0, best_angle, 0, 0,
            angle_error(best_angle, ideal_angle) <= step + 0.25,
            f"bruteforce_step={step:g}; best_corr={best_corr:.6f}",
        )
        append_result(rows, row, output_dir, gt, transformed, corrected)


def run_rotation_only_fm(gt: np.ndarray, geo, rows: list[ResultRow], output_dir: Path, step: float = 2.0) -> None:
    for angle in [0, 12, -34, 77, -123, 179]:
        transformed = rotate_image(gt, geo, angle)
        fm_diag = fourier_mellin_diagnostics(transformed, gt, geo, top_k=5)
        axis_diag = fourier_mellin_axis_diagnostics(transformed, gt, geo, top_k=5)
        variant_diag = fourier_mellin_variant_diagnostics(transformed, gt, geo, top_k=5)
        fm_angle = float(fm_diag["fm_angle"])
        axis0 = axis_diag["axis0"]
        axis1 = axis_diag["axis1"]
        zero_mean = variant_diag["zero_mean"]
        highpass = variant_diag["highpass"]
        radial_weight = variant_diag["radial_weight"]
        polar2d = variant_diag["polar2d"]
        brute_angle, _ = brute_force_rotation(transformed, gt, geo, step=step)
        ideal_angle = normalize_angle(-angle)
        corrected = transform_final_image(transformed, geo, fm_angle, 0.0, 0.0)
        fm_to_brute = angle_error(fm_angle, brute_angle)
        fm_to_ideal = angle_error(fm_angle, ideal_angle)
        print(
            "rotation_only_fourier_mellin_variant_diagnostic: "
            f"applied_angle={angle:g}, "
            f"ideal_angle={ideal_angle:.3f}, "
            f"brute_force_angle={brute_angle:.3f}, "
            f"current_fm_angle={fm_angle:.3f}, "
            f"current_top5={fm_diag['top5_fm_angles']}, "
            f"zero_mean_best={float(zero_mean['best_angle']):.3f}, "
            f"zero_mean_top5={zero_mean['top5_angles']}, "
            f"highpass_best={float(highpass['best_angle']):.3f}, "
            f"highpass_top5={highpass['top5_angles']}, "
            f"radial_weight_best={float(radial_weight['best_angle']):.3f}, "
            f"radial_weight_top5={radial_weight['top5_angles']}, "
            f"polar2d_best={float(polar2d['best_angle']):.3f}, "
            f"polar2d_top5={polar2d['top5_angles']}"
        )
        row = evaluate(
            "rotation_only_fourier_mellin_test", gt, corrected, angle, 0, 0,
            ideal_angle, 0, 0, fm_angle, 0, 0,
            fm_to_ideal <= max(step, 360.0 / geo.H) + 1.0,
            (
                f"bruteforce_angle={brute_angle:.3f}; "
                f"fm_vs_bruteforce_error={fm_to_brute:.3f}; "
                f"fm_vs_ideal_error={fm_to_ideal:.3f}"
            ),
            fm_angle=fm_angle,
            bruteforce_angle=brute_angle,
            fm_vs_bruteforce_error=fm_to_brute,
            fm_vs_ideal_error=fm_to_ideal,
            fm_peak_index=int(fm_diag["fm_peak_index"]),
            fm_lag=int(fm_diag["fm_lag"]),
            top5_fm_angles=str(fm_diag["top5_fm_angles"]),
            top5_fm_scores=str(fm_diag["top5_fm_scores"]),
            fm_axis0_best_angle=float(axis0["best_angle"]),
            fm_axis0_top5_angles=str(axis0["top5_angles"]),
            fm_axis1_best_angle=float(axis1["best_angle"]),
            fm_axis1_top5_angles=str(axis1["top5_angles"]),
            fm_zm_best_angle=float(zero_mean["best_angle"]),
            fm_zm_top5_angles=str(zero_mean["top5_angles"]),
            fm_highpass_best_angle=float(highpass["best_angle"]),
            fm_highpass_top5_angles=str(highpass["top5_angles"]),
            fm_radial_weight_best_angle=float(radial_weight["best_angle"]),
            fm_radial_weight_top5_angles=str(radial_weight["top5_angles"]),
            fm_polar2d_best_angle=float(polar2d["best_angle"]),
            fm_polar2d_top5_angles=str(polar2d["top5_angles"]),
        )
        append_result(rows, row, output_dir, gt, transformed, corrected)


def run_rotation_translation_oracle(gt: np.ndarray, geo, rows: list[ResultRow], output_dir: Path, step: float = 2.0) -> None:
    for angle, dy, dx in [(17, 5, -7), (-41, -8, 6), (93, 7, 9)]:
        transformed = shift_image(rotate_image(gt, geo, angle), geo, dy, dx)
        ideal_angle, ideal_dy, ideal_dx = ideal_correction_for_rotate_then_shift(angle, dy, dx)

        # Known ideal correction angle, estimate observed residual after rotation,
        # then negate it to get the correction shift.
        rotated = rotate_image(transformed, geo, ideal_angle)
        obs_dy, obs_dx, score = get_best_translation_fft(rotated, gt, geo)
        corr_dy = -obs_dy
        corr_dx = -obs_dx
        corrected = transform_final_image(transformed, geo, ideal_angle, corr_dy, corr_dx)
        row = evaluate(
            "rotation_translation_oracle_angle_test", gt, corrected, angle, dy, dx,
            ideal_angle, ideal_dy, ideal_dx, ideal_angle, corr_dy, corr_dx,
            abs(corr_dy - ideal_dy) <= 1.5 and abs(corr_dx - ideal_dx) <= 1.5,
            f"known angle; fft_score={score:.6g}; observed residual negated for correction",
            raw_fft_dy=obs_dy, raw_fft_dx=obs_dx,
            negated_fft_dy=corr_dy, negated_fft_dx=corr_dx,
        )
        append_result(rows, row, output_dir, gt, transformed, corrected)

        # Joint brute-force search:
        # For each candidate angle, estimate translation and score the fully corrected image.
        best = brute_force_rotation_translation(transformed, gt, geo, step=step)
        
        best_angle = best["angle"]
        corr_dy = best["dy"]
        corr_dx = best["dx"]
        score = best["fft_score"]
        best_corr = best["corr"]
        
        corrected = transform_final_image(transformed, geo, best_angle, corr_dy, corr_dx)
        
        row = evaluate(
            "rotation_translation_joint_bruteforce_test", gt, corrected, angle, dy, dx,
            ideal_angle, ideal_dy, ideal_dx, best_angle, corr_dy, corr_dx,
            angle_error(best_angle, ideal_angle) <= step + 0.25
            and abs(corr_dy - ideal_dy) <= 2.0
            and abs(corr_dx - ideal_dx) <= 2.0,
            (
                f"joint_bruteforce_step={step:g}; "
                f"best_corrected_corr={best_corr:.6f}; "
                f"fft_score={score:.6g}; "
                f"angle+translation selected jointly"
            ),
            negated_fft_dy=corr_dy,
            negated_fft_dx=corr_dx,
        )
        append_result(rows, row, output_dir, gt, transformed, corrected)

        # Coarse-to-fine joint search should recover the same cases with fewer
        # angle evaluations than a full 2-degree brute-force scan.
        ctf_best = coarse_to_fine_joint_search(transformed, gt, geo)
        ctf_angle = ctf_best["angle"]
        ctf_dy = ctf_best["dy"]
        ctf_dx = ctf_best["dx"]
        ctf_score = ctf_best["fft_score"]
        ctf_corr = ctf_best["score"]
        ctf_evals = int(ctf_best.get("n_evaluations", 0))

        ctf_corrected = transform_final_image(transformed, geo, ctf_angle, ctf_dy, ctf_dx)

        row = evaluate(
            "rotation_translation_coarse_to_fine_joint_test", gt, ctf_corrected, angle, dy, dx,
            ideal_angle, ideal_dy, ideal_dx, ctf_angle, ctf_dy, ctf_dx,
            angle_error(ctf_angle, ideal_angle) <= 0.75
            and abs(ctf_dy - ideal_dy) <= 2.0
            and abs(ctf_dx - ideal_dx) <= 2.0
            and ctf_evals < len(np.arange(-180.0, 180.0, step)),
            (
                f"coarse_to_fine_joint; evals={ctf_evals}; "
                f"full_2deg_evals={len(np.arange(-180.0, 180.0, step))}; "
                f"best_corrected_corr={ctf_corr:.6f}; fft_score={ctf_score:.6g}"
            ),
            negated_fft_dy=ctf_dy,
            negated_fft_dx=ctf_dx,
        )
        append_result(rows, row, output_dir, gt, transformed, ctf_corrected)


def run_noise_sweep(gt: np.ndarray, geo, rows: list[ResultRow], output_dir: Path, step: float = 2.0) -> None:
    rng = np.random.default_rng(12345)
    angle, dy, dx = 23.0, 4.0, -6.0
    clean_transformed = shift_image(rotate_image(gt, geo, angle), geo, dy, dx)
    ideal_angle, ideal_dy, ideal_dx = ideal_correction_for_rotate_then_shift(angle, dy, dx)
    for noise in [0, 0.05, 0.1, 0.2, 0.35]:
        transformed = add_noise(clean_transformed, noise, rng)

        # Brute-force angle then FFT observed residual, deliberately simple staged estimator.
        best = brute_force_rotation_translation(transformed, gt, geo, step=step)

        best_angle = best["angle"]
        corr_dy = best["dy"]
        corr_dx = best["dx"]
        score = best["fft_score"]
        best_corr = best["corr"]
        
        corrected = transform_final_image(transformed, geo, best_angle, corr_dy, corr_dx)
        corr = normalized_correlation(corrected, gt)
        
        pass_condition = corr > (0.98 if noise == 0 else 0.80)
        row = evaluate(
            "noise_sweep_joint_bruteforce_test", gt, corrected, angle, dy, dx,
            ideal_angle, ideal_dy, ideal_dx, best_angle, corr_dy, corr_dx,
            angle_error(best_angle, ideal_angle) <= step + 0.25
            and abs(corr_dy - ideal_dy) <= 2.5
            and abs(corr_dx - ideal_dx) <= 2.5,
            (
                f"noise={noise:g}; joint angle+translation brute force; "
                f"best_corrected_corr={best_corr:.6f}; fft_score={score:.6g}"
            ),
            corr_threshold=(0.98 if noise == 0 else 0.80),
            negated_fft_dy=corr_dy,
            negated_fft_dx=corr_dx,
        )
        
        
        append_result(rows, row, output_dir, gt, transformed, corrected)


def write_csv(rows: list[ResultRow], output_dir: Path) -> Path:
    csv_path = output_dir / "minimal_alignment_tests.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return csv_path


def summarize(rows: list[ResultRow], csv_path: Path) -> None:
    print(f"\nWrote {csv_path}")
    print(f"Rows: {len(rows)}; passed: {sum(r.passed for r in rows)}; failed: {sum(not r.passed for r in rows)}")
    first_fail = next((r for r in rows if not r.passed), None)
    if first_fail is None:
        print("First failing stage: none")
    else:
        print(f"First failing stage: {first_fail.test_name} ({first_fail.notes})")
    for name in dict.fromkeys(r.test_name for r in rows):
        group = [r for r in rows if r.test_name == name]
        print(
            f"{name}: pass={sum(r.passed for r in group)}/{len(group)}, "
            f"median_corr={np.median([r.corr for r in group]):.4f}, "
            f"median_angle_err={np.median([r.angle_error_deg for r in group]):.3f}, "
            f"median_abs_shift_err=({np.median([abs(r.dy_error_px) for r in group]):.3f}, "
            f"{np.median([abs(r.dx_error_px) for r in group]):.3f})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=96, help="Synthetic square image size (default: 96).")
    parser.add_argument("--output-dir", type=Path, default=Path("minimal_report"), help="Directory for CSV and failure PNGs.")
    parser.add_argument("--angle-step", type=float, default=2.0, help="Brute-force rotation grid step in degrees.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    geo = get_geometry_context((args.size, args.size))
    gt = make_asymmetric_template(args.size)

    rows: list[ResultRow] = []
    run_transform_inverse(gt, geo, rows, args.output_dir)
    run_translation_sign_diagnostic(gt, geo, rows, args.output_dir)
    run_translation_only(gt, geo, rows, args.output_dir)
    run_rotation_only_bruteforce(gt, geo, rows, args.output_dir, step=args.angle_step)
    run_rotation_only_fm(gt, geo, rows, args.output_dir, step=args.angle_step)
    run_rotation_translation_oracle(gt, geo, rows, args.output_dir, step=args.angle_step)
    run_noise_sweep(gt, geo, rows, args.output_dir, step=args.angle_step)

    csv_path = write_csv(rows, args.output_dir)
    summarize(rows, csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
