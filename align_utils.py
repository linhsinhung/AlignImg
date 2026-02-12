#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CryoEM 2D Alignment Utilities (CPU Fundamental v9)
Unified Math: Uses Float Center and FFT Correlation to match GPU logic.
"""

import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import center_of_mass

# === 0. Helper & Geometry Context ===

def normalize_angle(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0

class GeometryContext:
    """
    CPU geometry/cache object. Attribute style access to align with GPU usage:
        geo.cy, geo.cx, geo.window, geo.max_radius, ...
    """

    def __init__(self, shape):
        if len(shape) == 3:
            self.H, self.W = shape[1], shape[2]
        else:
            self.H, self.W = shape

        self.cy = self.H / 2.0
        self.cx = self.W / 2.0

        # Integer centers for FFT-shift reference
        self.cy_int = int(self.H // 2)
        self.cx_int = int(self.W // 2)

        self.max_radius = min(self.H, self.W) / 2.0

        # Window function (Hanning)
        hann_h = np.hanning(self.H)
        hann_w = np.hanning(self.W)
        self.window = np.outer(hann_h, hann_w).astype(np.float32)

        # Precompute grids / distances (float32 is enough)
        y_idx, x_idx = np.indices((self.H, self.W), dtype=np.float32)
        self.grid_y = y_idx
        self.grid_x = x_idx
        self.dist_sq = (x_idx - self.cx) ** 2 + (y_idx - self.cy) ** 2

        # Cache for circular masks
        self._mask_cache: dict[tuple[float | None, int], np.ndarray] = {}

        # Cache for 1D profile correlation in Fourier-Mellin
        # pad_len = H (same as your original logic), fft_len derived from H only
        self.prof_pad_len = self.H
        self.prof_fft_len = int(2 ** np.ceil(np.log2(2 * self.H - 1)))

    def get_circular_mask(self, diameter=None, soft_edge=5) -> np.ndarray:
        """
        Cached circular mask generator.

        Args:
            diameter: mask diameter in pixels; if None uses (max_radius - 1)*2
            soft_edge: cosine soft edge width (pixels)

        Returns:
            mask: (H, W) float32
        """
        key = (None if diameter is None else float(diameter), int(soft_edge))
        if key in self._mask_cache:
            return self._mask_cache[key]

        radius = (self.max_radius - 1.0) if diameter is None else (float(diameter) / 2.0)
        soft_edge_f = float(soft_edge)

        # Guard: soft_edge > radius can make inner negative
        inner = max(0.0, radius - soft_edge_f)

        mask = np.zeros((self.H, self.W), dtype=np.float32)
        dsq = self.dist_sq

        mask[dsq < inner**2] = 1.0

        edge = (dsq >= inner**2) & (dsq < radius**2)
        if np.any(edge) and soft_edge_f > 0:
            d = np.sqrt(dsq[edge])
            t = (d - inner) / soft_edge_f  # in [0, 1)
            mask[edge] = 0.5 * (1.0 + np.cos(t * np.pi))

        self._mask_cache[key] = mask
        return mask

def get_geometry_context(shape) -> GeometryContext:
    return GeometryContext(shape)

# === 1. Pre-processing ===

def apply_circular_mask(img: np.ndarray, geo: GeometryContext, diameter=None, soft_edge=5) -> np.ndarray:
    return img * geo.get_circular_mask(diameter=diameter, soft_edge=soft_edge)

def apply_lowpass_filter(img: np.ndarray, sigma=2.0) -> np.ndarray:
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma)

def calculate_center_of_mass_shift(img: np.ndarray, geo: GeometryContext, sigma=5):
    # Use cv2 to match Multi version
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    blurred = blurred - np.min(blurred)
    cy_com, cx_com = center_of_mass(blurred)

    dy = geo.cy - cy_com
    dx = geo.cx - cx_com
    return float(dy), float(dx)

# === 2. Geometric Transformations ===

def shift_image(img: np.ndarray, geo: GeometryContext, dy: float, dx: float) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (geo.W, geo.H), flags=cv2.INTER_LINEAR)


def rotate_image(img: np.ndarray, geo: GeometryContext, angle: float) -> np.ndarray:
    center = (geo.cx, geo.cy)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (geo.W, geo.H), flags=cv2.INTER_LINEAR)


def transform_final_image(img: np.ndarray, geo: GeometryContext, angle: float, dy: float, dx: float) -> np.ndarray:
    """
    Final transform application.

    IMPORTANT:
    - The sign conventions here are intentionally kept as your existing calibrated behavior.
    - Do NOT change unless re-validating CPU/GPU consistency on real datasets.
    """
    angle = normalize_angle(angle)

    # 1. Rotate
    M_rot = cv2.getRotationMatrix2D((geo.cx, geo.cy), angle, 1.0)
    rot_img = cv2.warpAffine(img, M_rot, (geo.W, geo.H), flags=cv2.INTER_LINEAR)

    # 2. Total Shift (keep your calibrated double-negative convention)
    total_dx = dx
    total_dy = dy
    M_shift = np.float32([[1, 0, total_dx], [0, 1, total_dy]])

    return cv2.warpAffine(rot_img, M_shift, (geo.W, geo.H), flags=cv2.INTER_LINEAR)

# === 3. Alignment Logic ===

def get_coarse_angle_fourier_mellin(img: np.ndarray, ref: np.ndarray, geo: GeometryContext) -> float:
    """
    Coarse rotation estimate via Fourier-Mellin approach.

    Uses:
    - geo.window
    - geo.max_radius
    - cached geo.prof_pad_len / geo.prof_fft_len for 1D FFT correlation
    """
    h, w = img.shape
    if h != geo.H or w != geo.W:
        # Safety: geometry must match image size
        raise ValueError(f"Image shape {img.shape} does not match geo ({geo.H}, {geo.W})")

    window = geo.window

    F_img = np.abs(fftshift(fft2(img * window)))
    F_ref = np.abs(fftshift(fft2(ref * window)))
    F_img = np.log(F_img + 1)
    F_ref = np.log(F_ref + 1)

    center = (geo.cx, geo.cy)
    max_radius = geo.max_radius

    polar_img = cv2.linearPolar(F_img, center, max_radius, cv2.WARP_FILL_OUTLIERS)
    polar_ref = cv2.linearPolar(F_ref, center, max_radius, cv2.WARP_FILL_OUTLIERS)

    prof_img = np.sum(polar_img, axis=1)
    prof_ref = np.sum(polar_ref, axis=1)

    # FFT correlation (cached lengths)
    pad_len = geo.prof_pad_len
    fft_len = geo.prof_fft_len

    prof_img_pad = np.pad(prof_img, (0, pad_len))
    prof_ref_pad = np.pad(prof_ref, (0, pad_len))

    fp_img = np.fft.fft(prof_img_pad, n=fft_len)
    fp_ref = np.fft.fft(prof_ref_pad, n=fft_len)

    corr = np.real(np.fft.ifft(fp_img * np.conj(fp_ref)))
    shift_idx = int(np.argmax(corr))

    # Circular lag logic
    lag = shift_idx - fft_len if shift_idx > fft_len // 2 else shift_idx

    angle_est = -(float(lag) / h) * 360.0
    return normalize_angle(angle_est)

def get_best_translation_fft(img: np.ndarray, ref: np.ndarray, geo: GeometryContext):
    f_img = fft2(img)
    f_ref = fft2(ref)

    cc_spec = f_img * np.conj(f_ref)
    cc_map = np.real(ifft2(cc_spec))
    cc_map = fftshift(cc_map)

    cy, cx = np.unravel_index(np.argmax(cc_map), cc_map.shape)

    # Use integer center for FFT shift reference
    dy = float(cy) - geo.cy_int
    dx = float(cx) - geo.cx_int
    max_cc = float(cc_map[cy, cx])

    return dy, dx, max_cc

def check_180_ambiguity(img: np.ndarray, ref: np.ndarray, angle_candidate: float, geo: GeometryContext):
    rot_1 = rotate_image(img, geo, angle_candidate)
    _, _, score_1 = get_best_translation_fft(rot_1, ref, geo)

    angle_180 = angle_candidate + 180.0
    rot_2 = rotate_image(img, geo, angle_180)
    _, _, score_2 = get_best_translation_fft(rot_2, ref, geo)

    if score_2 > score_1:
        return normalize_angle(angle_180), score_2
    else:
        return normalize_angle(angle_candidate), score_1

def refine_subpixel_parabolic(scores, center_idx: int, step: float) -> float:
    n = len(scores)
    if center_idx <= 0 or center_idx >= n - 1:
        return 0.0

    y_l = scores[center_idx - 1]
    y_c = scores[center_idx]
    y_r = scores[center_idx + 1]

    denom = y_l - 2 * y_c + y_r
    if abs(denom) < 1e-6:
        return 0.0

    delta = 0.5 * (y_l - y_r) / denom
    return float(delta) * float(step)

def fine_alignment_search(
    img: np.ndarray,
    ref: np.ndarray,
    coarse_angle: float,
    geo: GeometryContext,
    search_range=5,
    step=1,
):
    best_score = -float("inf")
    best_params = {"angle": coarse_angle, "dy": 0.0, "dx": 0.0, "score": -float("inf")}

    angles = np.arange(coarse_angle - search_range, coarse_angle + search_range + step, step)
    scores = []
    best_idx = 0  # FIX: ensure initialized even if scores become non-finite

    for ang in angles:
        rot_img = rotate_image(img, geo, float(ang))
        dy, dx, score = get_best_translation_fft(rot_img, ref, geo)

        # record score for parabolic refinement
        scores.append(score)

        # FIX: guard against NaN/Inf
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_params = {"angle": float(ang), "dy": float(dy), "dx": float(dx), "score": float(score)}
            best_idx = len(scores) - 1

    delta = refine_subpixel_parabolic(scores, best_idx, step)
    best_params["angle"] = normalize_angle(best_params["angle"] + delta)
    return best_params
