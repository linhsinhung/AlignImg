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

def _angle_from_fm_profile_index(index: int, geo: GeometryContext) -> float:
    """Convert a Fourier-Mellin correlation-bin index to the legacy angle convention."""
    fft_len = geo.prof_fft_len
    lag = int(index) - fft_len if int(index) > fft_len // 2 else int(index)
    angle_est = -(float(lag) / float(geo.H)) * 360.0
    return normalize_angle(angle_est)


def _valid_fm_lag_indices(geo: GeometryContext, profile_size: int | None = None) -> np.ndarray:
    """Indices that correspond to real linear-correlation lags, not padding.

    The 1D Fourier-Mellin profile is computed with zero-padding to fft_len, so
    the middle bins do not represent valid lags.  Only lags 0..H-1 and
    -(H-1)..-1 are valid for the original angular profile length H.
    """
    fft_len = geo.prof_fft_len if profile_size is None else int(profile_size)
    h = int(geo.H)
    pos = np.arange(0, min(h, fft_len), dtype=np.int64)
    neg_start = max(0, fft_len - (h - 1))
    neg = np.arange(neg_start, fft_len, dtype=np.int64)
    return np.concatenate((pos, neg))


def _separate_normalized_angles(angles, K: int | None = None, min_separation_deg=15) -> np.ndarray:
    """Normalize angles, remove duplicates, and enforce circular separation."""
    selected: list[float] = []
    min_sep = float(min_separation_deg)
    limit = None if K is None else int(K)
    for angle in np.asarray(angles, dtype=np.float32).ravel():
        norm_angle = normalize_angle(float(angle))
        if all(abs(normalize_angle(norm_angle - prev)) >= min_sep for prev in selected):
            selected.append(norm_angle)
            if limit is not None and len(selected) >= limit:
                break
    return np.asarray(selected, dtype=np.float32)


def expand_angles_with_180_ambiguity(candidate_angles, min_separation_deg=15) -> np.ndarray:
    """Include the standard 180-degree FM ambiguity for every candidate angle."""
    expanded: list[float] = []
    for angle in np.asarray(candidate_angles, dtype=np.float32).ravel():
        expanded.append(float(angle))
        expanded.append(float(angle) + 180.0)
    return _separate_normalized_angles(expanded, K=None, min_separation_deg=min_separation_deg)


def get_coarse_angle_fourier_mellin_profile(img: np.ndarray, ref: np.ndarray, geo: GeometryContext):
    """
    Coarse rotation estimate plus full Fourier-Mellin angular score profile.

    The returned profile is the same 1D FFT correlation landscape used by
    :func:`get_coarse_angle_fourier_mellin`.  Multi-candidate alignment treats
    high-scoring peaks in this landscape as *orientation branches*; each branch
    is still verified later by image-space translation/fine-angle scoring.
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

    corr = np.real(np.fft.ifft(fp_img * np.conj(fp_ref))).astype(np.float32, copy=False)
    valid_idx = _valid_fm_lag_indices(geo, profile_size=corr.size)
    valid_idx = valid_idx[valid_idx < corr.size]
    shift_idx = int(valid_idx[np.argmax(corr[valid_idx])])
    return _angle_from_fm_profile_index(shift_idx, geo), corr


def get_coarse_angle_fourier_mellin(img: np.ndarray, ref: np.ndarray, geo: GeometryContext) -> float:
    """Coarse rotation estimate via the legacy Fourier-Mellin best peak."""
    angle, _profile = get_coarse_angle_fourier_mellin_profile(img, ref, geo)
    return angle


def find_topk_angle_peaks_from_profile(profile, geo: GeometryContext, K=5, min_separation_deg=15):
    """Return top-K separated Fourier-Mellin peak angles in [-180, 180).

    Peaks are selected from valid linear-correlation lag bins only:
    ``0..H-1`` and ``fft_len-(H-1)..fft_len-1``.  The zero-padded middle of
    the FFT correlation array is intentionally ignored because those bins do not
    correspond to possible angular lags.  No special symmetry angles are
    assumed; separation only prevents near-duplicate candidates.
    """
    prof = np.asarray(profile, dtype=np.float32).ravel()
    if prof.size == 0 or K <= 0:
        return np.empty(0, dtype=np.float32)

    valid_idx = _valid_fm_lag_indices(geo, profile_size=prof.size)
    valid_idx = valid_idx[(valid_idx >= 0) & (valid_idx < prof.size)]
    valid_idx = valid_idx[np.isfinite(prof[valid_idx])]
    if valid_idx.size == 0:
        return np.array([0.0], dtype=np.float32)

    valid_scores = prof[valid_idx]
    left = np.roll(valid_scores, 1)
    right = np.roll(valid_scores, -1)
    peak_pos = np.flatnonzero((valid_scores >= left) & (valid_scores >= right))
    peak_idx = valid_idx[peak_pos] if peak_pos.size else valid_idx

    order = peak_idx[np.argsort(prof[peak_idx])[::-1]]
    ordered_angles = [_angle_from_fm_profile_index(int(idx), geo) for idx in order]
    selected = _separate_normalized_angles(ordered_angles, K=K, min_separation_deg=min_separation_deg)

    if selected.size == 0:
        best_idx = int(valid_idx[np.argmax(prof[valid_idx])])
        selected = np.array([_angle_from_fm_profile_index(best_idx, geo)], dtype=np.float32)
    return selected


def get_topk_coarse_angles_fourier_mellin(img: np.ndarray, ref: np.ndarray, geo: GeometryContext, K=5, min_separation_deg=15):
    """Convenience wrapper returning separated Fourier-Mellin candidate angles."""
    _best_angle, profile = get_coarse_angle_fourier_mellin_profile(img, ref, geo)
    return find_topk_angle_peaks_from_profile(profile, geo, K=K, min_separation_deg=min_separation_deg)


def softmax_scores(scores, temperature):
    """Candidate-relative, scale-normalized softmax for alignment scores.

    Raw FFT correlation magnitudes can vary substantially by image/reference,
    so the score differences are divided by the candidate-set standard
    deviation before applying the annealing temperature.  This makes
    temperature meaningful across particles while remaining purely relative to
    the candidate branches being compared for one particle.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return np.empty(0, dtype=np.float32)
    finite = np.isfinite(scores)
    if not np.any(finite):
        return np.full(scores.shape, 1.0 / float(scores.size), dtype=np.float32)

    safe_scores = np.where(finite, scores, np.min(scores[finite]))
    centered = safe_scores - np.max(safe_scores)
    scale = float(np.std(safe_scores[finite])) + 1e-6
    temp = max(float(temperature), 1e-6)
    logits = centered / (temp * scale)
    logits = np.clip(logits, -80.0, 80.0)
    weights = np.exp(logits).astype(np.float32)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(scores.shape, 1.0 / float(scores.size), dtype=np.float32)
    return (weights / total).astype(np.float32)

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


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Return zero-mean normalized cross-correlation for two images."""
    aa = np.asarray(a, dtype=np.float64) - float(np.mean(a))
    bb = np.asarray(b, dtype=np.float64) - float(np.mean(b))
    denom = np.sqrt(float(np.sum(aa * aa)) * float(np.sum(bb * bb)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(aa * bb) / denom)


def joint_angle_translation_score(img: np.ndarray, ref: np.ndarray, geo: GeometryContext, angle: float):
    """Score a correction angle jointly with its FFT-estimated translation.

    ``get_best_translation_fft`` returns the observed displacement after applying
    the candidate rotation, so the correction shift stored in the result is the
    negated observed displacement.  The final score is measured in image space
    on the fully corrected image, not from the raw FFT peak magnitude.
    """
    norm_angle = normalize_angle(float(angle))
    rotated = rotate_image(img, geo, norm_angle)
    obs_dy, obs_dx, fft_score = get_best_translation_fft(rotated, ref, geo)
    corr_dy = -obs_dy
    corr_dx = -obs_dx
    corrected = transform_final_image(img, geo, norm_angle, corr_dy, corr_dx)
    score = normalized_cross_correlation(corrected, ref)
    return {
        "angle": norm_angle,
        "dy": float(corr_dy),
        "dx": float(corr_dx),
        "score": float(score),
        "fft_score": float(fft_score),
    }


def _scan_joint_angles(img: np.ndarray, ref: np.ndarray, geo: GeometryContext, angles, seen: set[float] | None = None):
    """Evaluate unique angles for coarse-to-fine joint image-space search."""
    results = []
    if seen is None:
        seen = set()
    for ang in angles:
        norm_ang = normalize_angle(float(ang))
        key = round(norm_ang, 6)
        if key in seen:
            continue
        seen.add(key)
        results.append(joint_angle_translation_score(img, ref, geo, norm_ang))
    return results


def coarse_to_fine_joint_search(
    img: np.ndarray,
    ref: np.ndarray,
    geo: GeometryContext,
    global_step=10.0,
    mid_range=12.0,
    mid_step=2.0,
    fine_range=2.0,
    fine_step=0.5,
    topk=3,
):
    """Coarse-to-fine image-space joint rotation + translation search.

    Stage 1 scans the full circle, Stage 2 refines each of the best coarse
    branches, and Stage 3 performs a narrow scan around the best mid-stage
    branch.  Candidate angles are scored only after FFT translation correction
    and normalized cross-correlation against ``ref``.
    """
    topk = max(1, int(topk))
    global_step = float(global_step)
    mid_range = float(mid_range)
    mid_step = float(mid_step)
    fine_range = float(fine_range)
    fine_step = float(fine_step)
    if global_step <= 0.0 or mid_step <= 0.0 or fine_step <= 0.0:
        raise ValueError("global_step, mid_step, and fine_step must be positive")

    seen: set[float] = set()
    all_candidates = []

    # Stage 1: full-circle scan, -180 inclusive to 180 exclusive.
    stage1_angles = np.arange(-180.0, 180.0, global_step, dtype=np.float32)
    stage1 = _scan_joint_angles(img, ref, geo, stage1_angles, seen)
    all_candidates.extend({**cand, "stage": 1} for cand in stage1)
    top = sorted(stage1, key=lambda item: item["score"], reverse=True)[:topk]

    # Stage 2: refine around the top global candidates.
    stage2 = []
    for cand in top:
        center = float(cand["angle"])
        angles = np.arange(center - mid_range, center + mid_range + 0.5 * mid_step, mid_step, dtype=np.float32)
        stage2.extend(_scan_joint_angles(img, ref, geo, angles, seen))
    all_candidates.extend({**cand, "stage": 2} for cand in stage2)
    top = sorted(top + stage2, key=lambda item: item["score"], reverse=True)[:topk]

    # Stage 3: fine scan around the best candidate from Stages 1/2.
    best_center = float(top[0]["angle"]) if top else 0.0
    fine_angles = np.arange(best_center - fine_range, best_center + fine_range + 0.5 * fine_step, fine_step, dtype=np.float32)
    stage3 = _scan_joint_angles(img, ref, geo, fine_angles, seen)
    all_candidates.extend({**cand, "stage": 3} for cand in stage3)

    ranked = sorted(top + stage3, key=lambda item: item["score"], reverse=True)
    best = dict(ranked[0] if ranked else joint_angle_translation_score(img, ref, geo, 0.0))
    best["all_candidates"] = sorted(all_candidates, key=lambda item: item["score"], reverse=True)
    best["n_evaluations"] = int(len(all_candidates))
    return best


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



def fine_alignment_search_candidates(
    img: np.ndarray,
    ref: np.ndarray,
    candidate_angles,
    geo: GeometryContext,
    search_range=5,
    step=1,
):
    """Run image-space fine search around candidate orientation branches.

    Fourier-Mellin is used only to propose plausible coarse branches.  This
    function verifies each branch in image space using the existing translation
    FFT score and sorts the candidate results by score descending.
    """
    unique_angles: list[float] = []
    for ang in np.asarray(candidate_angles, dtype=np.float32).ravel():
        norm_ang = normalize_angle(float(ang))
        if all(abs(normalize_angle(norm_ang - prev)) > 1e-4 for prev in unique_angles):
            unique_angles.append(norm_ang)

    if not unique_angles:
        unique_angles = [0.0]

    results = []
    for input_rank, ang in enumerate(unique_angles):
        best = fine_alignment_search(img, ref, ang, geo, search_range=search_range, step=step)
        results.append(
            {
                "angle": float(best["angle"]),
                "dy": float(best["dy"]),
                "dx": float(best["dx"]),
                "score": float(best["score"]),
                "coarse_angle": float(ang),
                "input_rank": int(input_rank),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def _local_candidate_angles(current_angle: float, K: int, search_range: float):
    """Deterministic local alternatives for annealed non-FM iterations."""
    K = max(1, int(K))
    base = normalize_angle(float(current_angle))
    if K == 1:
        return np.array([base], dtype=np.float32)

    offsets = [0.0]
    radius = max(float(search_range), 1.0)
    step = radius / max(K - 1, 1)
    m = 1
    while len(offsets) < K:
        offsets.append(m * step)
        if len(offsets) < K:
            offsets.append(-m * step)
        m += 1
    return np.asarray([normalize_angle(base + off) for off in offsets[:K]], dtype=np.float32)


def align_one_cpu_multicandidate(
    img: np.ndarray,
    geo: GeometryContext,
    ref_match: np.ndarray,
    mask_diameter,
    lp_sigma: float,
    is_global_search: bool,
    search_range: float,
    search_step: float,
    current_bias_y: float,
    current_bias_x: float,
    current_angle: float,
    K: int = 1,
    temperature: float = 0.1,
    use_fm_candidates: bool = False,
    soft: bool = False,
    return_diagnostics: bool = False,
    legacy_single_candidate: bool = True,
    coarse_angle_mode: str = "grid_joint",
):
    """Align one CPU particle with optional annealed top-K orientation branches.

    The returned state remains a single best [angle, dy, dx, score] row for API
    compatibility.  When soft=True, the image returned for reference
    accumulation is a streaming weighted average over candidate aligned images;
    candidate images are generated one at a time and discarded.
    """
    K = max(1, int(K))
    if coarse_angle_mode not in {"fm", "grid_joint"}:
        raise ValueError("coarse_angle_mode must be 'fm' or 'grid_joint'")

    # Preserve the old alignment path as closely as possible for K=1/hard mode.
    if legacy_single_candidate and K == 1 and not soft:
        img_centered = shift_image(img, geo, current_bias_y, current_bias_x)
        img_masked = apply_circular_mask(img_centered, geo, diameter=mask_diameter)
        img_for_matching = apply_lowpass_filter(img_masked, sigma=lp_sigma)

        if is_global_search and coarse_angle_mode == "grid_joint":
            joint_best = coarse_to_fine_joint_search(img_for_matching, ref_match, geo)
            # The joint search returns correction shifts; existing state-update
            # math below expects the raw observed FFT residual, so negate here.
            best = {
                "angle": float(joint_best["angle"]),
                "dy": float(-joint_best["dy"]),
                "dx": float(-joint_best["dx"]),
                "score": float(joint_best["score"]),
                "fft_score": float(joint_best["fft_score"]),
                "coarse_angle_mode": "grid_joint",
                "n_evaluations": int(joint_best.get("n_evaluations", 0)),
            }
        else:
            if is_global_search:
                raw_angle = get_coarse_angle_fourier_mellin(img_for_matching, ref_match, geo)
                center_angle, _ = check_180_ambiguity(img_for_matching, ref_match, raw_angle, geo)
            else:
                center_angle = current_angle

            best = fine_alignment_search(
                img_for_matching, ref_match, center_angle, geo,
                search_range=search_range, step=search_step,
            )
        candidates = [best]
        weights = np.array([1.0], dtype=np.float32)
    else:
        # A) Apply pre-shift using the current translational bias.
        img_centered = shift_image(img, geo, current_bias_y, current_bias_x)

        # B) Mask + lowpass only for matching; final accumulation uses raw image.
        img_masked = apply_circular_mask(img_centered, geo, diameter=mask_diameter)
        img_for_matching = apply_lowpass_filter(img_masked, sigma=lp_sigma)

        # C) Fourier-Mellin proposes orientation branches; local alternatives
        # keep later annealed iterations near the previous best state.
        if is_global_search and coarse_angle_mode == "grid_joint":
            joint_best = coarse_to_fine_joint_search(img_for_matching, ref_match, geo, topk=K)
            candidates = [{
                "angle": float(joint_best["angle"]),
                "dy": float(-joint_best["dy"]),
                "dx": float(-joint_best["dx"]),
                "score": float(joint_best["score"]),
                "fft_score": float(joint_best["fft_score"]),
                "coarse_angle_mode": "grid_joint",
                "n_evaluations": int(joint_best.get("n_evaluations", 0)),
            }]
        elif use_fm_candidates:
            fm_angles = get_topk_coarse_angles_fourier_mellin(img_for_matching, ref_match, geo, K=K)
            # Preserve the basic Fourier-Mellin 180-degree ambiguity handling for
            # every branch.  This is not a hard-coded symmetry model; it simply
            # gives image-space scoring both FM-equivalent orientations to verify.
            candidate_angles = expand_angles_with_180_ambiguity(fm_angles, min_separation_deg=15)
            if (not is_global_search) and current_angle is not None:
                candidate_angles = _separate_normalized_angles(
                    np.concatenate((np.array([current_angle], dtype=np.float32), candidate_angles)),
                    K=None,
                    min_separation_deg=15,
                )
        else:
            candidate_angles = _local_candidate_angles(current_angle, K, search_range)

        # D) Verify each branch by image-space fine alignment and translation score.
        if not (is_global_search and coarse_angle_mode == "grid_joint"):
            candidates = fine_alignment_search_candidates(
                img_for_matching, ref_match, candidate_angles, geo,
                search_range=search_range, step=search_step,
            )
            if not use_fm_candidates:
                candidates = candidates[:K]
        weights = softmax_scores([c["score"] for c in candidates], temperature) if (soft and len(candidates) > 1) else np.eye(1, len(candidates), 0, dtype=np.float32).ravel()

    best = candidates[0]
    best_angle = float(best["angle"])
    best_score = float(best["score"])

    new_params_by_candidate = []
    weighted_aligned_img = np.zeros_like(img, dtype=np.float32)
    for j, cand in enumerate(candidates):
        res_dy, res_dx = float(cand["dy"]), float(cand["dx"])
        cand_angle = float(cand["angle"])

        # E) Convert residual dy/dx to the pre-rotation frame using the existing math.
        rad = np.deg2rad(-cand_angle)
        cos_r, sin_r = np.cos(rad), np.sin(rad)
        res_dx_pre = res_dx * cos_r - res_dy * sin_r
        res_dy_pre = res_dx * sin_r + res_dy * cos_r

        new_bias_y = float(current_bias_y) - res_dy_pre
        new_bias_x = float(current_bias_x) - res_dx_pre
        new_params_by_candidate.append((cand_angle, new_bias_y, new_bias_x, float(cand["score"])))

        # F/G) Streaming candidate image generation and weighted accumulation.
        if soft and len(candidates) > 1:
            final_shifted = shift_image(img, geo, new_bias_y, new_bias_x)
            aligned_img = rotate_image(final_shifted, geo, cand_angle)
            weighted_aligned_img += float(weights[j]) * aligned_img.astype(np.float32, copy=False)
        elif j == 0:
            final_shifted = shift_image(img, geo, new_bias_y, new_bias_x)
            weighted_aligned_img = rotate_image(final_shifted, geo, cand_angle).astype(np.float32, copy=False)
            if not soft:
                break

    best_angle, best_new_bias_y, best_new_bias_x, best_score = new_params_by_candidate[0]
    new_params = np.array([best_angle, best_new_bias_y, best_new_bias_x, best_score], dtype=np.float32)

    if not return_diagnostics:
        return new_params, weighted_aligned_img

    weight_entropy = -float(np.sum(weights * np.log(np.maximum(weights, 1e-12)))) if weights.size else 0.0
    diag = {
        "candidate_angles": np.asarray([c["angle"] for c in candidates], dtype=np.float32),
        "coarse_angles": np.asarray([c.get("coarse_angle", c["angle"]) for c in candidates], dtype=np.float32),
        "candidate_scores": np.asarray([c["score"] for c in candidates], dtype=np.float32),
        "candidate_weights": weights.astype(np.float32, copy=False),
        "max_candidate_weight": np.float32(np.max(weights) if weights.size else 1.0),
        "entropy_candidate_weight": np.float32(weight_entropy),
        "selected_rank": int(candidates[0].get("input_rank", 0)),
        "selected_angle": np.float32(best_angle),
        "coarse_angle_mode": candidates[0].get("coarse_angle_mode", coarse_angle_mode),
        "coarse_angle_evaluations": int(candidates[0].get("n_evaluations", 0)),
    }
    return new_params, weighted_aligned_img, diag
