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

from dataclasses import dataclass, asdict
import time
from typing import Literal

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


# =============================================================================
# Robust MAP-EM single-catalog alignment
# =============================================================================

WeightMode = Literal["none", "sigmoid", "hard_quantile"]


@dataclass
class MAPEMConfig:
    """Configuration for robust MAP-EM CPU alignment.

    Phase meaning:
        phase=1: hard-MAP pose inference + unweighted template update.
        phase=2: robust MAP-EM with inlier weights.
        phase=3: robust MAP-EM with pose-prior MAP inference.

    Current recommended Phase-3 baseline from real-data tests:
        weight_mode="sigmoid", keep_fraction=0.75,
        lambda_shift=0.01, sigma_shift_y=8, sigma_shift_x=8,
        lambda_angle=0.0

    Convention:
        params = [angle, dy, dx, posterior_score]
        aligned = transform_final_image(img, geo, angle, dy, dx)

    use_batched_scan is experimental and defaults to False; it only affects
    local MAP-EM angle scans when explicitly enabled.
    """

    phase: int = 3

    # Joint search parameters.
    global_step: float = 10.0
    mid_range: float = 12.0
    mid_step: float = 2.0
    fine_range: float = 2.0
    fine_step: float = 0.5
    topk: int = 3
    use_batched_scan: bool = False

    # Robust inlier weighting.
    weight_mode: WeightMode = "sigmoid"
    keep_fraction: float = 0.75
    score_threshold: float | None = None
    weight_temperature: float = 0.08
    min_weight: float = 0.0

    # Phase-3 pose priors.
    lambda_shift: float = 0.01
    sigma_shift_y: float = 8.0
    sigma_shift_x: float = 8.0
    lambda_angle: float = 0.0
    sigma_angle: float = 8.0

    # Reference update / diagnostics.
    normalize_reference: bool = False
    mask_soft_edge: int = 5
    diagnostics_n: int = 0

    def normalized(self) -> "MAPEMConfig":
        cfg = MAPEMConfig(**asdict(self))
        cfg.phase = int(cfg.phase)
        if cfg.phase not in {1, 2, 3}:
            raise ValueError("phase must be 1, 2, or 3")

        cfg.topk = max(1, int(cfg.topk))
        cfg.use_batched_scan = bool(cfg.use_batched_scan)
        cfg.keep_fraction = float(np.clip(cfg.keep_fraction, 1e-6, 1.0))
        cfg.weight_temperature = max(float(cfg.weight_temperature), 1e-6)
        cfg.min_weight = float(np.clip(cfg.min_weight, 0.0, 1.0))

        cfg.sigma_shift_y = max(float(cfg.sigma_shift_y), 1e-6)
        cfg.sigma_shift_x = max(float(cfg.sigma_shift_x), 1e-6)
        cfg.sigma_angle = max(float(cfg.sigma_angle), 1e-6)

        if cfg.phase == 1:
            cfg.weight_mode = "none"
            cfg.lambda_shift = 0.0
            cfg.lambda_angle = 0.0
        elif cfg.phase == 2:
            cfg.lambda_shift = 0.0
            cfg.lambda_angle = 0.0

        return cfg


def angle_diff_deg(a, b):
    """Signed circular angle difference a-b in degrees."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def mapem_iter_schedule(it: int, num_iterations: int) -> dict:
    """Annealed pose-search schedule."""
    if it == 0:
        return {"mode": "global", "lp_sigma": 3.0}
    if it == 1:
        return {"mode": "local", "angle_range": 8.0, "angle_step": 2.0, "lp_sigma": 1.0}
    if it == 2:
        return {"mode": "local", "angle_range": 4.0, "angle_step": 1.0, "lp_sigma": 0.0}
    return {"mode": "local", "angle_range": 2.0, "angle_step": 0.5, "lp_sigma": 0.0}


def mapem_warm_start_iter_schedule(it: int, num_iterations: int) -> dict:
    """Conservative local-only schedule for pose warm starts.

    Used when previous alignment parameters are available.
    This skips the global search and performs only small local refinements.
    """
    if it == 0:
        return {"mode": "local", "angle_range": 4.0, "angle_step": 1.0, "lp_sigma": 0.0}
    return {"mode": "local", "angle_range": 2.0, "angle_step": 0.5, "lp_sigma": 0.0}


def initialize_mapem_params(initial_params, n: int) -> tuple[np.ndarray, bool]:
    """Return canonical MAP-EM params and whether they came from a warm start."""
    params = np.zeros((int(n), 4), dtype=np.float32)
    if initial_params is None:
        return params, False

    seed = np.asarray(initial_params, dtype=np.float32)
    if seed.ndim != 2 or seed.shape[0] != int(n) or seed.shape[1] < 3:
        raise ValueError(
            "initial_params must have shape (N, 3) or (N, 4+); "
            f"got {seed.shape} for N={int(n)}"
        )
    if not np.all(np.isfinite(seed[:, :3])):
        raise ValueError("initial_params angle/dy/dx values must all be finite.")

    params[:, :3] = seed[:, :3]
    params[:, 0] = np.asarray([normalize_angle(a) for a in params[:, 0]], dtype=np.float32)
    if seed.shape[1] >= 4:
        params[:, 3] = seed[:, 3]
    return params, True


def pose_prior_scores(candidate: dict, prev_angle: float | None, schedule: dict, cfg: MAPEMConfig) -> tuple[float, float]:
    """Return (translation_prior_score, angle_prior_score)."""
    if cfg.phase < 3:
        return 0.0, 0.0

    dy = float(candidate["dy"])
    dx = float(candidate["dx"])
    translation_prior = -0.5 * ((dy / cfg.sigma_shift_y) ** 2 + (dx / cfg.sigma_shift_x) ** 2)

    if schedule.get("mode") == "global" or prev_angle is None or cfg.lambda_angle == 0.0:
        angle_prior = 0.0
    else:
        dtheta = float(angle_diff_deg(candidate["angle"], prev_angle))
        angle_prior = -0.5 * (dtheta / cfg.sigma_angle) ** 2

    return float(translation_prior), float(angle_prior)


def posteriorize_candidate(candidate: dict, prev_angle: float | None, schedule: dict, cfg: MAPEMConfig) -> dict:
    """Attach posterior-score components to a joint-search candidate."""
    out = dict(candidate)
    image_score = float(candidate["score"])
    t_prior, a_prior = pose_prior_scores(candidate, prev_angle, schedule, cfg)
    posterior = image_score + float(cfg.lambda_shift) * t_prior + float(cfg.lambda_angle) * a_prior

    out["image_score"] = image_score
    out["translation_prior_score"] = float(t_prior)
    out["angle_prior_score"] = float(a_prior)
    out["posterior_score"] = float(posterior)
    out["score"] = float(posterior)
    return out


def scan_joint_angles_mapem(img, ref, geo, angles, prev_angle: float | None, schedule: dict, cfg: MAPEMConfig, mask=None):
    """Evaluate candidate angles and rank by posterior score."""
    if cfg.use_batched_scan and schedule.get("mode") == "local":
        from .batch_cpu import scan_joint_angles_batched_cpu

        base_results = scan_joint_angles_batched_cpu(img, ref, geo, angles, mask=mask)
    else:
        base_results = scan_joint_angles(img, ref, geo, angles, mask=mask)
    results = [posteriorize_candidate(c, prev_angle, schedule, cfg) for c in base_results]
    results.sort(key=lambda d: d["posterior_score"], reverse=True)
    return results


def coarse_to_fine_mapem_search(img, ref, geo, prev_angle: float | None, schedule: dict, cfg: MAPEMConfig, mask=None):
    """Global coarse-to-fine MAP search using image score plus optional pose priors."""
    stage1_angles = np.arange(-180.0, 180.0, cfg.global_step, dtype=np.float32)
    stage1 = scan_joint_angles_mapem(img, ref, geo, stage1_angles, prev_angle, schedule, cfg, mask=mask)
    top1 = stage1[: cfg.topk]

    stage2_angles = []
    for cand in top1:
        center = float(cand["angle"])
        stage2_angles.extend(
            np.arange(
                center - cfg.mid_range,
                center + cfg.mid_range + 0.5 * cfg.mid_step,
                cfg.mid_step,
                dtype=np.float32,
            ).tolist()
        )
    stage2 = scan_joint_angles_mapem(img, ref, geo, stage2_angles, prev_angle, schedule, cfg, mask=mask)

    center = stage2[0]["angle"] if stage2 else (stage1[0]["angle"] if stage1 else 0.0)
    stage3_angles = np.arange(
        center - cfg.fine_range,
        center + cfg.fine_range + 0.5 * cfg.fine_step,
        cfg.fine_step,
        dtype=np.float32,
    )
    stage3 = scan_joint_angles_mapem(img, ref, geo, stage3_angles, prev_angle, schedule, cfg, mask=mask)

    all_candidates = stage1 + stage2 + stage3
    all_candidates.sort(key=lambda d: d["posterior_score"], reverse=True)

    if all_candidates:
        best = dict(all_candidates[0])
    else:
        best = posteriorize_candidate(joint_angle_translation_score(img, ref, geo, 0.0, mask=mask), prev_angle, schedule, cfg)

    best["all_candidates"] = all_candidates
    best["n_evaluations"] = int(len(stage1) + len(stage2) + len(stage3))
    return best


def local_mapem_search(img, ref, geo, center_angle: float, schedule: dict, cfg: MAPEMConfig, mask=None):
    """Local MAP pose search around previous angle."""
    angle_range = float(schedule["angle_range"])
    angle_step = float(schedule["angle_step"])
    angles = np.arange(
        float(center_angle) - angle_range,
        float(center_angle) + angle_range + 0.5 * angle_step,
        angle_step,
        dtype=np.float32,
    )
    results = scan_joint_angles_mapem(img, ref, geo, angles, float(center_angle), schedule, cfg, mask=mask)
    if results:
        best = dict(results[0])
        best["all_candidates"] = results
        best["n_evaluations"] = int(len(results))
        return best

    return {
        "angle": 0.0,
        "dy": 0.0,
        "dx": 0.0,
        "score": 0.0,
        "image_score": 0.0,
        "posterior_score": 0.0,
        "translation_prior_score": 0.0,
        "angle_prior_score": 0.0,
        "fft_score": 0.0,
        "all_candidates": [],
        "n_evaluations": 0,
    }


def estimate_inlier_weights(scores: np.ndarray, cfg: MAPEMConfig) -> tuple[np.ndarray, dict]:
    """Estimate inlier weights from posterior scores."""
    scores = np.asarray(scores, dtype=np.float32)
    n = scores.size

    if cfg.phase == 1 or cfg.weight_mode == "none":
        return np.ones(n, dtype=np.float32), {
            "weight_tau": np.nan,
            "weight_scale": np.nan,
            "effective_n": float(n),
        }

    finite = np.isfinite(scores)
    if not np.any(finite):
        return np.ones(n, dtype=np.float32), {
            "weight_tau": np.nan,
            "weight_scale": np.nan,
            "effective_n": float(n),
        }

    safe = scores.copy()
    safe[~finite] = np.min(scores[finite])

    if cfg.weight_mode == "hard_quantile":
        tau = float(np.quantile(safe, 1.0 - cfg.keep_fraction))
        weights = (safe >= tau).astype(np.float32)
        if float(np.sum(weights)) <= 0.0:
            weights[:] = 1.0
        return weights, {
            "weight_tau": tau,
            "weight_scale": np.nan,
            "effective_n": float(np.sum(weights)),
        }

    if cfg.weight_mode == "sigmoid":
        tau = float(cfg.score_threshold) if cfg.score_threshold is not None else float(np.quantile(safe, 1.0 - cfg.keep_fraction))
        robust_scale = float(np.std(safe[finite])) + 1e-6
        scale = max(float(cfg.weight_temperature), 0.25 * robust_scale, 1e-6)
        logits = np.clip((safe - tau) / scale, -80.0, 80.0)
        weights = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
        if cfg.min_weight > 0.0:
            weights = np.maximum(weights, np.float32(cfg.min_weight))
        return weights, {
            "weight_tau": tau,
            "weight_scale": scale,
            "effective_n": float(np.sum(weights)),
        }

    raise ValueError(f"Unknown weight_mode: {cfg.weight_mode}")


def align_one_mapem_cpu(img, ref_match, geo, prev_angle: float, schedule: dict, cfg: MAPEMConfig, mask=None):
    """Infer MAP pose for one particle."""
    if schedule["mode"] == "global":
        best = coarse_to_fine_mapem_search(img, ref_match, geo, prev_angle=None, schedule=schedule, cfg=cfg, mask=mask)
    else:
        best = local_mapem_search(img, ref_match, geo, center_angle=float(prev_angle), schedule=schedule, cfg=cfg, mask=mask)

    angle = float(best["angle"])
    dy = float(best["dy"])
    dx = float(best["dx"])
    posterior_score = float(best.get("posterior_score", best.get("score", 0.0)))
    aligned_img = transform_final_image(img, geo, angle, dy, dx)
    params = np.array([angle, dy, dx, posterior_score], dtype=np.float32)
    return params, aligned_img, best


def run_alignment_mapem_cpu(
    X,
    initial_ref,
    num_iterations: int = 4,
    mask_diameter=None,
    config: MAPEMConfig | None = None,
    verbose: bool = True,
    initial_params=None,
    search_mode: str | None = None,
):
    """Run robust MAP-EM CPU alignment.

    This is the recommended formal single-process backend.

    Returns:
        final_ref, history_refs, params, meta
    """
    cfg = (config or MAPEMConfig()).normalized()

    X = np.asarray(X, dtype=np.float32)
    initial_ref = np.asarray(initial_ref, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected X shape (N,H,W), got {X.shape}")
    if initial_ref.ndim != 2:
        raise ValueError(f"Expected initial_ref shape (H,W), got {initial_ref.shape}")

    n = int(X.shape[0])
    geo = get_geometry_context(X.shape)
    mask = geo.get_circular_mask(diameter=mask_diameter, soft_edge=cfg.mask_soft_edge)

    params, has_initial_params = initialize_mapem_params(initial_params, n)
    requested_search_mode = "auto" if search_mode is None else str(search_mode).strip().lower()
    if requested_search_mode not in {"auto", "global", "refine"}:
        raise ValueError("search_mode must be 'auto', 'global', or 'refine'.")
    if requested_search_mode == "refine" and not has_initial_params:
        raise ValueError("search_mode='refine' requires initial_params.")
    use_refine_schedule = requested_search_mode == "refine" or (
        requested_search_mode == "auto" and has_initial_params
    )
    resolved_search_mode = "refine" if use_refine_schedule else "global"

    current_ref = apply_circular_mask(initial_ref, geo, diameter=mask_diameter, soft_edge=cfg.mask_soft_edge)
    history_refs = [current_ref.copy()]

    meta = {
        "backend": "single",
        "engine": "align-single-mapem",
        "algorithm": "mapem",
        "config": asdict(cfg),
        "iterations": [],
        "num_particles": n,
        "num_iterations": int(num_iterations),
        "mask_diameter": mask_diameter,
        "has_initial_params": bool(has_initial_params),
        "search_mode": resolved_search_mode,
        "warm_start": bool(use_refine_schedule),
        "implemented": True,
    }

    weights = np.ones(n, dtype=np.float32)
    image_scores = np.zeros(n, dtype=np.float32)
    posterior_scores = np.zeros(n, dtype=np.float32)

    for it in range(int(num_iterations)):
        iter_t0 = time.perf_counter()
        if use_refine_schedule:
            schedule = mapem_warm_start_iter_schedule(it, int(num_iterations))
        else:
            schedule = mapem_iter_schedule(it, int(num_iterations))
        if verbose:
            print(
                f"[MAP-EM iter {it + 1}/{num_iterations}] "
                f"phase={cfg.phase} mode={schedule['mode']} "
                f"lp_sigma={schedule['lp_sigma']} weight={cfg.weight_mode}"
            )

        ref_t0 = time.perf_counter()
        ref_match = apply_lowpass_filter(current_ref, sigma=float(schedule["lp_sigma"]))
        timing_ref_prepare_s = time.perf_counter() - ref_t0

        estep_t0 = time.perf_counter()
        aligned_stack = np.empty_like(X, dtype=np.float32)
        diagnostics = []

        for i in range(n):
            p, aligned, best = align_one_mapem_cpu(
                X[i],
                ref_match,
                geo,
                prev_angle=float(params[i, 0]),
                schedule=schedule,
                cfg=cfg,
                mask=mask,
            )
            params[i] = p
            aligned_stack[i] = aligned
            image_scores[i] = float(best.get("image_score", best.get("score", 0.0)))
            posterior_scores[i] = float(best.get("posterior_score", best.get("score", 0.0)))

            if i < int(cfg.diagnostics_n):
                diagnostics.append({
                    "particle": i,
                    "angle": float(p[0]),
                    "dy": float(p[1]),
                    "dx": float(p[2]),
                    "posterior_score": float(p[3]),
                    "image_score": float(image_scores[i]),
                    "translation_prior_score": float(best.get("translation_prior_score", 0.0)),
                    "angle_prior_score": float(best.get("angle_prior_score", 0.0)),
                    "n_evaluations": int(best.get("n_evaluations", 0)),
                })

        timing_estep_s = time.perf_counter() - estep_t0

        weight_t0 = time.perf_counter()
        weights, weight_info = estimate_inlier_weights(posterior_scores, cfg)
        timing_weight_s = time.perf_counter() - weight_t0

        ref_update_t0 = time.perf_counter()
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-8:
            weights = np.ones(n, dtype=np.float32)
            weight_sum = float(n)

        new_ref = np.sum(aligned_stack * weights[:, None, None], axis=0) / weight_sum
        new_ref = apply_circular_mask(new_ref, geo, diameter=mask_diameter, soft_edge=cfg.mask_soft_edge)

        if cfg.normalize_reference:
            new_ref = (new_ref - np.mean(new_ref)) / (np.std(new_ref) + 1e-8)

        current_ref = new_ref.astype(np.float32, copy=False)
        history_refs.append(current_ref.copy())
        timing_reference_update_s = time.perf_counter() - ref_update_t0
        timing_total_iter_s = time.perf_counter() - iter_t0

        meta["iterations"].append({
            "iteration": it,
            "schedule": dict(schedule),
            "image_score_mean": float(np.mean(image_scores)),
            "posterior_score_mean": float(np.mean(posterior_scores)),
            "weight_mean": float(np.mean(weights)),
            "weight_min": float(np.min(weights)),
            "weight_max": float(np.max(weights)),
            **weight_info,
            "timing_ref_prepare_s": float(timing_ref_prepare_s),
            "timing_estep_s": float(timing_estep_s),
            "timing_weight_s": float(timing_weight_s),
            "timing_reference_update_s": float(timing_reference_update_s),
            "timing_total_iter_s": float(timing_total_iter_s),
            "diagnostics": diagnostics,
        })

        if verbose:
            print(
                "  timing: "
                f"ref_prepare={timing_ref_prepare_s:.3f}s "
                f"estep={timing_estep_s:.3f}s "
                f"weight={timing_weight_s:.3f}s "
                f"reference_update={timing_reference_update_s:.3f}s "
                f"total={timing_total_iter_s:.3f}s"
            )

    meta["last_weights"] = weights.astype(np.float32)
    meta["last_image_scores"] = image_scores.astype(np.float32)
    meta["last_posterior_scores"] = posterior_scores.astype(np.float32)
    return current_ref, history_refs, params, meta


def run_transform_mapem_cpu(X, params):
    """Apply MAP-EM transform parameters to a stack."""
    return run_transform_single_cpu(X, params)
