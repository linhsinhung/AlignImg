#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Algorithm Prototype: Robust MAP-EM single-catalog 2D alignment.

This file intentionally keeps a complete self-contained implementation of
Phase 1 / Phase 2 / Phase 3 for readability and algorithm development.

Production implementation:
    alignimg.api

This file is a prototype and is not imported by the public package.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import time
from typing import Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alignimg import _utils as au


WeightMode = Literal["none", "sigmoid", "hard_quantile"]


@dataclass
class MAPEMConfig:
    """Configuration for the robust MAP-EM prototype.

    phase:
        1 = hard MAP pose inference, unweighted update.
        2 = robust update using inlier weights, no pose prior.
        3 = robust update + pose priors during MAP inference.

    weight_mode:
        "none"          -> w_i = 1.
        "sigmoid"       -> w_i = sigmoid((score_i - tau) / T).
        "hard_quantile" -> keep top keep_fraction by score.

    For phase 3, translation prior score is:
        -0.5 * ((dy/sigma_shift_y)^2 + (dx/sigma_shift_x)^2)

    For local iterations, angle prior score is:
        -0.5 * (wrap(angle - prev_angle) / sigma_angle)^2
    """

    phase: int = 3

    # Search schedule defaults, intentionally close to align_single.py.
    global_step: float = 10.0
    mid_range: float = 12.0
    mid_step: float = 2.0
    fine_range: float = 2.0
    fine_step: float = 0.5
    topk: int = 3

    # Phase 2 robust update.
    weight_mode: WeightMode = "sigmoid"
    keep_fraction: float = 0.8
    score_threshold: float | None = None
    weight_temperature: float = 0.08
    min_weight: float = 0.0

    # Phase 3 pose priors.
    lambda_shift: float = 0.0
    sigma_shift_y: float = 8.0
    sigma_shift_x: float = 8.0
    lambda_angle: float = 0.0
    sigma_angle: float = 6.0

    # Reference update.
    normalize_reference: bool = False
    mask_soft_edge: int = 5

    # Diagnostics.
    diagnostics_n: int = 0

    def normalized(self) -> "MAPEMConfig":
        cfg = MAPEMConfig(**asdict(self))
        cfg.phase = int(cfg.phase)
        if cfg.phase not in {1, 2, 3}:
            raise ValueError("phase must be 1, 2, or 3")
        cfg.keep_fraction = float(np.clip(cfg.keep_fraction, 1e-6, 1.0))
        cfg.topk = max(1, int(cfg.topk))
        cfg.sigma_shift_y = max(float(cfg.sigma_shift_y), 1e-6)
        cfg.sigma_shift_x = max(float(cfg.sigma_shift_x), 1e-6)
        cfg.sigma_angle = max(float(cfg.sigma_angle), 1e-6)
        cfg.weight_temperature = max(float(cfg.weight_temperature), 1e-6)
        cfg.min_weight = float(np.clip(cfg.min_weight, 0.0, 1.0))
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
    """Annealed pose-search schedule.

    This represents decreasing pose uncertainty across iterations:
        iter 0: global pose search
        iter 1: local ±8°
        iter 2: local ±4°
        later : local ±2°
    """
    if it == 0:
        return {"mode": "global", "lp_sigma": 3.0}
    if it == 1:
        return {"mode": "local", "angle_range": 8.0, "angle_step": 2.0, "lp_sigma": 1.0}
    if it == 2:
        return {"mode": "local", "angle_range": 4.0, "angle_step": 1.0, "lp_sigma": 0.0}
    return {"mode": "local", "angle_range": 2.0, "angle_step": 0.5, "lp_sigma": 0.0}


def pose_prior_scores(candidate: dict, prev_angle: float | None, schedule: dict, cfg: MAPEMConfig) -> tuple[float, float]:
    """Return (translation_prior, angle_prior) log scores for one candidate."""
    if cfg.phase < 3:
        return 0.0, 0.0

    dy = float(candidate["dy"])
    dx = float(candidate["dx"])
    translation_prior = -0.5 * ((dy / cfg.sigma_shift_y) ** 2 + (dx / cfg.sigma_shift_x) ** 2)

    if schedule.get("mode") == "global" or prev_angle is None:
        angle_prior = 0.0
    else:
        dtheta = float(angle_diff_deg(candidate["angle"], prev_angle))
        angle_prior = -0.5 * (dtheta / cfg.sigma_angle) ** 2

    return translation_prior, angle_prior


def posteriorize_candidate(candidate: dict, prev_angle: float | None, schedule: dict, cfg: MAPEMConfig) -> dict:
    """Attach MAP posterior score components to a joint-search candidate."""
    out = dict(candidate)
    image_score = float(candidate["score"])
    t_prior, a_prior = pose_prior_scores(candidate, prev_angle, schedule, cfg)
    posterior = image_score + float(cfg.lambda_shift) * t_prior + float(cfg.lambda_angle) * a_prior

    out["image_score"] = image_score
    out["translation_prior_score"] = float(t_prior)
    out["angle_prior_score"] = float(a_prior)
    out["posterior_score"] = float(posterior)
    # Keep score as posterior score so downstream ranking uses MAP objective.
    out["score"] = float(posterior)
    return out


def scan_joint_angles_mapem(img, ref, geo, angles, prev_angle: float | None, schedule: dict, cfg: MAPEMConfig, mask=None):
    """Evaluate candidate angles and rank by posterior score."""
    base_results = au.scan_joint_angles(img, ref, geo, angles, mask=mask)
    results = [posteriorize_candidate(c, prev_angle, schedule, cfg) for c in base_results]
    results.sort(key=lambda d: d["posterior_score"], reverse=True)
    return results


def coarse_to_fine_mapem_search(img, ref, geo, prev_angle: float | None, schedule: dict, cfg: MAPEMConfig, mask=None):
    """Global coarse-to-fine MAP search using image score plus optional pose priors."""
    if schedule.get("mode") != "global":
        raise ValueError("coarse_to_fine_mapem_search is intended for global iterations")

    stage1_angles = np.arange(-180.0, 180.0, cfg.global_step, dtype=np.float32)
    stage1 = scan_joint_angles_mapem(img, ref, geo, stage1_angles, prev_angle, schedule, cfg, mask=mask)
    top1 = stage1[: cfg.topk]

    stage2_angles = []
    for cand in top1:
        center = float(cand["angle"])
        stage2_angles.extend(
            np.arange(center - cfg.mid_range, center + cfg.mid_range + 0.5 * cfg.mid_step, cfg.mid_step, dtype=np.float32).tolist()
        )
    stage2 = scan_joint_angles_mapem(img, ref, geo, stage2_angles, prev_angle, schedule, cfg, mask=mask)

    center = stage2[0]["angle"] if stage2 else (stage1[0]["angle"] if stage1 else 0.0)
    stage3_angles = np.arange(center - cfg.fine_range, center + cfg.fine_range + 0.5 * cfg.fine_step, cfg.fine_step, dtype=np.float32)
    stage3 = scan_joint_angles_mapem(img, ref, geo, stage3_angles, prev_angle, schedule, cfg, mask=mask)

    all_candidates = stage1 + stage2 + stage3
    all_candidates.sort(key=lambda d: d["posterior_score"], reverse=True)
    best = dict(all_candidates[0] if all_candidates else au.joint_angle_translation_score(img, ref, geo, 0.0, mask=mask))
    best["all_candidates"] = all_candidates
    best["n_evaluations"] = int(len(stage1) + len(stage2) + len(stage3))
    return best


def local_mapem_search(img, ref, geo, center_angle: float, schedule: dict, cfg: MAPEMConfig, mask=None):
    """Local MAP pose search around the previous angle."""
    angle_range = float(schedule["angle_range"])
    angle_step = float(schedule["angle_step"])
    angles = np.arange(center_angle - angle_range, center_angle + angle_range + 0.5 * angle_step, angle_step, dtype=np.float32)
    results = scan_joint_angles_mapem(img, ref, geo, angles, center_angle, schedule, cfg, mask=mask)
    if results:
        best = dict(results[0])
        best["all_candidates"] = results
        best["n_evaluations"] = int(len(results))
        return best
    return {"angle": 0.0, "dy": 0.0, "dx": 0.0, "score": 0.0, "image_score": 0.0, "posterior_score": 0.0, "fft_score": 0.0}


def estimate_inlier_weights(scores: np.ndarray, cfg: MAPEMConfig) -> tuple[np.ndarray, dict]:
    """Estimate inlier weights w_i from best posterior scores."""
    scores = np.asarray(scores, dtype=np.float32)
    n = scores.size
    if cfg.phase == 1 or cfg.weight_mode == "none":
        return np.ones(n, dtype=np.float32), {"weight_tau": np.nan, "weight_scale": np.nan, "effective_n": float(n)}

    finite = np.isfinite(scores)
    if not np.any(finite):
        return np.ones(n, dtype=np.float32), {"weight_tau": np.nan, "weight_scale": np.nan, "effective_n": float(n)}

    safe = scores.copy()
    safe[~finite] = np.min(scores[finite])

    if cfg.weight_mode == "hard_quantile":
        tau = float(np.quantile(safe, 1.0 - cfg.keep_fraction))
        weights = (safe >= tau).astype(np.float32)
        if np.sum(weights) <= 0:
            weights[:] = 1.0
        return weights, {"weight_tau": tau, "weight_scale": np.nan, "effective_n": float(np.sum(weights))}

    if cfg.weight_mode == "sigmoid":
        tau = float(cfg.score_threshold) if cfg.score_threshold is not None else float(np.quantile(safe, 1.0 - cfg.keep_fraction))
        # Use a robust scale unless user-specified temperature dominates.
        robust_scale = float(np.std(safe[finite])) + 1e-6
        scale = max(float(cfg.weight_temperature), 0.25 * robust_scale, 1e-6)
        logits = np.clip((safe - tau) / scale, -80.0, 80.0)
        weights = 1.0 / (1.0 + np.exp(-logits))
        weights = weights.astype(np.float32)
        if cfg.min_weight > 0:
            weights = np.maximum(weights, np.float32(cfg.min_weight))
        return weights, {"weight_tau": tau, "weight_scale": scale, "effective_n": float(np.sum(weights))}

    raise ValueError(f"Unknown weight_mode: {cfg.weight_mode}")


def align_one_mapem(img, ref_match, geo, prev_angle: float, schedule: dict, cfg: MAPEMConfig, mask=None):
    """Infer MAP pose for one particle under the configured phase."""
    if schedule["mode"] == "global":
        best = coarse_to_fine_mapem_search(img, ref_match, geo, prev_angle=None, schedule=schedule, cfg=cfg, mask=mask)
    else:
        best = local_mapem_search(img, ref_match, geo, center_angle=float(prev_angle), schedule=schedule, cfg=cfg, mask=mask)

    angle = float(best["angle"])
    dy = float(best["dy"])
    dx = float(best["dx"])
    posterior_score = float(best.get("posterior_score", best["score"]))
    aligned_img = au.transform_final_image(img, geo, angle, dy, dx)
    params = np.array([angle, dy, dx, posterior_score], dtype=np.float32)
    return params, aligned_img, best


def run_alignment_mapem(
    X,
    initial_ref,
    num_iterations: int = 4,
    mask_diameter=None,
    config: MAPEMConfig | None = None,
    verbose: bool = True,
):
    """Run robust MAP-EM alignment prototype.

    Returns:
        final_ref, history_refs, params, meta

    meta contains per-iteration weights and score diagnostics for algorithm study.
    """
    cfg = (config or MAPEMConfig()).normalized()
    X = np.asarray(X, dtype=np.float32)
    initial_ref = np.asarray(initial_ref, dtype=np.float32)
    n = int(X.shape[0])
    geo = au.get_geometry_context(X.shape)
    mask = geo.get_circular_mask(diameter=mask_diameter, soft_edge=cfg.mask_soft_edge)

    params = np.zeros((n, 4), dtype=np.float32)
    current_ref = au.apply_circular_mask(initial_ref, geo, diameter=mask_diameter, soft_edge=cfg.mask_soft_edge)
    history_refs = [current_ref.copy()]

    meta = {
        "backend": "single-v2-mapem",
        "engine": "alignimg-prototype-mapem-single",
        "config": asdict(cfg),
        "iterations": [],
        "num_particles": n,
        "num_iterations": int(num_iterations),
        "mask_diameter": mask_diameter,
    }

    for it in range(int(num_iterations)):
        schedule = mapem_iter_schedule(it, int(num_iterations))
        if verbose:
            print(f"[v2 iter {it + 1}/{num_iterations}] phase={cfg.phase} mode={schedule['mode']} lp_sigma={schedule['lp_sigma']} weight={cfg.weight_mode}")

        ref_match = au.apply_lowpass_filter(current_ref, sigma=float(schedule["lp_sigma"]))
        aligned_stack = np.empty_like(X, dtype=np.float32)
        best_records = []
        image_scores = np.zeros(n, dtype=np.float32)
        posterior_scores = np.zeros(n, dtype=np.float32)

        for i in range(n):
            p, aligned, best = align_one_mapem(
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
                best_records.append({
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

        weights, weight_info = estimate_inlier_weights(posterior_scores, cfg)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-8:
            weights = np.ones(n, dtype=np.float32)
            weight_sum = float(n)

        new_ref = np.sum(aligned_stack * weights[:, None, None], axis=0) / weight_sum
        new_ref = au.apply_circular_mask(new_ref, geo, diameter=mask_diameter, soft_edge=cfg.mask_soft_edge)
        if cfg.normalize_reference:
            new_ref = (new_ref - np.mean(new_ref)) / (np.std(new_ref) + 1e-8)
        current_ref = new_ref.astype(np.float32, copy=False)
        history_refs.append(current_ref.copy())

        meta["iterations"].append({
            "iteration": it,
            "schedule": dict(schedule),
            "image_score_mean": float(np.mean(image_scores)),
            "posterior_score_mean": float(np.mean(posterior_scores)),
            "weight_mean": float(np.mean(weights)),
            "weight_min": float(np.min(weights)),
            "weight_max": float(np.max(weights)),
            **weight_info,
            "diagnostics": best_records,
        })

    meta["last_weights"] = weights.astype(np.float32)
    meta["last_image_scores"] = image_scores.astype(np.float32)
    meta["last_posterior_scores"] = posterior_scores.astype(np.float32)
    return current_ref, history_refs, params, meta


def run_transform_mapem(X, params):
    """Apply final MAP-EM transform parameters to a stack."""
    X = np.asarray(X, dtype=np.float32)
    params = np.asarray(params, dtype=np.float32)
    geo = au.get_geometry_context(X.shape)
    out = np.empty_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        out[i] = au.transform_final_image(X[i], geo, float(params[i, 0]), float(params[i, 1]), float(params[i, 2]))
    return out


def _demo_synthetic_stack(n=48, size=96, seed=0, outlier_fraction=0.2):
    rng = np.random.default_rng(seed)
    geo = au.get_geometry_context((size, size))
    y, x = np.indices((size, size), dtype=np.float32)
    cy = (size - 1) / 2.0
    cx = (size - 1) / 2.0

    base = np.zeros((size, size), dtype=np.float32)
    for _ in range(7):
        ay = rng.uniform(-18, 18)
        ax = rng.uniform(-18, 18)
        s = rng.uniform(2.5, 6.0)
        amp = rng.uniform(0.6, 1.2)
        base += amp * np.exp(-(((y - (cy + ay)) ** 2 + (x - (cx + ax)) ** 2) / (2 * s * s)))
    base = au.apply_circular_mask(base, geo, diameter=size * 0.82)
    base = (base - np.mean(base)) / (np.std(base) + 1e-8)

    X = np.zeros((n, size, size), dtype=np.float32)
    is_outlier = rng.random(n) < float(outlier_fraction)
    for i in range(n):
        if is_outlier[i]:
            noise = rng.normal(0.0, 1.0, size=(size, size)).astype(np.float32)
            X[i] = au.apply_lowpass_filter(noise, sigma=1.5)
            continue
        angle = rng.uniform(-160.0, 160.0)
        dy = rng.uniform(-5.0, 5.0)
        dx = rng.uniform(-5.0, 5.0)
        img = au.transform_final_image(base, geo, angle, dy, dx)
        img += rng.normal(0.0, 0.35, size=(size, size)).astype(np.float32)
        X[i] = img

    initial_ref = np.mean(X, axis=0).astype(np.float32)
    return base, X, initial_ref, is_outlier


if __name__ == "__main__":
    gt, X, initial_ref, is_outlier = _demo_synthetic_stack()
    cfg = MAPEMConfig(phase=1, weight_mode="sigmoid", keep_fraction=0.75, diagnostics_n=3)
    t0 = time.perf_counter()
    final_ref, history, params, meta = run_alignment_mapem(X, initial_ref, num_iterations=4, mask_diameter=96 * 0.82, config=cfg, verbose=True)
    elapsed = time.perf_counter() - t0
    corrected = run_transform_mapem(X, params)
    print(f"final_ref mean/std: {float(np.mean(final_ref)):.6f}/{float(np.std(final_ref)):.6f}")
    print(f"params shape: {params.shape}; corrected shape: {corrected.shape}")
    print(f"outliers in demo: {int(np.sum(is_outlier))}/{len(is_outlier)}")
    print(f"last effective_n: {meta['iterations'][-1]['effective_n']:.3f}")
    print(f"elapsed: {elapsed:.3f}s")
