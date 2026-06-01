#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spyder-friendly real-data demo for the public AlignImg API.

Run this file directly in Spyder. It loads one MRC/MRCS stack, calls
alignimg_api.run_alignment() / alignimg_api.run_transform(), compares
Phase 1 / Phase 2 / Phase 3 MAP-EM behavior, saves CSV/PNG/NPY outputs,
and shows figures interactively.

This v3 script does NOT import align_single_v2.py.  It tests the formal
package path:

    alignimg_api.py  ->  align_utils.py

Expected API support:
    api.run_alignment(..., algorithm="mapem", config=api.make_mapem_config(...))
    api.run_transform(...)

Edit the CONFIG block before running.
"""

from __future__ import annotations

import os
import csv
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mrcfile

import align_utils as au
import alignimg_api as api


# =============================================================================
# CONFIG: edit these values in Spyder before running
# =============================================================================

DATA_PATH = "./test_align.mrcs"
GT_PATH = "./mu_aligned_mean.mrc"     # optional; set to None or "" if unavailable
OUTPUT_DIR = "realdata_v3_spyder_report"

# Data subset. Set N_PARTICLES = 0 to load all particles.
N_PARTICLES = 500
START_INDEX = 0

# Alignment settings
NUM_ITERATIONS = 4
MASK_DIAMETER = 80.0
BACKEND = "single"

# Optional GT display/evaluation correction
APPLY_GT_ROTATION = True
GT_ROTATION_DEG = 180.0

# Phase 2 / Phase 3 robust weighting
KEEP_FRACTION = 0.75
WEIGHT_TEMPERATURE = 0.08
SCORE_THRESHOLD = None       # None -> quantile threshold from KEEP_FRACTION
MIN_WEIGHT = 0.0

# Phase 3 pose priors: current recommended default is translation prior only.
LAMBDA_SHIFT = 0.01
SIGMA_SHIFT_Y = 8.0
SIGMA_SHIFT_X = 8.0
LAMBDA_ANGLE = 0.0
SIGMA_ANGLE = 8.0

# Pose-search settings
GLOBAL_STEP = 10.0
MID_RANGE = 12.0
MID_STEP = 2.0
FINE_RANGE = 2.0
FINE_STEP = 0.5
TOPK = 3

# Reference update / diagnostics
NORMALIZE_REFERENCE = False
DIAGNOSTICS_N = 0
MASK_SOFT_EDGE = 5

# Output behavior
SAVE_OUTPUTS = True
SHOW_FIGURES = True


# =============================================================================
# I/O helpers
# =============================================================================

def load_mrc_stack(path: str, n: Optional[int] = None, start: int = 0) -> np.ndarray:
    """Load a stack from MRC/MRCS with mmap, then copy selected images to float32."""
    with mrcfile.mmap(path, permissive=True, mode="r") as mrc:
        arr = np.asarray(mrc.data)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(f"Expected 2D image or 3D stack, got shape {arr.shape} from {path}")
        end = arr.shape[0] if n is None or int(n) <= 0 else min(arr.shape[0], int(start) + int(n))
        out = np.asarray(arr[int(start):end], dtype=np.float32).copy()
    return out


def load_mrc_image(path: Optional[str]) -> Optional[np.ndarray]:
    """Load optional 2D reference/GT image."""
    if path is None or str(path).strip() == "" or not os.path.exists(path):
        return None
    with mrcfile.open(path, permissive=True, mode="r") as mrc:
        img = np.asarray(mrc.data, dtype=np.float32).squeeze().copy()
    if img.ndim != 2:
        raise ValueError(f"Expected 2D reference image, got shape {img.shape} from {path}")
    return img


def ensure_float32_stack(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        X = X[None, ...]
    if X.ndim != 3:
        raise ValueError(f"Expected stack shape (N,H,W), got {X.shape}")
    return X


def normalize_image_for_display(img, low=0.1, high=99.9, pad_fraction=0.10):
    """Return robust display range with optional padding."""
    arr = np.asarray(img, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(arr, [low, high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        vmin = mean - 3.0 * std
        vmax = mean + 3.0 * std
    pad = pad_fraction * (vmax - vmin)
    return float(vmin - pad), float(vmax + pad)


def global_display_range(images, low=0.05, high=99.95, pad_fraction=0.20):
    """Compute one shared display range for all panels."""
    vals = []
    for img in images:
        if img is None:
            continue
        arr = np.asarray(img, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            vals.append(arr.ravel())
    if not vals:
        return 0.0, 1.0
    all_vals = np.concatenate(vals)
    vmin, vmax = np.percentile(all_vals, [low, high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        mean = float(np.mean(all_vals))
        std = float(np.std(all_vals))
        vmin = mean - 3.0 * std
        vmax = mean + 3.0 * std
    pad = pad_fraction * (vmax - vmin)
    return float(vmin - pad), float(vmax + pad)


def normalized_corr(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if mask is not None:
        keep = mask > 0.05
        a = a[keep]
        b = b[keep]
    a = a - np.mean(a)
    b = b - np.mean(b)
    return float(np.sum(a * b) / (np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-8))


def normalized_rmse(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if mask is not None:
        keep = mask > 0.05
        a = a[keep]
        b = b[keep]
    a = (a - np.mean(a)) / (np.std(a) + 1e-8)
    b = (b - np.mean(b)) / (np.std(b) + 1e-8)
    return float(np.sqrt(np.mean((a - b) ** 2)))


# =============================================================================
# API config presets
# =============================================================================

def make_api_mapem_config(phase: int, weight_mode: str, *, lambda_shift: float = 0.0, lambda_angle: float = 0.0):
    """Create MAP-EM config through the public API when available."""
    common = dict(
        phase=phase,
        weight_mode=weight_mode,
        global_step=GLOBAL_STEP,
        mid_range=MID_RANGE,
        mid_step=MID_STEP,
        fine_range=FINE_RANGE,
        fine_step=FINE_STEP,
        topk=TOPK,
        keep_fraction=KEEP_FRACTION,
        weight_temperature=WEIGHT_TEMPERATURE,
        score_threshold=SCORE_THRESHOLD,
        min_weight=MIN_WEIGHT,
        lambda_shift=lambda_shift,
        sigma_shift_y=SIGMA_SHIFT_Y,
        sigma_shift_x=SIGMA_SHIFT_X,
        lambda_angle=lambda_angle,
        sigma_angle=SIGMA_ANGLE,
        normalize_reference=NORMALIZE_REFERENCE,
        mask_soft_edge=MASK_SOFT_EDGE,
        diagnostics_n=DIAGNOSTICS_N,
    )

    if hasattr(api, "make_mapem_config"):
        return api.make_mapem_config(**common)

    # Fallback for direct align_utils use if an older API lacks make_mapem_config.
    if hasattr(au, "MAPEMConfig"):
        return au.MAPEMConfig(**common)

    raise RuntimeError(
        "This script requires the updated API with make_mapem_config(), "
        "or align_utils.MAPEMConfig."
    )


def config_to_dict(cfg) -> dict:
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    return {"config_repr": repr(cfg)}


def make_phase_configs() -> Dict[str, dict]:
    """Return phase configurations to run through alignimg_api."""
    return {
        "phase1_hard_map": {
            "algorithm": "mapem",
            "config": make_api_mapem_config(phase=1, weight_mode="none"),
        },
        "phase2_robust_sigmoid": {
            "algorithm": "mapem",
            "config": make_api_mapem_config(phase=2, weight_mode="sigmoid"),
        },
        "phase3_translation_prior": {
            "algorithm": "mapem",
            "config": make_api_mapem_config(
                phase=3,
                weight_mode="sigmoid",
                lambda_shift=LAMBDA_SHIFT,
                lambda_angle=LAMBDA_ANGLE,
            ),
        },
    }


# =============================================================================
# Evaluation / outputs
# =============================================================================

def estimate_gauge_if_reference_exists(
    final_ref: np.ndarray,
    corrected_mean: np.ndarray,
    gt_img: Optional[np.ndarray],
    geo: au.GeometryContext,
    mask: np.ndarray,
) -> dict:
    """Estimate global transform against optional reference image."""
    if gt_img is None:
        return {
            "has_gt": False,
            "global_angle": np.nan,
            "global_dy": np.nan,
            "global_dx": np.nan,
            "global_score": np.nan,
            "final_ref_gauge_corr": np.nan,
            "corrected_mean_gauge_corr": np.nan,
            "final_ref_gauge_nrmse": np.nan,
            "corrected_mean_gauge_nrmse": np.nan,
            "final_ref_global_corrected": None,
            "corrected_mean_global_corrected": None,
        }

    if gt_img.shape != final_ref.shape:
        raise ValueError(f"GT shape {gt_img.shape} does not match image shape {final_ref.shape}")

    best = au.coarse_to_fine_joint_search(final_ref, gt_img, geo, mask=mask)
    global_angle = float(best["angle"])
    global_dy = float(best["dy"])
    global_dx = float(best["dx"])
    global_score = float(best["score"])

    final_ref_gc = au.transform_final_image(final_ref, geo, global_angle, global_dy, global_dx)
    corrected_mean_gc = au.transform_final_image(corrected_mean, geo, global_angle, global_dy, global_dx)

    return {
        "has_gt": True,
        "global_angle": global_angle,
        "global_dy": global_dy,
        "global_dx": global_dx,
        "global_score": global_score,
        "final_ref_gauge_corr": normalized_corr(final_ref_gc, gt_img, mask),
        "corrected_mean_gauge_corr": normalized_corr(corrected_mean_gc, gt_img, mask),
        "final_ref_gauge_nrmse": normalized_rmse(final_ref_gc, gt_img, mask),
        "corrected_mean_gauge_nrmse": normalized_rmse(corrected_mean_gc, gt_img, mask),
        "final_ref_global_corrected": final_ref_gc,
        "corrected_mean_global_corrected": corrected_mean_gc,
    }


def summarize_run(
    name: str,
    cfg,
    algorithm: str,
    elapsed: float,
    final_ref: np.ndarray,
    corrected: np.ndarray,
    params: np.ndarray,
    meta: dict,
    gt_img: Optional[np.ndarray],
    geo: au.GeometryContext,
    mask: np.ndarray,
) -> Tuple[dict, dict]:
    corrected_mean = np.mean(corrected, axis=0)
    final_iter = meta.get("iterations", [])[-1] if meta.get("iterations") else {}
    gauge = estimate_gauge_if_reference_exists(final_ref, corrected_mean, gt_img, geo, mask)

    weights = np.asarray(meta.get("last_weights", np.ones(params.shape[0])), dtype=np.float32)
    image_scores = np.asarray(meta.get("last_image_scores", params[:, 3]), dtype=np.float32)
    posterior_scores = np.asarray(meta.get("last_posterior_scores", params[:, 3]), dtype=np.float32)

    cfg_dict = config_to_dict(cfg)
    shift_mag = np.sqrt(params[:, 1].astype(np.float32) ** 2 + params[:, 2].astype(np.float32) ** 2)
    large_shift = shift_mag > 20.0

    row = {
        "method": name,
        "backend": BACKEND,
        "algorithm": algorithm,
        "phase": cfg_dict.get("phase", np.nan),
        "weight_mode": cfg_dict.get("weight_mode", ""),
        "elapsed_s": elapsed,
        "num_particles": int(params.shape[0]),
        "effective_n": float(final_iter.get("effective_n", np.sum(weights))),
        "weight_mean": float(np.mean(weights)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "weight_median": float(np.median(weights)),
        "image_score_mean": float(np.mean(image_scores)),
        "posterior_score_mean": float(np.mean(posterior_scores)),
        "shift_mean_px": float(np.mean(shift_mag)),
        "shift_median_px": float(np.median(shift_mag)),
        "large_shift_gt20_count": int(np.sum(large_shift)),
        "large_shift_gt20_weight_mean": float(np.mean(weights[large_shift])) if np.any(large_shift) else np.nan,
        "final_ref_mean": float(np.mean(final_ref)),
        "final_ref_std": float(np.std(final_ref)),
        "corrected_mean_mean": float(np.mean(corrected_mean)),
        "corrected_mean_std": float(np.std(corrected_mean)),
        "global_angle": gauge["global_angle"],
        "global_dy": gauge["global_dy"],
        "global_dx": gauge["global_dx"],
        "global_score": gauge["global_score"],
        "final_ref_gauge_corr": gauge["final_ref_gauge_corr"],
        "corrected_mean_gauge_corr": gauge["corrected_mean_gauge_corr"],
        "final_ref_gauge_nrmse": gauge["final_ref_gauge_nrmse"],
        "corrected_mean_gauge_nrmse": gauge["corrected_mean_gauge_nrmse"],
        "lambda_shift": cfg_dict.get("lambda_shift", np.nan),
        "lambda_angle": cfg_dict.get("lambda_angle", np.nan),
        "sigma_shift_y": cfg_dict.get("sigma_shift_y", np.nan),
        "sigma_shift_x": cfg_dict.get("sigma_shift_x", np.nan),
        "sigma_angle": cfg_dict.get("sigma_angle", np.nan),
        "keep_fraction": cfg_dict.get("keep_fraction", np.nan),
        "weight_temperature": cfg_dict.get("weight_temperature", np.nan),
    }
    return row, gauge


def save_params_csv(path: Path, params: np.ndarray, weights: np.ndarray, image_scores: np.ndarray, posterior_scores: np.ndarray) -> None:
    shift_mag = np.sqrt(params[:, 1].astype(np.float32) ** 2 + params[:, 2].astype(np.float32) ** 2)
    df = pd.DataFrame({
        "particle": np.arange(params.shape[0]),
        "angle": params[:, 0],
        "dy": params[:, 1],
        "dx": params[:, 2],
        "shift_mag": shift_mag,
        "posterior_score_param": params[:, 3],
        "weight": weights,
        "image_score": image_scores,
        "posterior_score": posterior_scores,
    })
    df.to_csv(path, index=False, float_format="%.6f")


def save_iteration_csv(path: Path, meta: dict) -> None:
    rows = []
    for it in meta.get("iterations", []):
        row = {k: v for k, v in it.items() if k not in {"diagnostics", "schedule"}}
        sched = it.get("schedule", {})
        for k, v in sched.items():
            row[f"schedule_{k}"] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_summary_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# =============================================================================
# Plotting: save and show
# =============================================================================

def plot_phase_overview(raw_avg, init_ref, gt_img, results, save_path: Optional[Path] = None):
    has_gt = gt_img is not None
    rows = len(results)
    cols = 6 if has_gt else 4
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)

    display_images = [raw_avg, init_ref]
    if has_gt:
        display_images.append(gt_img)
    for res in results:
        display_images.append(res["final_ref"])
        display_images.append(res["corrected_mean"])
        if res.get("gauge", {}).get("corrected_mean_global_corrected") is not None:
            display_images.append(res["gauge"]["corrected_mean_global_corrected"])
    vmin, vmax = global_display_range(display_images, low=0.05, high=99.95, pad_fraction=0.20)

    for r, res in enumerate(results):
        phase_name = res["name"]
        imgs = [
            ("Raw avg", raw_avg),
            ("Initial ref", init_ref),
        ]
        if has_gt:
            imgs.append(("GT/ref", gt_img))
        imgs.extend([
            (f"{phase_name}\nfinal_ref", res["final_ref"]),
            (f"{phase_name}\ncorrected mean", res["corrected_mean"]),
        ])
        if has_gt:
            imgs.append((f"{phase_name}\ngauge-corr mean", res["gauge"]["corrected_mean_global_corrected"]))

        for c, (title, img) in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis("off")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160)
    return fig


def plot_weight_score_summary(results, save_path: Optional[Path] = None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for res in results:
        name = res["name"]
        axes[0].hist(res["weights"], bins=30, alpha=0.45, label=name)
        axes[1].hist(res["image_scores"], bins=30, alpha=0.45, label=name)
        axes[2].hist(res["posterior_scores"], bins=30, alpha=0.45, label=name)

    axes[0].set_title("Inlier weights")
    axes[1].set_title("Image scores")
    axes[2].set_title("Posterior scores")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160)
    return fig


def plot_iteration_curves(results, save_path: Optional[Path] = None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for res in results:
        name = res["name"]
        iters = res["meta"].get("iterations", [])
        x = np.arange(len(iters)) + 1
        eff = [it.get("effective_n", np.nan) for it in iters]
        wmean = [it.get("weight_mean", np.nan) for it in iters]
        score = [it.get("posterior_score_mean", np.nan) for it in iters]
        axes[0].plot(x, eff, marker="o", label=name)
        axes[1].plot(x, wmean, marker="o", label=name)
        axes[2].plot(x, score, marker="o", label=name)
    axes[0].set_title("Effective N")
    axes[1].set_title("Mean weight")
    axes[2].set_title("Mean posterior score")
    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160)
    return fig


def plot_shift_weight_summary(results, save_path: Optional[Path] = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for res in results:
        name = res["name"]
        params = res["params"]
        shift_mag = np.sqrt(params[:, 1].astype(np.float32) ** 2 + params[:, 2].astype(np.float32) ** 2)
        axes[0].hist(shift_mag, bins=40, alpha=0.45, label=name)
        axes[1].scatter(shift_mag, res["weights"], s=8, alpha=0.35, label=name)
    axes[0].set_title("Shift magnitude")
    axes[0].set_xlabel("sqrt(dy^2 + dx^2) [px]")
    axes[1].set_title("Weight vs shift magnitude")
    axes[1].set_xlabel("sqrt(dy^2 + dx^2) [px]")
    axes[1].set_ylabel("inlier weight")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160)
    return fig


# =============================================================================
# Main run: Spyder executes this directly
# =============================================================================

def run_three_phase_demo():
    outdir = Path(OUTPUT_DIR)
    if SAVE_OUTPUTS:
        outdir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Input stack not found: {DATA_PATH}\n"
            "Please edit DATA_PATH in the CONFIG block."
        )

    print("Using alignimg_api backend dispatcher:")
    if hasattr(api, "available_backends"):
        print(api.available_backends())

    print(f"Loading real data: {DATA_PATH}")
    n_load = None if N_PARTICLES is None or int(N_PARTICLES) <= 0 else int(N_PARTICLES)
    X = ensure_float32_stack(load_mrc_stack(DATA_PATH, n=n_load, start=START_INDEX))
    print(f"Loaded stack: {X.shape}, dtype={X.dtype}")

    gt_img = load_mrc_image(GT_PATH)
    if gt_img is not None:
        print(f"Loaded optional GT/ref: {GT_PATH}, shape={gt_img.shape}")
        if APPLY_GT_ROTATION and abs(float(GT_ROTATION_DEG)) > 1e-8:
            geo_gt = au.get_geometry_context(gt_img.shape)
            gt_img = au.rotate_image(gt_img, geo_gt, float(GT_ROTATION_DEG))
            print(f"Applied GT rotation: {GT_ROTATION_DEG} deg")
    else:
        print("No optional GT/ref found; gauge-corrected metrics will be skipped.")

    geo = au.get_geometry_context(X.shape)
    mask = geo.get_circular_mask(diameter=MASK_DIAMETER)
    raw_avg = np.mean(X, axis=0).astype(np.float32)
    init_ref = raw_avg.copy()

    configs = make_phase_configs()
    results = []
    summary_rows = []

    for name, spec in configs.items():
        cfg = spec["config"]
        algorithm = spec.get("algorithm", "mapem")
        print(f"\n=== Running {name} through alignimg_api ===")
        print(f"algorithm={algorithm}, backend={BACKEND}")
        print(f"config: {config_to_dict(cfg)}")
        t0 = time.perf_counter()
        final_ref, history, params, meta = api.run_alignment(
            X,
            init_ref,
            num_iterations=NUM_ITERATIONS,
            mask_diameter=MASK_DIAMETER,
            backend=BACKEND,
            algorithm=algorithm,
            config=cfg,
            verbose=True,
        )
        corrected = api.run_transform(
            X,
            params,
            backend=BACKEND,
            algorithm=algorithm,
        )
        elapsed = time.perf_counter() - t0
        corrected_mean = np.mean(corrected, axis=0).astype(np.float32)

        summary, gauge = summarize_run(
            name=name,
            cfg=cfg,
            algorithm=algorithm,
            elapsed=elapsed,
            final_ref=final_ref,
            corrected=corrected,
            params=params,
            meta=meta,
            gt_img=gt_img,
            geo=geo,
            mask=mask,
        )
        summary_rows.append(summary)

        weights = np.asarray(meta.get("last_weights", np.ones(params.shape[0])), dtype=np.float32)
        image_scores = np.asarray(meta.get("last_image_scores", params[:, 3]), dtype=np.float32)
        posterior_scores = np.asarray(meta.get("last_posterior_scores", params[:, 3]), dtype=np.float32)

        if SAVE_OUTPUTS:
            save_params_csv(outdir / f"{name}_params_weights.csv", params, weights, image_scores, posterior_scores)
            save_iteration_csv(outdir / f"{name}_iteration_summary.csv", meta)
            np.save(outdir / f"{name}_final_ref.npy", final_ref)
            np.save(outdir / f"{name}_corrected_mean.npy", corrected_mean)

        results.append({
            "name": name,
            "cfg": cfg,
            "algorithm": algorithm,
            "final_ref": final_ref,
            "corrected_mean": corrected_mean,
            "params": params,
            "meta": meta,
            "weights": weights,
            "image_scores": image_scores,
            "posterior_scores": posterior_scores,
            "gauge": gauge,
        })

        print(
            f"{name}: elapsed={elapsed:.2f}s, "
            f"effective_n={summary['effective_n']:.2f}, "
            f"weight_mean={summary['weight_mean']:.3f}, "
            f"shift_mean={summary['shift_mean_px']:.2f}px, "
            f"large_shift_gt20={summary['large_shift_gt20_count']}, "
            f"gauge_corr={summary['corrected_mean_gauge_corr']}"
        )

    summary_df = pd.DataFrame(summary_rows)
    print("\n=== Phase summary ===")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(summary_df)

    if SAVE_OUTPUTS:
        write_summary_csv(outdir / "phase_summary.csv", summary_rows)

    fig1 = plot_phase_overview(
        raw_avg,
        init_ref,
        gt_img,
        results,
        save_path=(outdir / "phase_overview.png") if SAVE_OUTPUTS else None,
    )
    fig2 = plot_weight_score_summary(
        results,
        save_path=(outdir / "weights_scores.png") if SAVE_OUTPUTS else None,
    )
    fig3 = plot_iteration_curves(
        results,
        save_path=(outdir / "iteration_curves.png") if SAVE_OUTPUTS else None,
    )
    fig4 = plot_shift_weight_summary(
        results,
        save_path=(outdir / "shift_weight_summary.png") if SAVE_OUTPUTS else None,
    )

    if SHOW_FIGURES:
        plt.show(block=False)

    if SAVE_OUTPUTS:
        print(f"\nWrote report to: {outdir.resolve()}")
        print("Key files:")
        print(f"  - {outdir / 'phase_summary.csv'}")
        print(f"  - {outdir / 'phase_overview.png'}")
        print(f"  - {outdir / 'weights_scores.png'}")
        print(f"  - {outdir / 'iteration_curves.png'}")
        print(f"  - {outdir / 'shift_weight_summary.png'}")

    return {
        "X": X,
        "gt_img": gt_img,
        "raw_avg": raw_avg,
        "init_ref": init_ref,
        "results": results,
        "summary": summary_df,
        "figures": (fig1, fig2, fig3, fig4),
    }


# Running the file in Spyder will execute everything below.
RUN_OUTPUT = run_three_phase_demo()
