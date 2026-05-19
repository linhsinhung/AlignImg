#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Accuracy benchmark for AlignImg alignment engines.

This script creates a deterministic synthetic stack with known ground-truth
rotations/translations, runs each available alignment engine, compares the
recovered alignment to the ground truth, and writes tabular + visual reports.

Example:
    python test_accuracy.py --n 80 --size 96 --iterations 4 --output-dir accuracy_report
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import align_utils as au
import alignimg_api as api


@dataclass(frozen=True)
class MethodConfig:
    name: str
    use_gpu: bool
    n_jobs: int | None
    use_multicandidate: bool = False


@dataclass
class MethodResult:
    name: str
    engine: str
    elapsed_s: float
    params: np.ndarray
    final_ref: np.ndarray
    corrected_stack: np.ndarray
    particle_metrics: dict[str, np.ndarray]
    summary: dict[str, float | str]
    final_ref_global_corrected: np.ndarray | None = None
    gauge_corrected_stack: np.ndarray | None = None


def angle_diff_deg(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """Return signed minimum angular difference a-b in degrees."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def rotation_matrix_2d(angle_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rad = np.deg2rad(angle_deg)
    return np.cos(rad), np.sin(rad)


def make_asymmetric_template(size: int, seed: int) -> np.ndarray:
    """Create an asymmetric, cryo-EM-like synthetic 2D reference."""
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

    # Add a weak deterministic texture so angle recovery is less ambiguous.
    texture = rng.normal(0.0, 0.04, size=(size, size)).astype(np.float32)
    template += au.apply_lowpass_filter(texture, sigma=2.0)
    template = au.apply_circular_mask(template, geo, diameter=size * 0.82)
    template = (template - np.mean(template)) / (np.std(template) + 1e-8)
    return template.astype(np.float32)


def generate_ground_truth_stack(
    n: int,
    size: int,
    noise: float,
    max_shift: float,
    max_angle: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ground_truth_template, noisy_stack, gt_table).

    gt_table columns:
        applied_angle, applied_dy, applied_dx,
        ideal_correction_angle, ideal_correction_dy, ideal_correction_dx

    The input image is generated as shift(rotate(gt, applied_angle), dy, dx).
    The ideal CPU correction for transform_final_image(), which applies
    rotate(image, angle) then shift(image, dy, dx), is:
        angle = -applied_angle
        [dx, dy] = -R(-applied_angle) @ [applied_dx, applied_dy]
    """
    rng = np.random.default_rng(seed)
    geo = au.get_geometry_context((size, size))
    gt = make_asymmetric_template(size, seed=seed + 101)

    applied_angles = rng.uniform(-max_angle, max_angle, size=n).astype(np.float32)
    applied_dys = rng.uniform(-max_shift, max_shift, size=n).astype(np.float32)
    applied_dxs = rng.uniform(-max_shift, max_shift, size=n).astype(np.float32)

    stack = np.empty((n, size, size), dtype=np.float32)
    for i, (angle, dy, dx) in enumerate(zip(applied_angles, applied_dys, applied_dxs)):
        img = au.rotate_image(gt, geo, float(angle))
        img = au.shift_image(img, geo, float(dy), float(dx))
        img += rng.normal(0.0, noise, size=(size, size)).astype(np.float32)
        stack[i] = img

    ideal_angles = angle_diff_deg(-applied_angles, 0.0).astype(np.float32)
    cos_r, sin_r = rotation_matrix_2d(-applied_angles)
    # R(-angle) @ [dx, dy] in x/y coordinates, then negate.
    ideal_dxs = -(applied_dxs * cos_r - applied_dys * sin_r)
    ideal_dys = -(applied_dxs * sin_r + applied_dys * cos_r)

    gt_table = np.stack(
        [applied_angles, applied_dys, applied_dxs, ideal_angles, ideal_dys, ideal_dxs],
        axis=1,
    ).astype(np.float32)
    return gt, stack, gt_table


def normalized_rmse(img: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> float:
    a = np.asarray(img, dtype=np.float32)
    b = np.asarray(ref, dtype=np.float32)
    if mask is not None:
        keep = mask > 0.05
        a = a[keep]
        b = b[keep]
    a = (a - np.mean(a)) / (np.std(a) + 1e-8)
    b = (b - np.mean(b)) / (np.std(b) + 1e-8)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def normalized_corr(img: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> float:
    a = np.asarray(img, dtype=np.float32)
    b = np.asarray(ref, dtype=np.float32)
    if mask is not None:
        keep = mask > 0.05
        a = a[keep]
        b = b[keep]
    a = a - np.mean(a)
    b = b - np.mean(b)
    return float(np.sum(a * b) / (np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-8))




def estimate_global_reference_transform(final_ref: np.ndarray, gt_img: np.ndarray, geo: au.GeometryContext) -> tuple[float, float, float, float]:
    """Estimate global gauge transform that maps final_ref into gt_img frame."""
    best = au.coarse_to_fine_joint_search(final_ref, gt_img, geo)
    return float(best["angle"]), float(best["dy"]), float(best["dx"]), float(best["score"])


def apply_global_gauge_to_corrected_stack(
    corrected_stack: np.ndarray,
    geo: au.GeometryContext,
    global_angle: float,
    global_dy: float,
    global_dx: float,
) -> np.ndarray:
    """Apply a common global gauge transform to all corrected particles."""
    out = np.empty_like(corrected_stack, dtype=np.float32)
    for i, img in enumerate(corrected_stack):
        out[i] = au.transform_final_image(img, geo, global_angle, global_dy, global_dx)
    return out

def evaluate_method(
    name: str,
    final_ref: np.ndarray,
    params: np.ndarray,
    corrected: np.ndarray,
    gt_img: np.ndarray,
    gt_table: np.ndarray,
    mask: np.ndarray,
    elapsed_s: float,
    engine: str,
    geo: au.GeometryContext,
) -> MethodResult:
    ideal_angle = gt_table[:, 3]
    ideal_dy = gt_table[:, 4]
    ideal_dx = gt_table[:, 5]

    angle_err = angle_diff_deg(params[:, 0], ideal_angle).astype(np.float32)
    dy_err = (params[:, 1] - ideal_dy).astype(np.float32)
    dx_err = (params[:, 2] - ideal_dx).astype(np.float32)
    shift_err = np.sqrt(dy_err**2 + dx_err**2).astype(np.float32)

    particle_nrmse = np.array([normalized_rmse(img, gt_img, mask) for img in corrected], dtype=np.float32)
    particle_corr = np.array([normalized_corr(img, gt_img, mask) for img in corrected], dtype=np.float32)
    mean_corrected = np.mean(corrected, axis=0)

    global_angle, global_dy, global_dx, global_score = estimate_global_reference_transform(final_ref, gt_img, geo)
    final_ref_global_corrected = au.transform_final_image(final_ref, geo, global_angle, global_dy, global_dx)
    gauge_corrected_stack = apply_global_gauge_to_corrected_stack(corrected, geo, global_angle, global_dy, global_dx)
    gauge_corrected_mean = np.mean(gauge_corrected_stack, axis=0)

    raw_angle_error = angle_diff_deg(params[:, 0], ideal_angle).astype(np.float32)
    global_angle_offset = float(np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(raw_angle_error))))))
    angle_error_gauge_corrected = angle_diff_deg(params[:, 0] - global_angle_offset, ideal_angle).astype(np.float32)

    params_angle = params[:, 0]
    params_dy = params[:, 1]
    params_dx = params[:, 2]
    combined_angle = angle_diff_deg(params_angle + global_angle, 0.0).astype(np.float32)
    theta = np.deg2rad(global_angle)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    combined_dx = (cos_t * params_dx + sin_t * params_dy + global_dx).astype(np.float32)
    combined_dy = (-sin_t * params_dx + cos_t * params_dy + global_dy).astype(np.float32)

    dy_error_gauge_corrected = (combined_dy - ideal_dy).astype(np.float32)
    dx_error_gauge_corrected = (combined_dx - ideal_dx).astype(np.float32)
    shift_error_gauge_corrected = np.sqrt(dy_error_gauge_corrected**2 + dx_error_gauge_corrected**2).astype(np.float32)

    gauge_particle_nrmse = np.array([normalized_rmse(img, gt_img, mask) for img in gauge_corrected_stack], dtype=np.float32)
    gauge_particle_corr = np.array([normalized_corr(img, gt_img, mask) for img in gauge_corrected_stack], dtype=np.float32)

    metrics = {
        "angle_error_deg": angle_err,
        "dy_error_px": dy_err,
        "dx_error_px": dx_err,
        "shift_error_px": shift_err,
        "particle_nrmse": particle_nrmse,
        "particle_corr": particle_corr,
        "score": params[:, 3].astype(np.float32),
        "raw_angle_error_deg": raw_angle_error,
        "angle_error_gauge_corrected_deg": angle_error_gauge_corrected,
        "combined_angle_deg": combined_angle,
        "combined_dy_px": combined_dy,
        "combined_dx_px": combined_dx,
        "dy_error_gauge_corrected_px": dy_error_gauge_corrected,
        "dx_error_gauge_corrected_px": dx_error_gauge_corrected,
        "shift_error_gauge_corrected_px": shift_error_gauge_corrected,
    }
    summary: dict[str, float | str] = {
        "method": name,
        "engine": engine,
        "elapsed_s": elapsed_s,
        "angle_mae_deg": float(np.mean(np.abs(angle_err))),
        "angle_rmse_deg": float(np.sqrt(np.mean(angle_err**2))),
        "dy_mae_px": float(np.mean(np.abs(dy_err))),
        "dx_mae_px": float(np.mean(np.abs(dx_err))),
        "shift_mae_px": float(np.mean(shift_err)),
        "shift_rmse_px": float(np.sqrt(np.mean(shift_err**2))),
        "particle_nrmse_mean": float(np.mean(particle_nrmse)),
        "particle_corr_mean": float(np.mean(particle_corr)),
        "final_ref_nrmse": normalized_rmse(final_ref, gt_img, mask),
        "corrected_mean_nrmse": normalized_rmse(mean_corrected, gt_img, mask),
        "corrected_mean_corr": normalized_corr(mean_corrected, gt_img, mask),
        "global_gauge_angle_deg": global_angle,
        "global_gauge_dy_px": global_dy,
        "global_gauge_dx_px": global_dx,
        "global_gauge_score": global_score,
        "gauge_corrected_mean_nrmse": normalized_rmse(gauge_corrected_mean, gt_img, mask),
        "gauge_corrected_mean_corr": normalized_corr(gauge_corrected_mean, gt_img, mask),
        "gauge_corrected_particle_nrmse_mean": float(np.mean(gauge_particle_nrmse)),
        "gauge_corrected_particle_corr_mean": float(np.mean(gauge_particle_corr)),
        "angle_mae_gauge_corrected_deg": float(np.mean(np.abs(angle_error_gauge_corrected))),
        "angle_rmse_gauge_corrected_deg": float(np.sqrt(np.mean(angle_error_gauge_corrected**2))),
        "dy_mae_gauge_corrected_px": float(np.mean(np.abs(dy_error_gauge_corrected))),
        "dx_mae_gauge_corrected_px": float(np.mean(np.abs(dx_error_gauge_corrected))),
        "shift_mae_gauge_corrected_px": float(np.mean(shift_error_gauge_corrected)),
        "shift_rmse_gauge_corrected_px": float(np.sqrt(np.mean(shift_error_gauge_corrected**2))),
    }
    return MethodResult(
        name,
        engine,
        elapsed_s,
        params,
        final_ref,
        corrected,
        metrics,
        summary,
        final_ref_global_corrected=final_ref_global_corrected,
        gauge_corrected_stack=gauge_corrected_stack,
    )


def write_ground_truth_csv(path: Path, gt_table: np.ndarray) -> None:
    headers = [
        "particle", "applied_angle_deg", "applied_dy_px", "applied_dx_px",
        "ideal_correction_angle_deg", "ideal_correction_dy_px", "ideal_correction_dx_px",
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i, row in enumerate(gt_table):
            writer.writerow([i, *[f"{float(v):.6f}" for v in row]])


def write_summary_csv(path: Path, results: list[MethodResult]) -> None:
    if not results:
        return
    keys = list(results[0].summary.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for result in results:
            writer.writerow(result.summary)


def write_particle_csv(path: Path, result: MethodResult, gt_table: np.ndarray) -> None:
    headers = [
        "particle", "applied_angle_deg", "applied_dy_px", "applied_dx_px",
        "ideal_angle_deg", "ideal_dy_px", "ideal_dx_px",
        "estimated_angle_deg", "estimated_dy_px", "estimated_dx_px", "score",
        "angle_error_deg", "dy_error_px", "dx_error_px", "shift_error_px",
        "angle_error_gauge_corrected_deg", "dy_error_gauge_corrected_px", "dx_error_gauge_corrected_px", "shift_error_gauge_corrected_px",
        "particle_nrmse", "particle_corr",
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(result.params.shape[0]):
            row = [
                i,
                gt_table[i, 0], gt_table[i, 1], gt_table[i, 2],
                gt_table[i, 3], gt_table[i, 4], gt_table[i, 5],
                result.params[i, 0], result.params[i, 1], result.params[i, 2], result.params[i, 3],
                result.particle_metrics["angle_error_deg"][i],
                result.particle_metrics["dy_error_px"][i],
                result.particle_metrics["dx_error_px"][i],
                result.particle_metrics["shift_error_px"][i],
                result.particle_metrics["angle_error_gauge_corrected_deg"][i],
                result.particle_metrics["dy_error_gauge_corrected_px"][i],
                result.particle_metrics["dx_error_gauge_corrected_px"][i],
                result.particle_metrics["shift_error_gauge_corrected_px"][i],
                result.particle_metrics["particle_nrmse"][i],
                result.particle_metrics["particle_corr"][i],
            ]
            writer.writerow([row[0], *[f"{float(v):.6f}" for v in row[1:]]])


def plot_overview(output_path: Path, gt_img: np.ndarray, raw_stack: np.ndarray, results: list[MethodResult]) -> None:
    rows = max(1, len(results))
    cols = 6
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)

    raw_avg = np.mean(raw_stack, axis=0)
    vmin, vmax = np.percentile(gt_img, [1, 99])
    for r, result in enumerate(results):
        final_ref_global_corrected = result.final_ref_global_corrected if result.final_ref_global_corrected is not None else result.final_ref
        gauge_stack = result.gauge_corrected_stack if result.gauge_corrected_stack is not None else result.corrected_stack
        gauge_corrected_mean = np.mean(gauge_stack, axis=0)
        row_imgs = [
            ("GT", gt_img),
            ("Raw average", raw_avg),
            (f"{result.name}\nfinal_ref", result.final_ref),
            (f"{result.name}\nfinal_ref global", final_ref_global_corrected),
            (f"{result.name}\ncorrected avg", np.mean(result.corrected_stack, axis=0)),
            (f"{result.name}\ngauge-corr avg", gauge_corrected_mean),
        ]
        for c, (title, img) in enumerate(row_imgs):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_angle_error_histograms(output_path: Path, results: list[MethodResult]) -> None:
    if not results:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for result in results:
        axes[0].hist(result.particle_metrics["raw_angle_error_deg"], bins=48, alpha=0.45, label=result.name)
        axes[1].hist(result.particle_metrics["angle_error_gauge_corrected_deg"], bins=48, alpha=0.45, label=result.name)
    axes[0].set_title("Raw angle error histogram")
    axes[1].set_title("Gauge-corrected angle error histogram")
    for ax in axes:
        ax.set_xlabel("Signed angle error (deg)")
        ax.set_ylabel("Particles")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_error_distributions(output_path: Path, results: list[MethodResult]) -> None:
    metric_specs = [
        ("angle_error_deg", "Angle error (deg)", True),
        ("shift_error_px", "Shift error (px)", False),
        ("particle_nrmse", "Particle NRMSE", False),
        ("particle_corr", "Particle correlation", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for ax, (metric, label, absolute) in zip(axes, metric_specs):
        data = []
        labels = []
        for result in results:
            values = result.particle_metrics[metric]
            if absolute:
                values = np.abs(values)
            data.append(values)
            labels.append(result.name)
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_parameter_scatter(output_path: Path, gt_table: np.ndarray, results: list[MethodResult]) -> None:
    rows = len(results)
    if rows == 0:
        return
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows), squeeze=False)
    ideal = [gt_table[:, 3], gt_table[:, 4], gt_table[:, 5]]
    names = ["Angle (deg)", "Dy (px)", "Dx (px)"]

    for r, result in enumerate(results):
        estimates = [result.params[:, 0], result.params[:, 1], result.params[:, 2]]
        for c, (ideal_values, estimated_values, label) in enumerate(zip(ideal, estimates, names)):
            ax = axes[r, c]
            ax.scatter(ideal_values, estimated_values, s=12, alpha=0.7)
            lo = float(min(np.min(ideal_values), np.min(estimated_values)))
            hi = float(max(np.max(ideal_values), np.max(estimated_values)))
            pad = max((hi - lo) * 0.05, 1e-3)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
            ax.set_xlabel(f"Ideal {label}")
            ax.set_ylabel(f"Estimated {label}")
            ax.set_title(f"{result.name}: estimated vs ideal {label}")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)





def plot_parameter_scatter_gauge_corrected(output_path: Path, gt_table: np.ndarray, results: list[MethodResult]) -> None:
    rows = len(results)
    if rows == 0:
        return
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows), squeeze=False)
    ideal = [gt_table[:, 3], gt_table[:, 4], gt_table[:, 5]]
    names = ["Angle (deg)", "Dy (px)", "Dx (px)"]

    for r, result in enumerate(results):
        estimates = [
            result.particle_metrics["combined_angle_deg"],
            result.particle_metrics["combined_dy_px"],
            result.particle_metrics["combined_dx_px"],
        ]
        for c, (ideal_values, estimated_values, label) in enumerate(zip(ideal, estimates, names)):
            ax = axes[r, c]
            ax.scatter(ideal_values, estimated_values, s=12, alpha=0.7)
            lo = float(min(np.min(ideal_values), np.min(estimated_values)))
            hi = float(max(np.max(ideal_values), np.max(estimated_values)))
            pad = max((hi - lo) * 0.05, 1e-3)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
            ax.set_xlabel(f"Ideal {label}")
            ax.set_ylabel(f"Gauge-corrected estimated {label}")
            ax.set_title(f"{result.name}: gauge-corrected estimated vs ideal {label}")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_candidate_diagnostics_csv(path: Path, diagnostics: list[dict], gt_table: np.ndarray) -> None:
    """Save candidate branch diagnostics for a small particle subset."""
    if not diagnostics:
        return
    headers = [
        "iteration", "particle", "candidate_rank", "candidate_angle_deg", "coarse_angle_deg",
        "candidate_score", "candidate_weight", "max_candidate_weight", "entropy_candidate_weight",
        "selected_angle_deg", "ideal_angle_deg", "selected_rank",
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for diag in diagnostics:
            particle = int(diag["particle"])
            ideal = float(gt_table[particle, 3])
            angles = np.asarray(diag.get("candidate_angles", []), dtype=np.float32)
            coarse = np.asarray(diag.get("coarse_angles", angles), dtype=np.float32)
            scores = np.asarray(diag.get("candidate_scores", []), dtype=np.float32)
            weights = np.asarray(diag.get("candidate_weights", []), dtype=np.float32)
            for rank, angle in enumerate(angles):
                writer.writerow([
                    int(diag.get("iteration", -1)), particle, rank,
                    f"{float(angle):.6f}",
                    f"{float(coarse[rank]) if rank < coarse.size else float(angle):.6f}",
                    f"{float(scores[rank]) if rank < scores.size else np.nan:.6f}",
                    f"{float(weights[rank]) if rank < weights.size else np.nan:.6f}",
                    f"{float(diag.get('max_candidate_weight', np.nan)):.6f}",
                    f"{float(diag.get('entropy_candidate_weight', np.nan)):.6f}",
                    f"{float(diag.get('selected_angle', np.nan)):.6f}",
                    f"{ideal:.6f}",
                    int(diag.get("selected_rank", 0)),
                ])


def plot_angle_error_comparison(output_path: Path, results: list[MethodResult]) -> None:
    if len(results) < 1:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for result in results:
        ax.hist(result.particle_metrics["angle_error_deg"], bins=48, alpha=0.45, label=result.name)
    ax.set_xlabel("Signed angle error (deg)")
    ax.set_ylabel("Particles")
    ax.set_title("Angle error branch structure")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_selected_rank_histogram(output_path: Path, diagnostics_by_method: dict[str, list[dict]]) -> None:
    if not diagnostics_by_method:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = []
    data = []
    for name, diagnostics in diagnostics_by_method.items():
        ranks = [int(d.get("selected_rank", 0)) for d in diagnostics]
        if ranks:
            labels.append(name)
            data.append(ranks)
    if not data:
        plt.close(fig)
        return
    bins = np.arange(0, max(max(r) for r in data) + 2) - 0.5
    for ranks, label in zip(data, labels):
        ax.hist(ranks, bins=bins, alpha=0.5, label=label)
    ax.set_xlabel("Selected candidate rank after image-space scoring")
    ax.set_ylabel("Particle/iteration diagnostics")
    ax.set_title("Selected candidate rank histogram")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_legacy_k1_parity_check(
    stack: np.ndarray,
    init_ref: np.ndarray,
    num_iterations: int,
    mask_diameter: float,
    output_dir: Path,
    tolerance: float,
    verbose: bool,
) -> None:
    """Confirm legacy mode and multicandidate K=1/hard mode are identical."""
    legacy_ref, _legacy_hist, legacy_params, _legacy_meta = api.run_alignment(
        stack,
        init_ref,
        num_iterations=num_iterations,
        mask_diameter=mask_diameter,
        use_gpu=False,
        n_jobs=1,
        verbose=verbose,
        use_multicandidate=False,
    )
    k1_ref, _k1_hist, k1_params, _k1_meta = api.run_alignment(
        stack,
        init_ref,
        num_iterations=num_iterations,
        mask_diameter=mask_diameter,
        use_gpu=False,
        n_jobs=1,
        verbose=verbose,
        use_multicandidate=True,
        topk_initial=1,
        anneal_multicandidate=True,
    )

    params_max_abs = float(np.max(np.abs(legacy_params - k1_params)))
    ref_max_abs = float(np.max(np.abs(legacy_ref - k1_ref)))
    passed = params_max_abs <= tolerance and ref_max_abs <= tolerance

    path = output_dir / "legacy_k1_parity_check.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["params_max_abs", "ref_max_abs", "tolerance", "passed"])
        writer.writerow([f"{params_max_abs:.9g}", f"{ref_max_abs:.9g}", f"{tolerance:.9g}", int(passed)])

    status = "PASS" if passed else "FAIL"
    print(
        f"Legacy K=1 parity check: {status} "
        f"(params_max_abs={params_max_abs:.3g}, ref_max_abs={ref_max_abs:.3g}, tol={tolerance:.3g})"
    )


def available_methods(requested: str) -> list[MethodConfig]:
    method_map = {
        "cpu-serial": MethodConfig("cpu-serial", use_gpu=False, n_jobs=1),
        "cpu-parallel": MethodConfig("cpu-parallel", use_gpu=False, n_jobs=-1),
        "gpu": MethodConfig("gpu", use_gpu=True, n_jobs=None),
    }
    if requested == "all":
        names = ["cpu-serial", "cpu-parallel"]
        if api.HAS_GPU:
            names.append("gpu")
        return [method_map[name] for name in names]
    return [method_map[name.strip()] for name in requested.split(",") if name.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark AlignImg alignment accuracy on synthetic ground truth data.")
    parser.add_argument("--n", type=int, default=80, help="Number of synthetic particles.")
    parser.add_argument("--size", type=int, default=96, help="Image height/width in pixels.")
    parser.add_argument("--noise", type=float, default=0.35, help="Gaussian noise standard deviation.")
    parser.add_argument("--max-shift", type=float, default=4.5, help="Maximum absolute applied shift in pixels.")
    parser.add_argument("--max-angle", type=float, default=170.0, help="Maximum absolute applied rotation in degrees.")
    parser.add_argument("--iterations", type=int, default=4, help="Alignment iterations passed to run_alignment().")
    parser.add_argument("--mask-diameter", type=float, default=None, help="Circular mask diameter; default is 82%% of image size.")
    parser.add_argument("--methods", default="all", help="Comma-separated methods: cpu-serial,cpu-parallel,gpu or all.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--output-dir", default="accuracy_report", help="Directory for CSV and PNG outputs.")
    parser.add_argument("--gpu-batch-size", type=int, default=256, help="Batch size for GPU method.")
    parser.add_argument("--verbose", action="store_true", help="Show run_alignment progress output.")
    parser.add_argument("--use-multicandidate", action="store_true", help="Enable annealed top-K orientation candidates.")
    parser.add_argument("--topk-initial", type=int, default=5, help="Initial top-K orientation candidates.")
    parser.add_argument("--oracle-ref", action="store_true", help="Use the ground-truth template as the initial reference.")
    parser.add_argument("--translation-only", action="store_true", help="Set max_angle=0 and evaluate shift recovery separately.")
    parser.add_argument("--rotation-only", action="store_true", help="Set max_shift=0 and evaluate rotation recovery separately.")
    parser.add_argument("--diagnostics-n", type=int, default=20, help="Number of particles for candidate diagnostics.")
    parser.add_argument("--check-legacy-k1", action="store_true", help="Run a serial parity check for legacy vs multicandidate K=1 hard mode.")
    parser.add_argument("--legacy-k1-tol", type=float, default=1e-6, help="Tolerance for --check-legacy-k1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.translation_only and args.rotation_only:
        raise ValueError("--translation-only and --rotation-only are mutually exclusive")
    if args.translation_only:
        args.max_angle = 0.0
    if args.rotation_only:
        args.max_shift = 0.0

    mask_diameter = args.mask_diameter if args.mask_diameter is not None else args.size * 0.82
    geo = au.get_geometry_context((args.size, args.size))
    mask = geo.get_circular_mask(diameter=mask_diameter)

    gt_img, stack, gt_table = generate_ground_truth_stack(
        n=args.n,
        size=args.size,
        noise=args.noise,
        max_shift=args.max_shift,
        max_angle=args.max_angle,
        seed=args.seed,
    )
    init_ref = gt_img.astype(np.float32, copy=True) if args.oracle_ref else np.mean(stack, axis=0).astype(np.float32)
    if args.oracle_ref:
        print("Using oracle ground-truth initial reference.")
    if args.translation_only:
        print("Translation-only mode: max_angle forced to 0.")
    elif args.rotation_only:
        print("Rotation-only mode: max_shift forced to 0.")
    else:
        print("End-to-end raw-average mode: GT pose comparison is strict and may include global reference-frame ambiguity.")
    write_ground_truth_csv(output_dir / "ground_truth_params.csv", gt_table)

    if args.check_legacy_k1:
        run_legacy_k1_parity_check(
            stack=stack,
            init_ref=init_ref,
            num_iterations=args.iterations,
            mask_diameter=mask_diameter,
            output_dir=output_dir,
            tolerance=args.legacy_k1_tol,
            verbose=args.verbose,
        )

    methods = available_methods(args.methods)
    if args.use_multicandidate:
        expanded_methods: list[MethodConfig] = []
        for method in methods:
            if method.use_gpu:
                expanded_methods.append(method)
            else:
                expanded_methods.append(MethodConfig(f"{method.name}-single", method.use_gpu, method.n_jobs, False))
                expanded_methods.append(MethodConfig(f"{method.name}-multicandidate", method.use_gpu, method.n_jobs, True))
        methods = expanded_methods

    results: list[MethodResult] = []
    diagnostics_by_method: dict[str, list[dict]] = {}
    for method in methods:
        print(f"\n=== Running {method.name} ===")
        start = time.perf_counter()
        final_ref, _history, params, meta = api.run_alignment(
            stack,
            init_ref,
            num_iterations=args.iterations,
            mask_diameter=mask_diameter,
            use_gpu=method.use_gpu,
            n_jobs=method.n_jobs,
            batch_size=args.gpu_batch_size,
            verbose=args.verbose,
            use_multicandidate=method.use_multicandidate and not method.use_gpu,
            topk_initial=args.topk_initial,
            diagnostics_n=args.diagnostics_n if (method.use_multicandidate and not method.use_gpu) else 0,
        )
        elapsed = time.perf_counter() - start
        corrected = api.run_transform(stack, params, engine=meta["engine"])
        result = evaluate_method(
            method.name,
            final_ref,
            params,
            corrected,
            gt_img,
            gt_table,
            mask,
            elapsed,
            meta["engine"],
            geo,
        )
        results.append(result)
        diagnostics = meta.get("candidate_diagnostics", [])
        if diagnostics:
            diagnostics_by_method[method.name] = diagnostics
            write_candidate_diagnostics_csv(output_dir / f"{method.name}_candidate_diagnostics.csv", diagnostics, gt_table)
        write_particle_csv(output_dir / f"{method.name}_particle_errors.csv", result, gt_table)
        print(
            f"{method.name}: engine={result.engine}, elapsed={elapsed:.2f}s, "
            f"angle_MAE={result.summary['angle_mae_deg']:.3f} deg, "
            f"shift_MAE={result.summary['shift_mae_px']:.3f} px, "
            f"corrected_mean_corr={result.summary['corrected_mean_corr']:.3f}"
        )

    write_summary_csv(output_dir / "summary.csv", results)
    plot_overview(output_dir / "overview.png", gt_img, stack, results)
    plot_error_distributions(output_dir / "error_distributions.png", results)
    plot_parameter_scatter(output_dir / "parameter_scatter.png", gt_table, results)
    plot_parameter_scatter_gauge_corrected(output_dir / "parameter_scatter_gauge_corrected.png", gt_table, results)
    plot_angle_error_histograms(output_dir / "angle_error_histograms.png", results)
    plot_angle_error_comparison(output_dir / "angle_error_comparison.png", results)
    plot_selected_rank_histogram(output_dir / "selected_candidate_rank.png", diagnostics_by_method)

    print(f"\nWrote accuracy report to: {output_dir.resolve()}")
    print("Key files:")
    print(f"  - {output_dir / 'summary.csv'}")
    print(f"  - {output_dir / 'overview.png'}")
    print(f"  - {output_dir / 'error_distributions.png'}")
    print(f"  - {output_dir / 'parameter_scatter.png'}")
    print(f"  - {output_dir / 'parameter_scatter_gauge_corrected.png'}")
    print(f"  - {output_dir / 'angle_error_comparison.png'}")


if __name__ == "__main__":
    # Make multiprocessing safe on platforms that use spawn.
    api.multiprocessing.freeze_support()
    main()
