#!/usr/bin/env python3
"""Benchmark homogeneous RF and MRA poses against disjoint RELION references."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import socket
import time
import traceback
from typing import Any

import mrcfile
import numpy as np

import alignimg as ai
from alignimg._geometry import CENTER_CONVENTION

if __package__:
    from tools.prepare_re2dc_70s_pose_benchmark import jsonable
else:
    from prepare_re2dc_70s_pose_benchmark import jsonable


DEFAULT_BENCHMARK = Path(
    "data/re2dc_70s_testdata/prepared/re2dc_70s_pose_benchmark.json"
)
DEFAULT_OUTPUT = Path("validation-results/re2dc-70s-pose-benchmark.json")
ALL_MODES = (
    "oracle",
    "known_reference",
    "homogeneous_rf",
    "fixed_mra",
    "open_mra",
    "adaptive_known_reference",
    "adaptive_fixed_mra",
)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def distribution_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("alignimg", "alignimg-gpu", "numpy", "scipy", "cupy-cuda12x"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def environment_info() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "alignimg_version": ai.__version__,
        "versions": distribution_versions(),
        "backends": ai.available_alignment_backends(),
    }


def resolve_artifact(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.exists() or path.is_absolute():
        return path
    candidate = manifest_path.parent / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(value)


def load_benchmark(
    manifest_path: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, np.ndarray], ai.PoseSet]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    particle_path = resolve_artifact(manifest_path, artifacts["particles"])
    reference_path = resolve_artifact(manifest_path, artifacts["references"])
    truth_path = resolve_artifact(manifest_path, artifacts["truth"])
    with mrcfile.mmap(particle_path, permissive=True, mode="r") as mrc:
        particles = np.asarray(mrc.data, dtype=np.float32).copy()
    with mrcfile.mmap(reference_path, permissive=True, mode="r") as mrc:
        references = np.asarray(mrc.data, dtype=np.float32).copy()
    with np.load(truth_path, allow_pickle=False) as saved:
        truth = {name: saved[name].copy() for name in saved.files}
    stored_center = str(np.asarray(truth["center_convention"]).item())
    if stored_center != CENTER_CONVENTION:
        raise ValueError(
            f"benchmark center convention {stored_center!r} does not match "
            f"{CENTER_CONVENTION!r}"
        )
    component = np.asarray(truth["component_index"], dtype=np.int32)
    relion_poses = ai.PoseSet(
        truth["relion_pose_angle_deg"],
        truth["relion_pose_shift_y_px"],
        truth["relion_pose_shift_x_px"],
        np.zeros(len(component), dtype=np.bool_),
    )
    if particles.ndim != 3 or references.ndim != 3:
        raise ValueError("benchmark particles and references must be image stacks")
    if particles.shape[1:] != references.shape[1:]:
        raise ValueError("benchmark particle/reference shapes do not match")
    if len(particles) != len(component) or len(particles) != len(relion_poses):
        raise ValueError("benchmark truth particle count does not match the stack")
    if np.any(component < 0) or np.any(component >= len(references)):
        raise ValueError("benchmark component labels are outside the reference range")
    return manifest, particles, references, truth, relion_poses


def subset_poses(poses: ai.PoseSet, selected: np.ndarray) -> ai.PoseSet:
    return ai.PoseSet(
        poses.angle_deg[selected],
        poses.shift_y_px[selected],
        poses.shift_x_px[selected],
        poses.mirror[selected],
    )


def wrapped_angle(values: np.ndarray, period: float = 360.0) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) + period / 2.0) % period - period / 2.0


def rotation_matrix(angle_deg: float) -> np.ndarray:
    radians = np.deg2rad(float(angle_deg))
    return np.asarray(
        [
            [np.cos(radians), np.sin(radians)],
            [-np.sin(radians), np.cos(radians)],
        ],
        dtype=np.float64,
    )


def compose_pose_gauge(reference: ai.PoseSet, gauge: ai.PoseSet) -> ai.PoseSet:
    """Return gauge∘reference using AlignImg's rotate-then-shift convention."""
    if len(gauge) != 1:
        raise ValueError("gauge must contain exactly one pose")
    if np.any(reference.mirror) or bool(gauge.mirror[0]):
        raise ValueError(
            "the homogeneous pose benchmark does not support mirror gauges"
        )
    angle = wrapped_angle(reference.angle_deg + float(gauge.angle_deg[0]))
    rotation = rotation_matrix(float(gauge.angle_deg[0]))
    reference_shift = np.column_stack(
        (reference.shift_x_px, reference.shift_y_px)
    ).astype(np.float64)
    gauge_shift = np.asarray(
        [gauge.shift_x_px[0], gauge.shift_y_px[0]], dtype=np.float64
    )
    composed_shift = reference_shift @ rotation.T + gauge_shift
    return ai.PoseSet(
        angle.astype(np.float32),
        composed_shift[:, 1].astype(np.float32),
        composed_shift[:, 0].astype(np.float32),
        np.zeros(len(reference), dtype=np.bool_),
    )


def fit_pose_gauge(predicted: ai.PoseSet, reference: ai.PoseSet) -> ai.PoseSet:
    """Fit one robust global SE(2) transform relating two pose coordinate frames."""
    if len(predicted) != len(reference) or len(predicted) == 0:
        raise ValueError("pose sets must have equal positive length")
    if np.any(predicted.mirror) or np.any(reference.mirror):
        raise ValueError("the homogeneous pose benchmark requires mirror-free poses")
    delta = wrapped_angle(predicted.angle_deg - reference.angle_deg)
    circular_mean = np.rad2deg(
        np.arctan2(
            np.mean(np.sin(np.deg2rad(delta))),
            np.mean(np.cos(np.deg2rad(delta))),
        )
    )
    unwrapped = circular_mean + wrapped_angle(delta - circular_mean)
    gauge_angle = float(wrapped_angle(np.asarray([np.median(unwrapped)]))[0])
    rotation = rotation_matrix(gauge_angle)
    reference_shift = np.column_stack(
        (reference.shift_x_px, reference.shift_y_px)
    ).astype(np.float64)
    predicted_shift = np.column_stack(
        (predicted.shift_x_px, predicted.shift_y_px)
    ).astype(np.float64)
    shift_residual = predicted_shift - reference_shift @ rotation.T
    gauge_shift = np.median(shift_residual, axis=0)
    return ai.PoseSet(
        np.asarray([gauge_angle], dtype=np.float32),
        np.asarray([gauge_shift[1]], dtype=np.float32),
        np.asarray([gauge_shift[0]], dtype=np.float32),
        np.asarray([False]),
    )


def quantiles(values: np.ndarray) -> dict[str, float]:
    result = np.quantile(
        np.asarray(values, dtype=np.float64), (0.0, 0.5, 0.9, 0.95, 1.0)
    )
    return dict(zip(("minimum", "median", "p90", "p95", "maximum"), result))


def pose_error_metrics(predicted: ai.PoseSet, reference: ai.PoseSet) -> dict[str, Any]:
    gauge = fit_pose_gauge(predicted, reference)
    expected = compose_pose_gauge(reference, gauge)
    angle_error = np.abs(wrapped_angle(predicted.angle_deg - expected.angle_deg))
    angle_error_180 = np.abs(
        wrapped_angle(predicted.angle_deg - expected.angle_deg, period=180.0)
    )
    shift_error = np.hypot(
        predicted.shift_y_px - expected.shift_y_px,
        predicted.shift_x_px - expected.shift_x_px,
    )
    return {
        "particle_count": len(predicted),
        "gauge": {
            "angle_deg": float(gauge.angle_deg[0]),
            "shift_y_px": float(gauge.shift_y_px[0]),
            "shift_x_px": float(gauge.shift_x_px[0]),
        },
        "angle_absolute_error_deg": quantiles(angle_error),
        "angle_modulo_180_absolute_error_deg": quantiles(angle_error_180),
        "shift_error_px": quantiles(shift_error),
        "angle_within_2_8125_deg": float(np.mean(angle_error <= 2.8125)),
        "angle_within_5_625_deg": float(np.mean(angle_error <= 5.625)),
        "angle_within_11_25_deg": float(np.mean(angle_error <= 11.25)),
        "shift_within_1_px": float(np.mean(shift_error <= 1.0)),
        "shift_within_2_px": float(np.mean(shift_error <= 2.0)),
        "mirror_mismatch_fraction": float(
            np.mean(predicted.mirror != reference.mirror)
        ),
    }


def image_correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    return float(
        np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    )


def stack_correlations(images: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(images, dtype=np.float64).reshape(len(images), -1)
    target = np.asarray(reference, dtype=np.float64).ravel()
    values -= values.mean(axis=1, keepdims=True)
    target -= target.mean()
    return (values @ target) / np.maximum(
        np.linalg.norm(values, axis=1) * np.linalg.norm(target), 1e-12
    )


def stable_frc_summary(result: ai.AlignmentResult) -> dict[str, Any] | None:
    if not result.diagnostics:
        return None
    final = result.diagnostics[-1]
    if "frc_0143_stable_cutoff_cyc_per_px" not in final:
        return None
    weights = np.asarray(final["halfset_effective_weight"], dtype=np.float64)
    cutoff = np.asarray(final["frc_0143_stable_cutoff_cyc_per_px"], dtype=np.float64)
    reliable = np.min(weights, axis=0) >= 10.0
    return {
        "minimum_halfset_effective_weight": np.min(weights, axis=0),
        "reliable_component_mask": reliable,
        "stable_cutoff_cyc_per_px": cutoff,
        "reliable_median_stable_cutoff_cyc_per_px": (
            float(np.median(cutoff[reliable])) if np.any(reliable) else None
        ),
    }


def evaluate_result(
    result: ai.AlignmentResult,
    particles: np.ndarray,
    oracle_references: np.ndarray,
    relion_poses: ai.PoseSet,
    true_component: np.ndarray,
    *,
    backend: str,
    seconds: float,
    require_correct_assignment: bool,
) -> dict[str, Any]:
    assigned = np.asarray(result.reference_assignments, dtype=np.int32)
    output: dict[str, Any] = {
        "seconds": seconds,
        "particles_per_second": len(particles) / max(seconds, 1e-12),
        "particle_iterations_per_second": (
            len(particles) * len(result.diagnostics) / max(seconds, 1e-12)
        ),
        "assignment_accuracy": float(np.mean(assigned == true_component)),
        "assignment_counts": np.bincount(assigned, minlength=len(oracle_references)),
        "responsibility_sum_max_error": float(
            np.max(np.abs(result.responsibilities.sum(axis=1) - 1.0))
        ),
        "mean_max_responsibility": float(
            np.mean(np.max(result.responsibilities, axis=1))
        ),
        "pose_entropy": quantiles(result.pose_entropy),
        "map_posterior": quantiles(result.map_posterior),
        "stable_frc": stable_frc_summary(result),
        "metadata": result.metadata,
        "per_component": [],
    }
    if result.metadata.get("search_strategy") == "adaptive_posterior":
        names = (
            "mean_coarse_cell_count",
            "mean_selected_coarse_cell_count",
            "mean_selected_coarse_mass",
            "mean_fine_candidate_count",
            "mean_retained_fine_mass",
            "adaptive_cap_hit_count",
            "boundary_hit_count",
            "mean_normalized_pose_entropy",
            "rescue_entropy_trigger_count",
            "rescue_map_trigger_count",
            "rescue_particle_count",
            "rescue_accepted_count",
            "rescue_rejected_count",
            "mean_rescue_score_gain",
            "rescue_scheduled_count",
        )
        output["adaptive_diagnostics"] = [
            {name: item[name] for name in names if name in item}
            for item in result.diagnostics
        ]
    for component in range(len(oracle_references)):
        selected = true_component == component
        if require_correct_assignment:
            selected &= assigned == component
        indices = np.flatnonzero(selected)
        if len(indices) < 2:
            output["per_component"].append(
                {"component_index": component, "particle_count": len(indices)}
            )
            continue
        predicted = subset_poses(result.poses, indices)
        reference = subset_poses(relion_poses, indices)
        pose_metrics = pose_error_metrics(predicted, reference)
        gauge_values = pose_metrics["gauge"]
        gauge = ai.PoseSet(
            np.asarray([gauge_values["angle_deg"]]),
            np.asarray([gauge_values["shift_y_px"]]),
            np.asarray([gauge_values["shift_x_px"]]),
            np.asarray([False]),
        )
        oracle_in_result_frame = ai.transform_images(
            oracle_references[component][None], gauge, backend="cpu"
        )[0]
        aligned = ai.transform_images(particles[indices], predicted, backend=backend)
        aligned_mean = aligned.mean(axis=0)
        output["per_component"].append(
            {
                "component_index": component,
                "particle_count": len(indices),
                "pose": pose_metrics,
                "result_reference_to_gauged_oracle_correlation": image_correlation(
                    result.references[component], oracle_in_result_frame
                ),
                "aligned_mean_to_gauged_oracle_correlation": image_correlation(
                    aligned_mean, oracle_in_result_frame
                ),
                "mean_aligned_particle_ncc_to_result_reference": float(
                    np.mean(stack_correlations(aligned, result.references[component]))
                ),
            }
        )
    return output


def oracle_metrics(
    particles: np.ndarray,
    references: np.ndarray,
    relion_poses: ai.PoseSet,
    component: np.ndarray,
) -> dict[str, Any]:
    aligned = ai.transform_images(particles, relion_poses, backend="cpu")
    per_component = []
    for index in range(len(references)):
        selected = component == index
        mean = aligned[selected].mean(axis=0)
        per_component.append(
            {
                "component_index": index,
                "particle_count": int(np.sum(selected)),
                "evaluation_mean_to_disjoint_reference_correlation": image_correlation(
                    mean, references[index]
                ),
                "mean_particle_ncc_to_disjoint_reference": float(
                    np.mean(stack_correlations(aligned[selected], references[index]))
                ),
                "mean_particle_ncc_to_evaluation_mean": float(
                    np.mean(stack_correlations(aligned[selected], mean))
                ),
            }
        )
    return {"per_component": per_component}


def reproducibility_metrics(
    first: ai.AlignmentResult, second: ai.AlignmentResult
) -> dict[str, Any]:
    angle_delta = np.abs(wrapped_angle(first.poses.angle_deg - second.poses.angle_deg))
    shift_delta = np.hypot(
        first.poses.shift_y_px - second.poses.shift_y_px,
        first.poses.shift_x_px - second.poses.shift_x_px,
    )
    return {
        "assignment_agreement": float(
            np.mean(first.reference_assignments == second.reference_assignments)
        ),
        "mirror_agreement": float(np.mean(first.poses.mirror == second.poses.mirror)),
        "angle_absolute_delta_deg": quantiles(angle_delta),
        "shift_delta_px": quantiles(shift_delta),
        "responsibility_max_absolute_delta": float(
            np.max(np.abs(first.responsibilities - second.responsibilities))
        ),
        "reference_correlations": np.asarray(
            [
                image_correlation(left, right)
                for left, right in zip(first.references, second.references)
            ]
        ),
    }


def pose_change_metrics(initial: ai.PoseSet, refined: ai.PoseSet) -> dict[str, Any]:
    if len(initial) != len(refined):
        raise ValueError("initial and refined pose sets must have equal length")
    angle_delta = np.abs(wrapped_angle(refined.angle_deg - initial.angle_deg))
    shift_delta = np.hypot(
        refined.shift_y_px - initial.shift_y_px,
        refined.shift_x_px - initial.shift_x_px,
    )
    return {
        "angle_absolute_delta_deg": quantiles(angle_delta),
        "shift_delta_px": quantiles(shift_delta),
        "mirror_change_fraction": float(np.mean(initial.mirror != refined.mirror)),
    }


def fixed_mra_k1_equivalence(
    joint: ai.AlignmentResult,
    independent: list[ai.AlignmentResult],
    component: np.ndarray,
) -> list[dict[str, Any]]:
    """Compare fixed-class MRA slices with independently processed K=1 runs."""
    output = []
    for index, single in enumerate(independent):
        selected = np.asarray(component) == index
        angle_delta = np.abs(
            wrapped_angle(joint.poses.angle_deg[selected] - single.poses.angle_deg)
        )
        shift_delta = np.hypot(
            joint.poses.shift_y_px[selected] - single.poses.shift_y_px,
            joint.poses.shift_x_px[selected] - single.poses.shift_x_px,
        )
        output.append(
            {
                "component_index": index,
                "maximum_angle_delta_deg": float(np.max(angle_delta)),
                "maximum_shift_delta_px": float(np.max(shift_delta)),
                "maximum_responsibility_delta": float(
                    np.max(
                        np.abs(
                            joint.responsibilities[selected, index]
                            - single.responsibilities[:, 0]
                        )
                    )
                ),
                "reference_correlation": image_correlation(
                    joint.references[index], single.references[0]
                ),
            }
        )
    return output


def repeat_alignment(callable_, count: int):
    primary, primary_seconds = timed(callable_)
    repeats = []
    for repeat_index in range(1, count):
        repeated, seconds = timed(callable_)
        repeats.append(
            {
                "repeat_index": repeat_index,
                "seconds": seconds,
                "comparison_to_primary": reproducibility_metrics(primary, repeated),
            }
        )
    return primary, primary_seconds, repeats


def summarize_homogeneous_rf_runs(
    runs: list[dict[str, Any]], class_numbers: np.ndarray
) -> list[dict[str, Any]]:
    summary = []
    for class_number in class_numbers:
        selected = [
            run for run in runs if run["relion_class_number"] == int(class_number)
        ]
        angle_median = np.asarray(
            [
                run["per_component"][0]["pose"]["angle_absolute_error_deg"]["median"]
                for run in selected
            ]
        )
        shift_median = np.asarray(
            [
                run["per_component"][0]["pose"]["shift_error_px"]["median"]
                for run in selected
            ]
        )
        reference_correlation = np.asarray(
            [
                run["per_component"][0]["result_reference_to_gauged_oracle_correlation"]
                for run in selected
            ]
        )
        summary.append(
            {
                "relion_class_number": int(class_number),
                "seed_count": len(selected),
                "median_pose_angle_error_deg_across_seeds": quantiles(angle_median),
                "median_pose_shift_error_px_across_seeds": quantiles(shift_median),
                "reference_correlation_across_seeds": quantiles(reference_correlation),
            }
        )
    return summary


def save_result(
    base: Path,
    name: str,
    result: ai.AlignmentResult,
    *,
    pixel_size_angstrom: float,
) -> dict[str, str]:
    reference_path = Path(f"{base}.{name}.references.mrcs")
    result_path = Path(f"{base}.{name}.result.npz")
    with mrcfile.new(reference_path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(result.references, dtype=np.float32))
        mrc.voxel_size = pixel_size_angstrom
        mrc.update_header_stats()
    history = (
        np.asarray(result.reference_history, dtype=np.float32)
        if result.reference_history
        else result.references[None]
    )
    np.savez_compressed(
        result_path,
        alignimg_version=np.asarray(ai.__version__),
        center_convention=np.asarray(result.metadata["center_convention"]),
        reference_history=history,
        angle_deg=result.poses.angle_deg,
        shift_y_px=result.poses.shift_y_px,
        shift_x_px=result.poses.shift_x_px,
        mirror=result.poses.mirror,
        assignments=result.reference_assignments,
        responsibilities=result.responsibilities,
        inlier_weights=result.inlier_weights,
        pose_entropy=result.pose_entropy,
        map_posterior=result.map_posterior,
    )
    return {"references": str(reference_path), "result": str(result_path)}


def alignment_config(
    *,
    workflow: str,
    iterations: int,
    args: argparse.Namespace,
    seed: int,
) -> ai.AlignmentConfig:
    is_rf = workflow == "reference_free"
    return ai.AlignmentConfig(
        max_iterations=iterations,
        top_l=args.top_l,
        angle_samples=args.angle_samples,
        proposal_angles_per_reference=args.proposal_angles,
        translation_range=args.translation_range,
        translation_step=1.0,
        candidate_scoring=getattr(args, "candidate_scoring", "raster"),
        score_model=getattr(args, "score_model", "fourier_ncc"),
        reference_update=getattr(args, "reference_update", "spatial"),
        temperature_start=0.08,
        temperature_end=0.02 if is_rf else 0.05,
        temperature_anneal_iterations=(min(10, iterations) if is_rf else None),
        mirror_search=False,
        random_seed=seed,
        robust_weighting=False,
        halfset_diagnostics=True,
        center_references=is_rf,
        store_history=True,
        batch_size=args.batch_size,
        memory_fraction=args.memory_fraction,
    ).normalized(workflow="reference_free" if is_rf else "global")


def adaptive_alignment_config(
    *,
    args: argparse.Namespace,
    seed: int,
) -> ai.AlignmentConfig:
    return ai.AlignmentConfig(
        search_strategy="adaptive_posterior",
        max_iterations=args.adaptive_iterations,
        top_l=args.top_l,
        angle_samples=args.angle_samples,
        proposal_angles_per_reference=args.proposal_angles,
        translation_range=args.translation_range,
        translation_step=1.0,
        candidate_scoring=getattr(args, "candidate_scoring", "raster"),
        score_model=getattr(args, "score_model", "fourier_ncc"),
        reference_update=getattr(args, "reference_update", "spatial"),
        coarse_angle_step=args.coarse_angle_step,
        coarse_shift_step=args.coarse_shift_step,
        local_angle_range=args.local_angle_range,
        local_shift_range=args.local_shift_range,
        adaptive_fraction=args.adaptive_fraction,
        oversampling_order=args.oversampling_order,
        max_adaptive_cells=(args.max_adaptive_cells or None),
        rescue_uncertain_particles=args.rescue_uncertain_particles,
        rescue_normalized_entropy_threshold=(args.rescue_normalized_entropy_threshold),
        rescue_map_posterior_threshold=args.rescue_map_posterior_threshold,
        rescue_max_fraction=args.rescue_max_fraction,
        rescue_min_score_improvement=args.rescue_min_score_improvement,
        temperature_start=0.08,
        temperature_end=0.05,
        mirror_search=False,
        random_seed=seed,
        robust_weighting=False,
        halfset_diagnostics=True,
        center_references=False,
        store_history=True,
        batch_size=args.batch_size,
        memory_fraction=args.memory_fraction,
    ).normalized(workflow="refine")


def timed(callable_):
    started = time.perf_counter()
    result = callable_()
    return result, time.perf_counter() - started


def main(args: argparse.Namespace) -> None:
    modes = tuple(args.only) if args.only else ALL_MODES
    invalid = sorted(set(modes).difference(ALL_MODES))
    if invalid:
        raise ValueError(f"unknown modes: {invalid}")
    if args.deterministic_repeats < 1:
        raise ValueError("deterministic_repeats must be positive")
    manifest, particles, references, truth, relion_poses = load_benchmark(
        args.benchmark
    )
    component = np.asarray(truth["component_index"], dtype=np.int32)
    class_numbers = np.asarray(truth["relion_class_numbers"], dtype=np.int32)
    pixel_size = float(np.asarray(truth["pixel_size_angstrom"]).item())
    base = args.output.with_suffix("")
    report: dict[str, Any] = {
        "schema": "alignimg.re2dc-70s-pose-benchmark.v1",
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_info(),
        "input": {
            "benchmark_manifest": str(args.benchmark),
            "particle_count": len(particles),
            "image_shape": list(particles.shape[1:]),
            "component_count": len(references),
            "relion_class_numbers": class_numbers,
            "relion_confidence_quantiles": quantiles(
                np.asarray(truth["relion_confidence"], dtype=np.float64)
            ),
            "data_manifest": manifest,
        },
        "scope_note": (
            "Pose errors are measured after fitting one global SE(2) gauge per RELION "
            "class. Open-MRA assignment accuracy is diagnostic and is not interpreted "
            "as biological classification accuracy."
        ),
        "parameters": {
            "backend": args.backend,
            "candidate_scoring": getattr(args, "candidate_scoring", "raster"),
            "score_model": getattr(args, "score_model", "fourier_ncc"),
            "reference_update": getattr(args, "reference_update", "spatial"),
            "batch_size": args.batch_size,
            "memory_fraction": args.memory_fraction,
            "angle_samples": args.angle_samples,
            "proposal_angles": args.proposal_angles,
            "top_l": args.top_l,
            "translation_range": args.translation_range,
            "global_iterations": args.global_iterations,
            "rf_iterations": args.rf_iterations,
            "mra_iterations": args.mra_iterations,
            "adaptive_iterations": args.adaptive_iterations,
            "coarse_angle_step": args.coarse_angle_step,
            "coarse_shift_step": args.coarse_shift_step,
            "local_angle_range": args.local_angle_range,
            "local_shift_range": args.local_shift_range,
            "adaptive_fraction": args.adaptive_fraction,
            "oversampling_order": args.oversampling_order,
            "max_adaptive_cells": args.max_adaptive_cells,
            "rescue_uncertain_particles": args.rescue_uncertain_particles,
            "rescue_normalized_entropy_threshold": (
                args.rescue_normalized_entropy_threshold
            ),
            "rescue_map_posterior_threshold": args.rescue_map_posterior_threshold,
            "rescue_max_fraction": args.rescue_max_fraction,
            "rescue_min_score_improvement": args.rescue_min_score_improvement,
            "rf_seeds": args.rf_seeds,
            "deterministic_repeats": args.deterministic_repeats,
            "modes": modes,
        },
        "runs": {},
    }
    write_json(args.output, report)
    adaptive_results: list[ai.AlignmentResult] = []
    try:
        if "oracle" in modes:
            print("RUN  RELION oracle", flush=True)
            report["runs"]["oracle"] = oracle_metrics(
                particles, references, relion_poses, component
            )
            write_json(args.output, report)

        if "known_reference" in modes:
            print("RUN  K=1 known-reference global alignment", flush=True)
            known_runs = []
            for index, class_number in enumerate(class_numbers):
                selected = np.flatnonzero(component == index)
                config = alignment_config(
                    workflow="global",
                    iterations=args.global_iterations,
                    args=args,
                    seed=int(class_number),
                )
                alignment_call = lambda selected=selected, index=index, config=config: (
                    ai.align_to_references(
                        particles[selected],
                        references[index],
                        config=config,
                        backend=args.backend,
                    )
                )
                result, seconds, repeats = repeat_alignment(
                    alignment_call, args.deterministic_repeats
                )
                local_component = np.zeros(len(selected), dtype=np.int32)
                metrics = evaluate_result(
                    result,
                    particles[selected],
                    references[index][None],
                    subset_poses(relion_poses, selected),
                    local_component,
                    backend=args.backend,
                    seconds=seconds,
                    require_correct_assignment=False,
                )
                metrics.update(
                    relion_class_number=int(class_number),
                    reproducibility_repeats=repeats,
                    config=asdict(config),
                    artifacts=save_result(
                        base,
                        f"known-reference.class-{int(class_number)}",
                        result,
                        pixel_size_angstrom=pixel_size,
                    ),
                )
                known_runs.append(metrics)
                report["runs"]["known_reference"] = known_runs
                write_json(args.output, report)

        if "homogeneous_rf" in modes:
            print("RUN  homogeneous K=1 reference-free alignment", flush=True)
            rf_runs = []
            for seed in args.rf_seeds:
                for index, class_number in enumerate(class_numbers):
                    selected = np.flatnonzero(component == index)
                    config = alignment_config(
                        workflow="reference_free",
                        iterations=args.rf_iterations,
                        args=args,
                        seed=seed,
                    )
                    result, seconds = timed(
                        lambda selected=selected, config=config: (
                            ai.reference_free_align(
                                particles[selected],
                                n_components=1,
                                config=config,
                                backend=args.backend,
                            )
                        )
                    )
                    local_component = np.zeros(len(selected), dtype=np.int32)
                    metrics = evaluate_result(
                        result,
                        particles[selected],
                        references[index][None],
                        subset_poses(relion_poses, selected),
                        local_component,
                        backend=args.backend,
                        seconds=seconds,
                        require_correct_assignment=False,
                    )
                    metrics.update(
                        relion_class_number=int(class_number),
                        random_seed=seed,
                        config=asdict(config),
                        artifacts=save_result(
                            base,
                            f"homogeneous-rf.class-{int(class_number)}.seed-{seed}",
                            result,
                            pixel_size_angstrom=pixel_size,
                        ),
                    )
                    rf_runs.append(metrics)
                    report["runs"]["homogeneous_rf"] = rf_runs
                    write_json(args.output, report)
            report["runs"]["homogeneous_rf_summary"] = summarize_homogeneous_rf_runs(
                rf_runs, class_numbers
            )
            write_json(args.output, report)

        if "fixed_mra" in modes:
            print("RUN  fixed-class MRA pose alignment", flush=True)
            config = alignment_config(
                workflow="global",
                iterations=args.mra_iterations,
                args=args,
                seed=0,
            )
            priors = ai.make_class_priors(
                assignments=component,
                n_components=len(references),
                trust=1.0,
            )
            alignment_call = lambda: ai.align_to_references(
                particles,
                references,
                class_priors=priors,
                config=config,
                backend=args.backend,
            )
            result, seconds, repeats = repeat_alignment(
                alignment_call, args.deterministic_repeats
            )
            if not np.array_equal(result.reference_assignments, component):
                raise AssertionError("fixed-class MRA changed a class assignment")
            metrics = evaluate_result(
                result,
                particles,
                references,
                relion_poses,
                component,
                backend=args.backend,
                seconds=seconds,
                require_correct_assignment=False,
            )
            metrics.update(
                reproducibility_repeats=repeats,
                config=asdict(config),
                artifacts=save_result(
                    base,
                    "fixed-mra",
                    result,
                    pixel_size_angstrom=pixel_size,
                ),
            )
            report["runs"]["fixed_mra"] = metrics
            write_json(args.output, report)

        if "open_mra" in modes:
            print("RUN  open MRA diagnostic", flush=True)
            config = alignment_config(
                workflow="global",
                iterations=args.mra_iterations,
                args=args,
                seed=0,
            )
            alignment_call = lambda: ai.align_to_references(
                particles,
                references,
                config=config,
                backend=args.backend,
            )
            result, seconds, repeats = repeat_alignment(
                alignment_call, args.deterministic_repeats
            )
            metrics = evaluate_result(
                result,
                particles,
                references,
                relion_poses,
                component,
                backend=args.backend,
                seconds=seconds,
                require_correct_assignment=True,
            )
            metrics.update(
                reproducibility_repeats=repeats,
                config=asdict(config),
                artifacts=save_result(
                    base,
                    "open-mra",
                    result,
                    pixel_size_angstrom=pixel_size,
                ),
            )
            report["runs"]["open_mra"] = metrics
            write_json(args.output, report)

        if "adaptive_known_reference" in modes:
            print("RUN  adaptive K=1 known-reference refinement", flush=True)
            adaptive_runs = []
            for index, class_number in enumerate(class_numbers):
                selected = np.flatnonzero(component == index)
                initializer_config = alignment_config(
                    workflow="global",
                    iterations=args.global_iterations,
                    args=args,
                    seed=int(class_number),
                )
                initializer, initializer_seconds = timed(
                    lambda selected=selected, index=index, initializer_config=initializer_config: (
                        ai.align_to_references(
                            particles[selected],
                            references[index],
                            config=initializer_config,
                            backend=args.backend,
                        )
                    )
                )
                local_component = np.zeros(len(selected), dtype=np.int32)
                initializer_metrics = evaluate_result(
                    initializer,
                    particles[selected],
                    references[index][None],
                    subset_poses(relion_poses, selected),
                    local_component,
                    backend=args.backend,
                    seconds=initializer_seconds,
                    require_correct_assignment=False,
                )
                initializer_metrics.update(
                    config=asdict(initializer_config),
                    artifacts=save_result(
                        base,
                        f"adaptive-known-reference.class-{int(class_number)}.initial",
                        initializer,
                        pixel_size_angstrom=pixel_size,
                    ),
                )
                config = adaptive_alignment_config(
                    args=args,
                    seed=int(class_number),
                )
                refinement_call = (
                    lambda selected=selected, initializer=initializer, config=config: (
                        ai.refine_alignment(
                            particles[selected],
                            initializer.references,
                            initializer.poses,
                            config=config,
                            backend=args.backend,
                        )
                    )
                )
                result, seconds, repeats = repeat_alignment(
                    refinement_call, args.deterministic_repeats
                )
                metrics = evaluate_result(
                    result,
                    particles[selected],
                    references[index][None],
                    subset_poses(relion_poses, selected),
                    local_component,
                    backend=args.backend,
                    seconds=seconds,
                    require_correct_assignment=False,
                )
                metrics.update(
                    relion_class_number=int(class_number),
                    initializer=initializer_metrics,
                    refinement_pose_change=pose_change_metrics(
                        initializer.poses, result.poses
                    ),
                    reproducibility_repeats=repeats,
                    config=asdict(config),
                    artifacts=save_result(
                        base,
                        f"adaptive-known-reference.class-{int(class_number)}",
                        result,
                        pixel_size_angstrom=pixel_size,
                    ),
                )
                adaptive_runs.append(metrics)
                adaptive_results.append(result)
                report["runs"]["adaptive_known_reference"] = adaptive_runs
                write_json(args.output, report)

        if "adaptive_fixed_mra" in modes:
            print("RUN  adaptive fixed-class MRA refinement", flush=True)
            priors = ai.make_class_priors(
                assignments=component,
                n_components=len(references),
                trust=1.0,
            )
            initializer_config = alignment_config(
                workflow="global",
                iterations=args.mra_iterations,
                args=args,
                seed=0,
            )
            initializer, initializer_seconds = timed(
                lambda: ai.align_to_references(
                    particles,
                    references,
                    class_priors=priors,
                    config=initializer_config,
                    backend=args.backend,
                )
            )
            if not np.array_equal(initializer.reference_assignments, component):
                raise AssertionError("fixed-class initializer changed an assignment")
            initializer_metrics = evaluate_result(
                initializer,
                particles,
                references,
                relion_poses,
                component,
                backend=args.backend,
                seconds=initializer_seconds,
                require_correct_assignment=False,
            )
            initializer_metrics.update(
                config=asdict(initializer_config),
                artifacts=save_result(
                    base,
                    "adaptive-fixed-mra.initial",
                    initializer,
                    pixel_size_angstrom=pixel_size,
                ),
            )
            config = adaptive_alignment_config(args=args, seed=0)
            refinement_call = lambda: ai.refine_alignment(
                particles,
                initializer.references,
                initializer.poses,
                class_priors=priors,
                config=config,
                backend=args.backend,
            )
            result, seconds, repeats = repeat_alignment(
                refinement_call, args.deterministic_repeats
            )
            if not np.array_equal(result.reference_assignments, component):
                raise AssertionError("adaptive fixed-class MRA changed an assignment")
            metrics = evaluate_result(
                result,
                particles,
                references,
                relion_poses,
                component,
                backend=args.backend,
                seconds=seconds,
                require_correct_assignment=False,
            )
            metrics.update(
                initializer=initializer_metrics,
                refinement_pose_change=pose_change_metrics(
                    initializer.poses, result.poses
                ),
                reproducibility_repeats=repeats,
                config=asdict(config),
                artifacts=save_result(
                    base,
                    "adaptive-fixed-mra",
                    result,
                    pixel_size_angstrom=pixel_size,
                ),
            )
            if len(adaptive_results) == len(references):
                equivalence = fixed_mra_k1_equivalence(
                    result, adaptive_results, component
                )
                metrics["fixed_mra_k1_equivalence"] = equivalence
                if any(
                    item["maximum_angle_delta_deg"] > 1e-6
                    or item["maximum_shift_delta_px"] > 1e-6
                    or item["maximum_responsibility_delta"] > 2e-5
                    or item["reference_correlation"] < 0.999999
                    for item in equivalence
                ):
                    raise AssertionError(
                        "fixed-class MRA diverged from independent K=1 refinement"
                    )
            report["runs"]["adaptive_fixed_mra"] = metrics
            write_json(args.output, report)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(args.output, report)
        raise
    report["status"] = "completed"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output, report)
    print(f"Report saved to: {args.output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--backend", choices=("cpu", "cuda", "cupy", "gpu", "auto"), default="cuda"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument(
        "--candidate-scoring",
        choices=("raster", "fourier"),
        default="raster",
    )
    parser.add_argument(
        "--score-model",
        choices=("fourier_ncc", "whitened_fourier_ncc"),
        default="fourier_ncc",
        help="Fourier score weighting model (default: unweighted NCC).",
    )
    parser.add_argument(
        "--reference-update",
        choices=("spatial", "fourier"),
        default="spatial",
        help="Reference M-step domain (default: spatial).",
    )
    parser.add_argument("--angle-samples", type=int, default=128)
    parser.add_argument("--proposal-angles", type=int, default=8)
    parser.add_argument("--top-l", type=int, default=8)
    parser.add_argument("--translation-range", type=float, default=20.0)
    parser.add_argument("--global-iterations", type=int, default=3)
    parser.add_argument("--rf-iterations", type=int, default=15)
    parser.add_argument("--mra-iterations", type=int, default=3)
    parser.add_argument("--adaptive-iterations", type=int, default=3)
    parser.add_argument("--coarse-angle-step", type=float, default=6.0)
    parser.add_argument("--coarse-shift-step", type=float, default=1.0)
    parser.add_argument("--local-angle-range", type=float, default=15.0)
    parser.add_argument("--local-shift-range", type=float, default=3.0)
    parser.add_argument("--adaptive-fraction", type=float, default=0.999)
    parser.add_argument("--oversampling-order", type=int, default=1)
    parser.add_argument(
        "--max-adaptive-cells",
        type=int,
        default=32,
        help="Adaptive coarse-cell safety cap; use 0 to disable.",
    )
    parser.add_argument("--rescue-uncertain-particles", action="store_true")
    parser.add_argument("--rescue-normalized-entropy-threshold", type=float)
    parser.add_argument("--rescue-map-posterior-threshold", type=float)
    parser.add_argument("--rescue-max-fraction", type=float, default=0.05)
    parser.add_argument("--rescue-min-score-improvement", type=float, default=0.02)
    parser.add_argument("--rf-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--deterministic-repeats", type=int, default=2)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=ALL_MODES,
        help="Run only selected benchmark modes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
