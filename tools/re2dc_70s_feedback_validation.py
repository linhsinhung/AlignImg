#!/usr/bin/env python3
"""Compare fixed-class and corrective-MRA refinement on prepared 70S data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import socket
import time
import traceback
from typing import Any
import warnings

import mrcfile
import numpy as np

import alignimg as ai
from alignimg._geometry import CENTER_CONVENTION


DEFAULT_STACK = Path(
    "data/re2dc_70s_testdata/prepared/re2dc_70s_n1000_s128.mrcs"
)
DEFAULT_REFERENCES = Path(
    "validation-results/re2dc-70s-final-n1000.rf.seed-0.references.mrcs"
)
DEFAULT_RF_RESULT = Path(
    "validation-results/re2dc-70s-final-n1000.rf.seed-0.result.npz"
)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def transition_matrix(
    initial: np.ndarray, final: np.ndarray, component_count: int
) -> np.ndarray:
    matrix = np.zeros((component_count, component_count), dtype=np.int64)
    np.add.at(matrix, (initial, final), 1)
    return matrix


def stack_correlations(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    first = values.reshape(len(values), -1).astype(np.float64)
    second = targets.reshape(len(targets), -1).astype(np.float64)
    first -= first.mean(axis=1, keepdims=True)
    second -= second.mean(axis=1, keepdims=True)
    numerator = np.sum(first * second, axis=1)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def pose_delta(initial: ai.PoseSet, final: ai.PoseSet) -> dict[str, Any]:
    angle = np.abs(
        (final.angle_deg - initial.angle_deg + 180.0) % 360.0 - 180.0
    )
    shift = np.sqrt(
        (final.shift_y_px - initial.shift_y_px) ** 2
        + (final.shift_x_px - initial.shift_x_px) ** 2
    )
    return {
        "angle_absolute_deg": {
            "median": float(np.median(angle)),
            "p95": float(np.quantile(angle, 0.95)),
            "maximum": float(np.max(angle)),
        },
        "shift_magnitude_px": {
            "median": float(np.median(shift)),
            "p95": float(np.quantile(shift, 0.95)),
            "maximum": float(np.max(shift)),
        },
        "mirror_change_fraction": float(np.mean(final.mirror != initial.mirror)),
    }


def reproducibility_metrics(
    first: ai.AlignmentResult, second: ai.AlignmentResult
) -> dict[str, Any]:
    angle = np.abs(
        (first.poses.angle_deg - second.poses.angle_deg + 180.0) % 360.0 - 180.0
    )
    shift = np.hypot(
        first.poses.shift_y_px - second.poses.shift_y_px,
        first.poses.shift_x_px - second.poses.shift_x_px,
    )
    correlations = stack_correlations(first.references, second.references)
    metrics = {
        "assignment_agreement": float(
            np.mean(first.reference_assignments == second.reference_assignments)
        ),
        "mirror_agreement": float(np.mean(first.poses.mirror == second.poses.mirror)),
        "maximum_angle_delta_deg": float(np.max(angle)),
        "maximum_shift_delta_px": float(np.max(shift)),
        "maximum_responsibility_delta": float(
            np.max(np.abs(first.responsibilities - second.responsibilities))
        ),
        "reference_correlations": correlations,
    }
    metrics["passed"] = bool(
        metrics["assignment_agreement"] == 1.0
        and metrics["mirror_agreement"] == 1.0
        and metrics["maximum_angle_delta_deg"] <= 1e-6
        and metrics["maximum_shift_delta_px"] <= 1e-6
        and metrics["maximum_responsibility_delta"] <= 2e-7
        and np.min(correlations) >= 0.999999
    )
    return metrics


def stable_frc_metrics(
    diagnostic: dict[str, Any],
    *,
    minimum_halfset_weight: float,
    pixel_size: float,
) -> dict[str, Any]:
    weights = np.asarray(diagnostic["halfset_effective_weight"], dtype=np.float64)
    cutoff = np.asarray(
        diagnostic["frc_0143_stable_cutoff_cyc_per_px"], dtype=np.float64
    )
    reliable = np.min(weights, axis=0) >= minimum_halfset_weight
    valid = cutoff[reliable & (cutoff > 0.0)]
    resolutions = pixel_size / valid if len(valid) else np.asarray([], dtype=np.float64)
    return {
        "minimum_required_effective_weight_per_half": minimum_halfset_weight,
        "reliable_component_mask": reliable,
        "stable_cutoff_cyc_per_px": cutoff,
        "reliable_median_stable_cutoff_cyc_per_px": (
            float(np.median(valid)) if len(valid) else None
        ),
        "reliable_median_resolution_angstrom": (
            float(np.median(resolutions)) if len(resolutions) else None
        ),
    }


def save_result(
    base: Path,
    mode: str,
    result: ai.AlignmentResult,
    *,
    pixel_size: float,
) -> dict[str, str]:
    reference_path = Path(f"{base}.{mode}.references.mrcs")
    result_path = Path(f"{base}.{mode}.result.npz")
    with mrcfile.new(reference_path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(result.references, dtype=np.float32))
        mrc.voxel_size = pixel_size
        mrc.update_header_stats()
    np.savez_compressed(
        result_path,
        alignimg_version=np.asarray(ai.__version__),
        center_convention=np.asarray(result.metadata["center_convention"]),
        reference_history=np.asarray(result.reference_history, dtype=np.float32),
        angle_deg=result.poses.angle_deg,
        shift_y_px=result.poses.shift_y_px,
        shift_x_px=result.poses.shift_x_px,
        mirror=result.poses.mirror,
        assignments=result.reference_assignments,
        responsibilities=result.responsibilities,
        inlier_weights=result.inlier_weights,
    )
    return {"references": str(reference_path), "result": str(result_path)}


def result_metrics(
    result: ai.AlignmentResult,
    *,
    initial_assignments: np.ndarray,
    initial_poses: ai.PoseSet,
    initial_references: np.ndarray,
    seconds: float,
    minimum_halfset_weight: float,
    pixel_size: float,
) -> dict[str, Any]:
    component_count = len(initial_references)
    transitions = transition_matrix(
        initial_assignments, result.reference_assignments, component_count
    )
    final = result.diagnostics[-1]
    pose_search_keys = (
        "mean_screened_angle_count",
        "mean_active_reference_count",
        "mean_angular_mode_count",
        "mean_posterior_support_count",
        "translation_fit_attempt_count",
        "translation_fit_accept_count",
        "angle_fit_attempt_count",
        "angle_fit_accept_count",
        "quadratic_boundary_hit_count",
        "quadratic_flat_or_convex_reject_count",
        "quadratic_out_of_bounds_reject_count",
        "quadratic_exact_reject_count",
        "mean_quadratic_objective_gain",
        "correlation_map_ifft_count",
    )
    return {
        "seconds": seconds,
        "particles_per_second": len(result.poses) / max(seconds, 1e-12),
        "particle_iterations_per_second": (
            len(result.poses) * len(result.diagnostics) / max(seconds, 1e-12)
        ),
        "assignment_transition_matrix": transitions,
        "reassignment_fraction": float(
            np.mean(result.reference_assignments != initial_assignments)
        ),
        "hard_assignment_counts": np.bincount(
            result.reference_assignments, minlength=component_count
        ),
        "pose_delta_from_input": pose_delta(initial_poses, result.poses),
        "input_output_reference_correlations": stack_correlations(
            result.references, initial_references
        ),
        "mean_max_responsibility": float(
            np.mean(np.max(result.responsibilities, axis=1))
        ),
        "responsibility_sum_max_error": float(
            np.max(np.abs(result.responsibilities.sum(axis=1) - 1.0))
        ),
        "effective_component_weight": final["effective_component_weight"],
        "stable_frc": stable_frc_metrics(
            final,
            minimum_halfset_weight=minimum_halfset_weight,
            pixel_size=pixel_size,
        ),
        "temperature_trajectory": [
            item["temperature"] for item in result.diagnostics
        ],
        "reference_relative_change_trajectory": [
            item["reference_relative_change"] for item in result.diagnostics
        ],
        "mean_expected_fourier_ncc_trajectory": [
            item["mean_expected_fourier_ncc"] for item in result.diagnostics
        ],
        "mean_max_responsibility_trajectory": [
            item["mean_max_responsibility"] for item in result.diagnostics
        ],
        "pose_search_trajectory": [
            {key: item[key] for key in pose_search_keys if key in item}
            for item in result.diagnostics
        ],
        "metadata": result.metadata,
    }


def load_inputs(args: argparse.Namespace):
    with mrcfile.mmap(args.stack, permissive=True, mode="r") as mrc:
        images = np.asarray(mrc.data, dtype=np.float32).copy()
        pixel_size = float(mrc.voxel_size.x)
    with mrcfile.mmap(args.references, permissive=True, mode="r") as mrc:
        references = np.asarray(mrc.data, dtype=np.float32).copy()
    with np.load(args.rf_result) as saved:
        assignments = np.asarray(saved["assignments"], dtype=np.int32)
        responsibilities = np.asarray(saved["responsibilities"], dtype=np.float32)
        poses = ai.PoseSet(
            saved["angle_deg"],
            saved["shift_y_px"],
            saved["shift_x_px"],
            saved["mirror"],
        )
        stored_center = (
            str(np.asarray(saved["center_convention"]).item())
            if "center_convention" in saved.files
            else None
        )
    if images.ndim != 3 or references.ndim != 3:
        raise ValueError("images and references must both be three-dimensional stacks.")
    if images.shape[1:] != references.shape[1:]:
        raise ValueError("images and references must have matching image shapes.")
    if len(images) != len(assignments) or len(images) != len(poses):
        raise ValueError("RF result particle count does not match the image stack.")
    if responsibilities.shape != (len(images), len(references)):
        raise ValueError("RF responsibilities do not match image/reference counts.")
    if pixel_size <= 0.0:
        raise ValueError("prepared stack must record a positive MRC voxel size.")
    converted_from_v1_4 = stored_center is None
    if converted_from_v1_4:
        warnings.warn(
            "RF result has no center_convention; treating it as a pre-1.5 "
            "geometric-center result and converting its poses.",
            UserWarning,
            stacklevel=2,
        )
        poses = ai.convert_v1_4_poses_to_integer_center(poses, images.shape[1])
    elif stored_center != CENTER_CONVENTION:
        raise ValueError(
            f"unsupported RF pose center convention {stored_center!r}; "
            f"expected {CENTER_CONVENTION!r}"
        )
    center_migration = {
        "stored_center_convention": stored_center,
        "active_center_convention": CENTER_CONVENTION,
        "converted_from_v1_4_geometric_center": converted_from_v1_4,
    }
    return (
        images,
        references,
        assignments,
        responsibilities,
        poses,
        pixel_size,
        center_migration,
    )


def main(args: argparse.Namespace) -> None:
    if args.minimum_frc_halfset_weight < 0.0:
        raise ValueError("minimum_frc_halfset_weight must be non-negative.")
    if getattr(args, "deterministic_repeats", 1) < 1:
        raise ValueError("deterministic_repeats must be positive.")
    (
        images,
        references,
        assignments,
        responsibilities,
        poses,
        pixel_size,
        center_migration,
    ) = load_inputs(args)
    component_count = len(references)
    config_values = asdict(ai.AlignmentConfig.preset("refine"))
    search_strategy = getattr(args, "search_strategy", config_values["search_strategy"])
    config_values.update(
        search_strategy=search_strategy,
        max_iterations=args.iterations,
        temperature_anneal_iterations=args.anneal_iterations,
        candidate_scoring=getattr(args, "candidate_scoring", "fourier"),
        score_model=getattr(args, "score_model", "fourier_ncc"),
        reference_update=getattr(args, "reference_update", "fourier"),
        batch_size=args.batch_size,
        memory_fraction=args.memory_fraction,
        halfset_diagnostics=True,
        profile_execution=getattr(args, "profile_execution", False),
    )
    if search_strategy == "quadratic_refine":
        config_values.update(
            local_angle_range=7.0,
            coarse_angle_step=1.0,
            local_shift_range=3.0,
            rescue_uncertain_particles=False,
        )
    config = ai.AlignmentConfig(**config_values).normalized(workflow="refine")
    base = args.output.with_suffix("")
    report: dict[str, Any] = {
        "schema": "alignimg.re2dc-70s-feedback-validation.v1",
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "alignimg_version": ai.__version__,
            "backends": ai.available_alignment_backends(),
        },
        "inputs": {
            "stack": str(args.stack),
            "references": str(args.references),
            "rf_result": str(args.rf_result),
            "particle_count": len(images),
            "component_count": component_count,
            "pixel_size_angstrom": pixel_size,
            "pose_center_migration": center_migration,
        },
        "scope_note": (
            "Reassignment is an alignment diagnostic, not biological class accuracy."
        ),
        "parameters": {
            "backend": args.backend,
            "corrective_trust": args.corrective_trust,
            "corrective_prior_source": args.corrective_prior_source,
            "minimum_frc_halfset_weight": args.minimum_frc_halfset_weight,
            "config": asdict(config),
        },
        "runs": {},
    }
    write_json(args.output, report)
    try:
        prior_inputs = {
            "assignments": assignments,
            "n_components": component_count,
        }
        modes = [("fixed", 1.0, prior_inputs)]
        if args.corrective_prior_source == "responsibilities":
            corrective_inputs = {"responsibilities": responsibilities}
        else:
            corrective_inputs = prior_inputs
        modes.append(("corrective", args.corrective_trust, corrective_inputs))
        for mode, trust, feedback in modes:
            print(f"RUN  {mode} feedback refinement", flush=True)
            priors = ai.make_class_priors(**feedback, trust=trust)
            started = time.perf_counter()
            result = ai.refine_alignment(
                images,
                references,
                poses,
                class_priors=priors,
                config=config,
                backend=args.backend,
            )
            seconds = time.perf_counter() - started
            if not np.all(np.isfinite(result.references)):
                raise RuntimeError(f"{mode} refinement produced non-finite references")
            if not np.all(np.isfinite(result.responsibilities)):
                raise RuntimeError(
                    f"{mode} refinement produced non-finite responsibilities"
                )
            if np.max(np.abs(result.responsibilities.sum(axis=1) - 1.0)) > 1e-5:
                raise RuntimeError(f"{mode} responsibilities are not normalized")
            if (
                args.backend in {"cpu", "cuda", "cupy"}
                and result.metadata["backend"] != args.backend
            ):
                raise RuntimeError(f"{mode} refinement unexpectedly changed backend")
            if (
                search_strategy == "quadratic_refine"
                and args.backend == "cuda"
                and result.metadata.get("quadratic_peak_backend") != "native_cuda"
            ):
                raise RuntimeError(
                    f"{mode} refinement did not use native CUDA peak fitting"
                )
            metrics = result_metrics(
                result,
                initial_assignments=assignments,
                initial_poses=poses,
                initial_references=references,
                seconds=seconds,
                minimum_halfset_weight=args.minimum_frc_halfset_weight,
                pixel_size=pixel_size,
            )
            metrics["artifacts"] = save_result(
                base, mode, result, pixel_size=pixel_size
            )
            repeat_metrics = []
            for repeat_index in range(
                1, getattr(args, "deterministic_repeats", 1)
            ):
                repeated = ai.refine_alignment(
                    images,
                    references,
                    poses,
                    class_priors=priors,
                    config=config,
                    backend=args.backend,
                )
                comparison = reproducibility_metrics(result, repeated)
                comparison["repeat_index"] = repeat_index
                repeat_metrics.append(comparison)
            metrics["reproducibility_repeats"] = repeat_metrics
            metrics["reproducible"] = all(
                item["passed"] for item in repeat_metrics
            )
            report["runs"][mode] = metrics
            write_json(args.output, report)
            print(f"DONE {mode}: {seconds:.3f} s", flush=True)
        if report["runs"]["fixed"]["reassignment_fraction"] != 0.0:
            raise RuntimeError("fixed feedback refinement changed a class assignment")
        if not all(run["reproducible"] for run in report["runs"].values()):
            raise RuntimeError("feedback refinement was not reproducible")
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
    write_json(args.output, report)
    print(f"Report saved to: {args.output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", type=Path, default=DEFAULT_STACK)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--rf-result", type=Path, default=DEFAULT_RF_RESULT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation-results/re2dc-70s-feedback.json"),
    )
    parser.add_argument(
        "--backend", choices=("cpu", "cuda", "cupy", "gpu", "auto"), default="cuda"
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--anneal-iterations", type=int, default=3)
    parser.add_argument(
        "--search-strategy",
        choices=("adaptive_posterior", "quadratic_refine"),
        default="adaptive_posterior",
    )
    parser.add_argument(
        "--candidate-scoring", choices=("raster", "fourier"), default="fourier"
    )
    parser.add_argument(
        "--score-model",
        choices=("fourier_ncc", "whitened_fourier_ncc"),
        default="fourier_ncc",
    )
    parser.add_argument(
        "--reference-update", choices=("spatial", "fourier"), default="fourier"
    )
    parser.add_argument("--corrective-trust", type=float, default=0.9)
    parser.add_argument(
        "--corrective-prior-source",
        choices=("assignments", "responsibilities"),
        default="responsibilities",
        help=(
            "Classification feedback used by corrective refinement. Soft RF "
            "responsibilities are recommended; hard assignments are a "
            "conservative fallback."
        ),
    )
    parser.add_argument("--minimum-frc-halfset-weight", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--profile-execution", action="store_true")
    parser.add_argument("--deterministic-repeats", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
