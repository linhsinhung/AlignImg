#!/usr/bin/env python3
"""Validate the 2.2 refine preset on the 3,050-particle K=1 user workflow."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
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

try:
    from tools.re2dc_70s_feedback_validation import jsonable
except ModuleNotFoundError:
    from re2dc_70s_feedback_validation import jsonable


EXPECTED_VERSION = "2.2.0"
DEFAULT_PARTICLES = Path("data/local/test_align.mrcs")
DEFAULT_REFERENCE = Path("data/local/mu_aligned_mean.mrc")
DEFAULT_OUTPUT = Path(
    "validation-results/performance/quadratic-refine/dev5-cuda-local-n3050-ab.json"
)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_images(path: Path) -> tuple[np.ndarray, float | None]:
    with mrcfile.mmap(path, permissive=True, mode="r") as mrc:
        values = np.asarray(mrc.data, dtype=np.float32).copy()
        pixel_size = float(mrc.voxel_size.x)
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError(f"{path} must contain square 2-D images")
    if values.shape[-1] % 2:
        raise ValueError("AlignImg 2.x requires an even image size")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains non-finite values")
    return values, pixel_size if pixel_size > 0.0 else None


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def pose_delta(first: ai.PoseSet, second: ai.PoseSet) -> dict[str, Any]:
    angle = np.abs((second.angle_deg - first.angle_deg + 180.0) % 360.0 - 180.0)
    shift = np.hypot(
        second.shift_y_px - first.shift_y_px,
        second.shift_x_px - first.shift_x_px,
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
    }


def refine_configs(
    *,
    iterations: int,
    batch_size: int,
    memory_fraction: float,
    profile_execution: bool,
) -> dict[str, ai.AlignmentConfig]:
    quadratic = replace(
        ai.AlignmentConfig.preset("refine"),
        max_iterations=iterations,
        batch_size=batch_size,
        memory_fraction=memory_fraction,
        apply_final_pose_to_raw=True,
        halfset_diagnostics=True,
        profile_execution=profile_execution,
    ).normalized(workflow="refine")
    adaptive_values = asdict(quadratic)
    adaptive_values.update(
        search_strategy="adaptive_posterior",
        local_angle_range=15.0,
        coarse_angle_step=6.0,
        local_shift_range=3.0,
        coarse_shift_step=1.0,
        adaptive_fraction=0.999,
        oversampling_order=1,
        max_adaptive_cells=None,
        rescue_uncertain_particles=False,
    )
    return {
        "adaptive_posterior": ai.AlignmentConfig(**adaptive_values).normalized(
            workflow="refine"
        ),
        "quadratic_refine": quadratic,
    }


def environment(backend: str) -> dict[str, Any]:
    if ai.__version__ != EXPECTED_VERSION:
        raise RuntimeError(
            f"AlignImg module is {ai.__version__}, expected {EXPECTED_VERSION}"
        )
    gpu = None
    native = None
    if backend != "cpu":
        import alignimg_gpu

        if alignimg_gpu.__version__ != EXPECTED_VERSION:
            raise RuntimeError("alignimg-gpu version does not match AlignImg")
        gpu = alignimg_gpu.__version__
        if backend == "cuda":
            from alignimg_gpu.backend import _native_module

            module = _native_module()
            if module is None or module.__version__ != EXPECTED_VERSION:
                raise RuntimeError("native CUDA extension version does not match")
            native = module.__version__
    available = ai.available_alignment_backends()
    if not available.get(backend, {}).get("available", False):
        raise RuntimeError(f"backend is unavailable: {backend}")
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "alignimg_version": ai.__version__,
        "alignimg_gpu_version": gpu,
        "native_cuda_version": native,
        "backends": available,
    }


def save_result(
    base: Path,
    strategy: str,
    result: ai.AlignmentResult,
    *,
    pixel_size: float | None,
) -> dict[str, str]:
    result_path = Path(f"{base}.{strategy}.result.npz")
    average_path = Path(f"{base}.{strategy}.class-average.mrc")
    reference_path = Path(f"{base}.{strategy}.soft-reference.mrc")
    for path, values in (
        (average_path, result.class_averages[0]),
        (reference_path, result.references[0]),
    ):
        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(np.asarray(values, dtype=np.float32))
            if pixel_size is not None:
                mrc.voxel_size = pixel_size
            mrc.update_header_stats()
    np.savez_compressed(
        result_path,
        alignimg_version=np.asarray(ai.__version__),
        center_convention=np.asarray(result.metadata["center_convention"]),
        reference_history=np.asarray(result.reference_history, dtype=np.float32),
        soft_references=result.references,
        class_averages=result.class_averages,
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
    return {
        "result": str(result_path),
        "class_average": str(average_path),
        "soft_reference": str(reference_path),
    }


def result_summary(
    result: ai.AlignmentResult,
    *,
    seconds: float,
    initial_poses: ai.PoseSet,
    supplied_reference: np.ndarray,
    requested_backend: str,
) -> dict[str, Any]:
    for name, values in (
        ("references", result.references),
        ("class averages", result.class_averages),
        ("responsibilities", result.responsibilities),
        ("inlier weights", result.inlier_weights),
    ):
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"{name} contain non-finite values")
    if result.metadata.get("backend") != requested_backend:
        raise RuntimeError("refinement changed the requested backend")
    if np.any(result.reference_assignments != 0):
        raise RuntimeError("K=1 refinement produced a nonzero assignment")
    row_error = float(np.max(np.abs(result.responsibilities.sum(axis=1) - 1.0)))
    if row_error > 1e-5:
        raise RuntimeError("responsibilities are not normalized")
    memory_plans = result.metadata.get("gpu_memory_plans", [])
    return {
        "seconds": seconds,
        "particle_iterations_per_second": (
            len(result.poses) * len(result.diagnostics) / max(seconds, 1e-12)
        ),
        "iteration_seconds": [item["seconds"] for item in result.diagnostics],
        "mean_expected_fourier_ncc_trajectory": [
            item["mean_expected_fourier_ncc"] for item in result.diagnostics
        ],
        "reference_relative_change_trajectory": [
            item["reference_relative_change"] for item in result.diagnostics
        ],
        "responsibility_sum_max_error": row_error,
        "pose_delta_from_global": pose_delta(initial_poses, result.poses),
        "soft_reference_correlation_to_supplied": correlation(
            result.references[0], supplied_reference
        ),
        "raw_class_average_correlation_to_supplied": correlation(
            result.class_averages[0], supplied_reference
        ),
        "soft_reference_correlation_to_raw_average": correlation(
            result.references[0], result.class_averages[0]
        ),
        "class_average_estimator": result.metadata["class_average_estimator"],
        "memory_plan_count": len(memory_plans),
        "map_chunk_sizes": sorted(
            {
                int(plan["map_chunk_size"])
                for plan in memory_plans
                if plan.get("map_chunk_size") is not None
            }
        ),
        "metadata": result.metadata,
    }


def acceptance_gate(
    adaptive: dict[str, Any],
    quadratic: dict[str, Any],
    *,
    backend: str,
) -> dict[str, Any]:
    checks = {
        "quadratic_not_more_than_5_percent_slower": (
            quadratic["seconds"] <= 1.05 * adaptive["seconds"]
        ),
        "soft_reference_coherence_with_raw_average_within_0_005": (
            quadratic["soft_reference_correlation_to_raw_average"]
            >= adaptive["soft_reference_correlation_to_raw_average"] - 0.005
        ),
        "raw_average_correlation_within_0_005": (
            quadratic["raw_class_average_correlation_to_supplied"]
            >= adaptive["raw_class_average_correlation_to_supplied"] - 0.005
        ),
        "raw_average_estimator_used": (
            quadratic["class_average_estimator"] == "final_map_pose_inlier_weighted_raw"
        ),
        "responsibilities_normalized": (
            quadratic["responsibility_sum_max_error"] <= 1e-5
        ),
        "requested_backend_retained": (quadratic["metadata"].get("backend") == backend),
        "no_gpu_fallback": (
            backend == "cpu" or quadratic["metadata"].get("gpu_fallback_reason") is None
        ),
        "quadratic_peak_backend_correct": (
            backend == "cpu"
            or quadratic["metadata"].get("quadratic_peak_backend")
            == ("native_cuda" if backend == "cuda" else "cupy")
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "speedup_adaptive_over_quadratic": (
            adaptive["seconds"] / max(quadratic["seconds"], 1e-12)
        ),
        "observations": {
            "soft_reference_correlation_delta": (
                quadratic["soft_reference_correlation_to_supplied"]
                - adaptive["soft_reference_correlation_to_supplied"]
            ),
            "raw_average_correlation_delta": (
                quadratic["raw_class_average_correlation_to_supplied"]
                - adaptive["raw_class_average_correlation_to_supplied"]
            ),
            "soft_reference_to_raw_average_correlation_delta": (
                quadratic["soft_reference_correlation_to_raw_average"]
                - adaptive["soft_reference_correlation_to_raw_average"]
            ),
        },
    }


def main(args: argparse.Namespace) -> int:
    if args.iterations != 2:
        raise ValueError("Stage 5 freezes exactly two refine iterations")
    report: dict[str, Any] = {
        "schema": "alignimg.quadratic-stage5-local-ab.v1",
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.output, report)
    try:
        report["environment"] = environment(args.backend)
        images, pixel_size = load_images(args.particles)
        references, _ = load_images(args.reference)
        if len(images) != 3050 or images.shape[1:] != (100, 100):
            raise ValueError("Stage 5 requires the frozen 3050x100x100 particle stack")
        if references.shape != (1, 100, 100):
            raise ValueError("Stage 5 requires one 100x100 reference")
        configs = refine_configs(
            iterations=args.iterations,
            batch_size=args.batch_size,
            memory_fraction=args.memory_fraction,
            profile_execution=args.profile_execution,
        )
        global_config = replace(
            ai.AlignmentConfig.preset("global_accurate"),
            max_iterations=1,
            angle_samples=256,
            proposal_angles_per_reference=8,
            translation_range=6.0,
            batch_size=args.batch_size,
            memory_fraction=args.memory_fraction,
            halfset_diagnostics=True,
            profile_execution=args.profile_execution,
        ).normalized(workflow="global")
        report.update(
            inputs={
                "particles": str(args.particles),
                "reference": str(args.reference),
                "particle_count": len(images),
                "image_shape": list(images.shape[1:]),
                "pixel_size_angstrom": pixel_size,
                "sha256": {
                    "particles": sha256(args.particles),
                    "reference": sha256(args.reference),
                },
            },
            parameters={
                "backend": args.backend,
                "global_config": asdict(global_config),
                "refine_configs": {
                    name: asdict(config) for name, config in configs.items()
                },
            },
            runs={},
        )
        write_json(args.output, report)

        print("RUN  shared global initializer", flush=True)
        started = time.perf_counter()
        global_result = ai.align_to_references(
            images, references, config=global_config, backend=args.backend
        )
        global_seconds = time.perf_counter() - started
        input_path = Path(f"{args.output.with_suffix('')}.inputs.npz")
        np.savez_compressed(
            input_path,
            alignimg_version=np.asarray(ai.__version__),
            references=global_result.references,
            angle_deg=global_result.poses.angle_deg,
            shift_y_px=global_result.poses.shift_y_px,
            shift_x_px=global_result.poses.shift_x_px,
            mirror=global_result.poses.mirror,
        )
        report["global_initializer"] = {
            "seconds": global_seconds,
            "reference_correlation_to_supplied": correlation(
                global_result.references[0], references[0]
            ),
            "metadata": global_result.metadata,
            "artifact": str(input_path),
            "artifact_sha256": sha256(input_path),
        }
        write_json(args.output, report)

        results = {}
        results_by_strategy: dict[str, ai.AlignmentResult] = {}
        base = args.output.with_suffix("")
        for strategy, config in configs.items():
            print(f"RUN  {strategy} K=1 refine", flush=True)
            started = time.perf_counter()
            result = ai.refine_alignment(
                images,
                global_result.references,
                global_result.poses,
                config=config,
                backend=args.backend,
            )
            seconds = time.perf_counter() - started
            summary = result_summary(
                result,
                seconds=seconds,
                initial_poses=global_result.poses,
                supplied_reference=references[0],
                requested_backend=args.backend,
            )
            summary["artifacts"] = save_result(
                base, strategy, result, pixel_size=pixel_size
            )
            summary["artifact_sha256"] = {
                name: sha256(Path(path)) for name, path in summary["artifacts"].items()
            }
            results[strategy] = summary
            results_by_strategy[strategy] = result
            report["runs"] = results
            write_json(args.output, report)

        adaptive = results["adaptive_posterior"]
        quadratic = results["quadratic_refine"]
        report["comparison"] = {
            "pose_delta_adaptive_to_quadratic": pose_delta(
                results_by_strategy["adaptive_posterior"].poses,
                results_by_strategy["quadratic_refine"].poses,
            ),
            "class_average_cross_correlation": correlation(
                results_by_strategy["adaptive_posterior"].class_averages[0],
                results_by_strategy["quadratic_refine"].class_averages[0],
            ),
            "soft_reference_cross_correlation": correlation(
                results_by_strategy["adaptive_posterior"].references[0],
                results_by_strategy["quadratic_refine"].references[0],
            ),
        }
        report["gate"] = acceptance_gate(adaptive, quadratic, backend=args.backend)
        report["status"] = (
            "completed" if report["gate"]["passed"] else "completed_with_failed_gates"
        )
        write_json(args.output, report)
        print(f"Report saved to: {args.output.resolve()}", flush=True)
        return 0 if report["gate"]["passed"] else 1
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(args.output, report)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=Path, default=DEFAULT_PARTICLES)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--backend", choices=("cpu", "cuda", "cupy"), default="cuda")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--profile-execution", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
