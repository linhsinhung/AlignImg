#!/usr/bin/env python3
"""Align a particle stack to one known reference and save reusable artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import socket
import time
from typing import Any

import mrcfile
import numpy as np

import alignimg as ai


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_stack(path: Path) -> tuple[np.ndarray, float | None]:
    with mrcfile.mmap(path, permissive=True, mode="r") as mrc:
        values = np.asarray(mrc.data, dtype=np.float32)
        pixel_size = float(mrc.voxel_size.x)
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError(f"{path} must contain square 2-D images.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains non-finite values.")
    return values, pixel_size if pixel_size > 0 else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("cpu", "cuda", "cupy", "auto"), default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--global-iterations", type=int, default=1)
    parser.add_argument("--refine-iterations", type=int, default=3)
    parser.add_argument("--angle-samples", type=int, default=256)
    parser.add_argument("--proposal-angles", type=int, default=8)
    parser.add_argument("--translation-range", type=float, default=6.0)
    parser.add_argument("--apply-final-pose-to-raw", action="store_true")
    parser.add_argument("--profile-execution", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_directory
    if args.refine_iterations < 0:
        raise ValueError("refine_iterations must be non-negative.")
    particles, pixel_size = load_stack(args.particles)
    references, _ = load_stack(args.reference)
    if len(references) != 1:
        raise ValueError("reference must contain exactly one image.")
    if particles.shape[1:] != references.shape[1:]:
        raise ValueError("particle and reference image sizes do not match.")
    output.mkdir(parents=True, exist_ok=False)

    global_config = replace(
        ai.AlignmentConfig.preset("global_accurate"),
        max_iterations=args.global_iterations,
        angle_samples=args.angle_samples,
        proposal_angles_per_reference=args.proposal_angles,
        translation_range=args.translation_range,
        batch_size=args.batch_size,
        memory_fraction=args.memory_fraction,
        profile_execution=args.profile_execution,
        halfset_diagnostics=True,
        store_history=True,
        apply_final_pose_to_raw=(
            args.apply_final_pose_to_raw and args.refine_iterations == 0
        ),
    ).normalized(workflow="global")
    started = time.perf_counter()
    global_result = ai.align_to_references(
        particles,
        references,
        config=global_config,
        backend=args.backend,
    )
    global_seconds = time.perf_counter() - started

    final_result = global_result
    refine_seconds = 0.0
    refine_config = None
    if args.refine_iterations > 0:
        refine_config = replace(
            ai.AlignmentConfig.preset("refine"),
            max_iterations=args.refine_iterations,
            batch_size=args.batch_size,
            memory_fraction=args.memory_fraction,
            halfset_diagnostics=True,
            store_history=True,
            apply_final_pose_to_raw=args.apply_final_pose_to_raw,
            profile_execution=args.profile_execution,
        ).normalized(workflow="refine")
        started = time.perf_counter()
        final_result = ai.refine_alignment(
            particles,
            global_result.references,
            global_result.poses,
            config=refine_config,
            backend=args.backend,
        )
        refine_seconds = time.perf_counter() - started

    average_path = output / "class_average.mrc"
    with mrcfile.new(average_path) as mrc:
        mrc.set_data(np.asarray(final_result.class_averages[0], dtype=np.float32))
        if pixel_size is not None:
            mrc.voxel_size = pixel_size

    history = list(global_result.reference_history)
    if final_result is not global_result:
        history.extend(final_result.reference_history[1:])
    result_path = output / "result.npz"
    np.savez_compressed(
        result_path,
        alignimg_version=np.asarray(ai.__version__),
        center_convention=np.asarray(final_result.metadata["center_convention"]),
        reference_history=np.asarray(history, dtype=np.float32),
        soft_references=np.asarray(final_result.references, dtype=np.float32),
        class_averages=np.asarray(final_result.class_averages, dtype=np.float32),
        angle_deg=final_result.poses.angle_deg,
        shift_y_px=final_result.poses.shift_y_px,
        shift_x_px=final_result.poses.shift_x_px,
        mirror=final_result.poses.mirror,
        assignments=final_result.reference_assignments,
        responsibilities=final_result.responsibilities,
        inlier_weights=final_result.inlier_weights,
        pose_entropy=final_result.pose_entropy,
        map_posterior=final_result.map_posterior,
        candidate_angle_deg=final_result.candidates.angle_deg,
        candidate_shift_y_px=final_result.candidates.shift_y_px,
        candidate_shift_x_px=final_result.candidates.shift_x_px,
        candidate_score=final_result.candidates.score,
        candidate_posterior=final_result.candidates.posterior,
        global_angle_deg=global_result.poses.angle_deg,
        global_shift_y_px=global_result.poses.shift_y_px,
        global_shift_x_px=global_result.poses.shift_x_px,
    )

    report = {
        "schema": "alignimg.known-reference.v1",
        "status": "completed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "alignimg_version": ai.__version__,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "backends": ai.available_alignment_backends(),
        },
        "input": {
            "particles": str(args.particles),
            "reference": str(args.reference),
            "particle_count": len(particles),
            "image_shape": list(particles.shape[1:]),
            "pixel_size_angstrom": pixel_size,
        },
        "parameters": {
            "backend": args.backend,
            "global_config": asdict(global_config),
            "refine_config": asdict(refine_config) if refine_config else None,
        },
        "summary": {
            "global_seconds": global_seconds,
            "refine_seconds": refine_seconds,
            "total_seconds": global_seconds + refine_seconds,
            "mean_inlier_weight": float(np.mean(final_result.inlier_weights)),
            "mean_pose_entropy": float(np.mean(final_result.pose_entropy)),
            "mean_map_posterior": float(np.mean(final_result.map_posterior)),
            "maximum_absolute_shift_px": float(
                np.max(
                    np.hypot(
                        final_result.poses.shift_y_px,
                        final_result.poses.shift_x_px,
                    )
                )
            ),
            "assignment_values": np.unique(final_result.reference_assignments),
            "responsibility_sum_max_error": float(
                np.max(np.abs(final_result.responsibilities.sum(axis=1) - 1.0))
            ),
            "class_average_estimator": final_result.metadata["class_average_estimator"],
        },
        "global": {
            "diagnostics": global_result.diagnostics,
            "metadata": global_result.metadata,
        },
        "refine": {
            "diagnostics": (
                final_result.diagnostics if final_result is not global_result else []
            ),
            "metadata": (
                final_result.metadata if final_result is not global_result else None
            ),
        },
        "artifacts": {
            "class_average": average_path.name,
            "result": result_path.name,
        },
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stages = "global alignment"
    if args.refine_iterations > 0:
        stages += " followed by continuous quadratic robust refinement"
    print(f"RUN  {stages}", flush=True)
    report = run(args)
    print(json.dumps(jsonable(report["summary"]), sort_keys=True), flush=True)
    print(f"Report saved to: {args.output_directory / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
