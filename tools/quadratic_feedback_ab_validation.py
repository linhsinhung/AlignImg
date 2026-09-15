#!/usr/bin/env python3
"""Close Stage 4 with a fixed-feedback adaptive/quadratic RELION pose A/B."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
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

try:
    from tools.prepare_re2dc_70s_pose_benchmark import relion_to_alignimg_poses
    from tools.re2dc_70s_feedback_validation import (
        DEFAULT_REFERENCES,
        DEFAULT_RF_RESULT,
        DEFAULT_STACK,
        load_inputs,
        result_metrics,
        save_result,
        write_json,
    )
    from tools.re2dc_70s_pose_benchmark import (
        compose_pose_gauge,
        fit_pose_gauge,
        image_correlation,
        subset_poses,
        wrapped_angle,
    )
    from tools.re2dc_70s_relion_benchmark import (
        DEFAULT_RELION_STAR,
        align_relion_metadata,
        optimal_label_mapping,
    )
except ModuleNotFoundError:
    from prepare_re2dc_70s_pose_benchmark import relion_to_alignimg_poses
    from re2dc_70s_feedback_validation import (
        DEFAULT_REFERENCES,
        DEFAULT_RF_RESULT,
        DEFAULT_STACK,
        load_inputs,
        result_metrics,
        save_result,
        write_json,
    )
    from re2dc_70s_pose_benchmark import (
        compose_pose_gauge,
        fit_pose_gauge,
        image_correlation,
        subset_poses,
        wrapped_angle,
    )
    from re2dc_70s_relion_benchmark import (
        DEFAULT_RELION_STAR,
        align_relion_metadata,
        optimal_label_mapping,
    )


DEFAULT_PREPARED_STAR = Path(
    "data/re2dc_70s_testdata/prepared/re2dc_70s_n1000_s128.star"
)
DEFAULT_QUADRATIC_REPORT = Path(
    "validation-results/performance/quadratic-refine/dev4-cuda-feedback-i5.json"
)
DEFAULT_QUADRATIC_RESULT = DEFAULT_QUADRATIC_REPORT.with_suffix(".fixed.result.npz")
DEFAULT_QUADRATIC_REFERENCES = DEFAULT_QUADRATIC_REPORT.with_suffix(
    ".fixed.references.mrcs"
)
DEFAULT_OUTPUT = Path(
    "validation-results/performance/quadratic-refine/"
    "dev4-cuda-feedback-scientific-ab.json"
)
RELION_ANGLE_HALF_STEP_DEG = 2.8125


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float]:
    result = np.quantile(np.asarray(values, dtype=np.float64), (0.5, 0.9, 0.95, 1.0))
    return dict(zip(("median", "p90", "p95", "maximum"), result, strict=True))


def load_saved_alignment(
    result_path: Path,
    reference_path: Path,
    *,
    particle_count: int,
    component_count: int,
) -> dict[str, Any]:
    with np.load(result_path) as saved:
        stored_center = str(np.asarray(saved["center_convention"]).item())
        if stored_center != CENTER_CONVENTION:
            raise ValueError(
                f"quadratic result uses {stored_center!r}, expected {CENTER_CONVENTION!r}"
            )
        poses = ai.PoseSet(
            saved["angle_deg"],
            saved["shift_y_px"],
            saved["shift_x_px"],
            saved["mirror"],
        )
        assignments = np.asarray(saved["assignments"], dtype=np.int32)
        responsibilities = np.asarray(saved["responsibilities"], dtype=np.float32)
        stored_version = str(np.asarray(saved["alignimg_version"]).item())
    with mrcfile.mmap(reference_path, permissive=True, mode="r") as mrc:
        references = np.asarray(mrc.data, dtype=np.float32).copy()
    if len(poses) != particle_count or assignments.shape != (particle_count,):
        raise ValueError("quadratic result particle count does not match the stack")
    if responsibilities.shape != (particle_count, component_count):
        raise ValueError("quadratic responsibilities have the wrong shape")
    if references.shape[0] != component_count:
        raise ValueError("quadratic references have the wrong component count")
    for name, values in (
        ("angle", poses.angle_deg),
        ("shift_y", poses.shift_y_px),
        ("shift_x", poses.shift_x_px),
        ("responsibilities", responsibilities),
        ("references", references),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"quadratic {name} contains non-finite values")
    if np.max(np.abs(responsibilities.sum(axis=1) - 1.0)) > 1e-5:
        raise ValueError("quadratic responsibilities are not normalized")
    return {
        "version": stored_version,
        "poses": poses,
        "assignments": assignments,
        "responsibilities": responsibilities,
        "references": references,
    }


def make_relion_subsets(
    relion_class: np.ndarray,
    confidence: np.ndarray,
    assignments: np.ndarray,
    *,
    component_count: int,
    minimum_confidence: float,
    minimum_component_particles: int,
) -> tuple[list[tuple[int, np.ndarray]], dict[str, Any]]:
    mapping, agreement, _, _, _ = optimal_label_mapping(relion_class, assignments)
    subsets = []
    counts = np.zeros(component_count, dtype=np.int64)
    for component in range(component_count):
        mapped_class = mapping.get(component)
        if mapped_class is None:
            continue
        selected = np.flatnonzero(
            (assignments == component)
            & (relion_class == mapped_class)
            & (confidence >= minimum_confidence)
        )
        counts[component] = len(selected)
        if len(selected) >= minimum_component_particles:
            subsets.append((component, selected))
    if not subsets:
        raise ValueError("no component passed the RELION subset filters")
    return subsets, {
        "optimal_alignimg_to_relion_class_mapping": mapping,
        "full_stack_optimal_mapping_agreement": agreement,
        "minimum_relion_confidence": minimum_confidence,
        "minimum_component_particles": minimum_component_particles,
        "matched_confident_count_by_component": counts,
        "included_components": [component for component, _ in subsets],
        "included_particle_count": int(sum(len(selected) for _, selected in subsets)),
    }


def alignment_accuracy(
    poses: ai.PoseSet,
    references: np.ndarray,
    images: np.ndarray,
    relion_poses: ai.PoseSet,
    subsets: list[tuple[int, np.ndarray]],
) -> dict[str, Any]:
    angle_chunks = []
    shift_chunks = []
    per_component = []
    for component, selected in subsets:
        predicted = subset_poses(poses, selected)
        truth = subset_poses(relion_poses, selected)
        gauge = fit_pose_gauge(predicted, truth)
        expected = compose_pose_gauge(truth, gauge)
        angle = np.abs(wrapped_angle(predicted.angle_deg - expected.angle_deg))
        shift = np.hypot(
            predicted.shift_y_px - expected.shift_y_px,
            predicted.shift_x_px - expected.shift_x_px,
        )
        oracle = np.mean(
            ai.transform_images(images[selected], truth, backend="cpu"), axis=0
        )
        gauged_oracle = ai.transform_images(oracle[None], gauge, backend="cpu")[0]
        pose_applied_average = np.mean(
            ai.transform_images(images[selected], predicted, backend="cpu"), axis=0
        )
        angle_chunks.append(angle)
        shift_chunks.append(shift)
        per_component.append(
            {
                "component": component,
                "particle_count": len(selected),
                "gauge": {
                    "angle_deg": float(gauge.angle_deg[0]),
                    "shift_y_px": float(gauge.shift_y_px[0]),
                    "shift_x_px": float(gauge.shift_x_px[0]),
                },
                "angle_error_deg": quantiles(angle),
                "shift_error_px": quantiles(shift),
                "mstep_reference_correlation": image_correlation(
                    references[component], gauged_oracle
                ),
                "pose_applied_average_correlation": image_correlation(
                    pose_applied_average, gauged_oracle
                ),
            }
        )
    return {
        "particle_count": int(sum(len(values) for _, values in subsets)),
        "component_count": len(subsets),
        "angle_error_deg": quantiles(np.concatenate(angle_chunks)),
        "shift_error_px": quantiles(np.concatenate(shift_chunks)),
        "per_component": per_component,
    }


def scientific_gate(
    initial: dict[str, Any],
    adaptive: dict[str, Any],
    quadratic: dict[str, Any],
    *,
    angle_half_step_deg: float = RELION_ANGLE_HALF_STEP_DEG,
) -> dict[str, Any]:
    initial_by_component = {
        item["component"]: item for item in initial["per_component"]
    }
    adaptive_by_component = {
        item["component"]: item for item in adaptive["per_component"]
    }
    quadratic_by_component = {
        item["component"]: item for item in quadratic["per_component"]
    }
    components = sorted(quadratic_by_component)
    checks = {
        "each_component_angle_improves_from_input": all(
            quadratic_by_component[index]["angle_error_deg"]["median"]
            < initial_by_component[index]["angle_error_deg"]["median"]
            for index in components
        ),
        "each_component_shift_improves_from_input": all(
            quadratic_by_component[index]["shift_error_px"]["median"]
            < initial_by_component[index]["shift_error_px"]["median"]
            for index in components
        ),
        "aggregate_angle_within_relion_half_step_of_adaptive": (
            quadratic["angle_error_deg"]["median"]
            <= adaptive["angle_error_deg"]["median"] + angle_half_step_deg
        ),
        "aggregate_shift_strictly_better_than_adaptive": (
            quadratic["shift_error_px"]["median"] < adaptive["shift_error_px"]["median"]
        ),
        "angle_p95_not_over_10_percent_worse": (
            quadratic["angle_error_deg"]["p95"]
            <= 1.1 * adaptive["angle_error_deg"]["p95"]
        ),
        "shift_p95_not_over_10_percent_worse": (
            quadratic["shift_error_px"]["p95"]
            <= 1.1 * adaptive["shift_error_px"]["p95"]
        ),
        "each_pose_applied_average_correlation_within_0_005": all(
            quadratic_by_component[index]["pose_applied_average_correlation"]
            >= adaptive_by_component[index]["pose_applied_average_correlation"] - 0.005
            for index in components
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observations_not_used_as_gates": {
            "quadratic_angle_strictly_better_than_adaptive": (
                quadratic["angle_error_deg"]["median"]
                < adaptive["angle_error_deg"]["median"]
            ),
            "mstep_reference_correlation_delta_by_component": {
                index: (
                    quadratic_by_component[index]["mstep_reference_correlation"]
                    - adaptive_by_component[index]["mstep_reference_correlation"]
                )
                for index in components
            },
            "mstep_reference_comparison_is_confounded_by_robust_weight_scope": (
                "adaptive uses global robust weighting; quadratic fixed feedback "
                "uses per-reference robust weighting to preserve K=N/K=1 equivalence"
            ),
        },
        "relion_angle_half_step_deg": angle_half_step_deg,
    }


def adaptive_config_from_quadratic_report(
    quadratic_report: dict[str, Any],
    *,
    batch_size: int,
    memory_fraction: float,
    profile_execution: bool,
) -> ai.AlignmentConfig:
    config = dict(quadratic_report["parameters"]["config"])
    if config.get("search_strategy") != "quadratic_refine":
        raise ValueError("comparison report is not a quadratic-refine run")
    config.update(
        search_strategy="adaptive_posterior",
        local_angle_range=15.0,
        coarse_angle_step=6.0,
        local_shift_range=3.0,
        coarse_shift_step=1.0,
        adaptive_fraction=0.999,
        oversampling_order=1,
        max_adaptive_cells=None,
        rescue_uncertain_particles=False,
        batch_size=batch_size,
        memory_fraction=memory_fraction,
        profile_execution=profile_execution,
    )
    return ai.AlignmentConfig(**config).normalized(workflow="refine")


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.minimum_relion_confidence <= 1.0:
        raise ValueError("minimum_relion_confidence must be between zero and one")
    if args.minimum_component_particles < 1:
        raise ValueError("minimum_component_particles must be positive")


def main(args: argparse.Namespace) -> int:
    validate_args(args)
    reused_adaptive_report = (
        json.loads(args.reuse_adaptive_report.read_text("utf-8"))
        if args.reuse_adaptive_report is not None
        else None
    )
    report: dict[str, Any] = {
        "schema": "alignimg.quadratic-feedback-scientific-ab.v1",
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "alignimg_version": ai.__version__,
            "backends": ai.available_alignment_backends(),
        },
    }
    write_json(args.output, report)
    try:
        (
            images,
            input_references,
            input_assignments,
            _,
            input_poses,
            pixel_size,
            center_migration,
        ) = load_inputs(args)
        quadratic_report = json.loads(args.quadratic_report.read_text("utf-8"))
        if quadratic_report.get("status") != "completed":
            raise ValueError("quadratic feedback report is not completed")
        quadratic = load_saved_alignment(
            args.quadratic_result,
            args.quadratic_references,
            particle_count=len(images),
            component_count=len(input_references),
        )
        if not np.array_equal(quadratic["assignments"], input_assignments):
            raise ValueError("quadratic fixed result changed class assignments")

        metadata = align_relion_metadata(args.relion_star, args.prepared_star)
        required = (
            metadata["max_value_probability"],
            metadata["angle_psi_deg"],
            metadata["origin_x_angstrom"],
            metadata["origin_y_angstrom"],
        )
        if any(values is None for values in required):
            raise ValueError("RELION STAR is missing confidence or pose fields")
        relion_class = np.asarray(metadata["class_number"], dtype=np.int64)
        confidence = np.asarray(metadata["max_value_probability"], dtype=np.float64)
        relion_poses = relion_to_alignimg_poses(
            np.asarray(metadata["angle_psi_deg"]),
            np.asarray(metadata["origin_x_angstrom"]),
            np.asarray(metadata["origin_y_angstrom"]),
            pixel_size,
        )
        subsets, subset_report = make_relion_subsets(
            relion_class,
            confidence,
            input_assignments,
            component_count=len(input_references),
            minimum_confidence=args.minimum_relion_confidence,
            minimum_component_particles=args.minimum_component_particles,
        )

        config = adaptive_config_from_quadratic_report(
            quadratic_report,
            batch_size=args.batch_size,
            memory_fraction=args.memory_fraction,
            profile_execution=args.profile_execution,
        )
        report.update(
            inputs={
                "stack": str(args.stack),
                "references": str(args.references),
                "rf_result": str(args.rf_result),
                "prepared_star": str(args.prepared_star),
                "relion_star": str(args.relion_star),
                "quadratic_report": str(args.quadratic_report),
                "quadratic_result": str(args.quadratic_result),
                "quadratic_references": str(args.quadratic_references),
                "particle_count": len(images),
                "component_count": len(input_references),
                "pixel_size_angstrom": pixel_size,
                "pose_center_migration": center_migration,
                "sha256": {
                    "stack": file_sha256(args.stack),
                    "references": file_sha256(args.references),
                    "rf_result": file_sha256(args.rf_result),
                    "quadratic_result": file_sha256(args.quadratic_result),
                    "quadratic_references": file_sha256(args.quadratic_references),
                },
            },
            subset=subset_report,
            parameters={
                "backend": args.backend,
                "adaptive_config": asdict(config),
                "quadratic_config": quadratic_report["parameters"]["config"],
            },
        )
        write_json(args.output, report)

        if reused_adaptive_report is None:
            print("RUN  adaptive fixed feedback baseline", flush=True)
            priors = ai.make_class_priors(
                assignments=input_assignments,
                n_components=len(input_references),
                trust=1.0,
            )
            started = time.perf_counter()
            adaptive_result = ai.refine_alignment(
                images,
                input_references,
                input_poses,
                class_priors=priors,
                config=config,
                backend=args.backend,
            )
            seconds = time.perf_counter() - started
            if adaptive_result.metadata.get("backend") != args.backend:
                raise RuntimeError("adaptive fixed run changed backend")
            if not np.array_equal(
                adaptive_result.reference_assignments, input_assignments
            ):
                raise RuntimeError("adaptive fixed run changed class assignments")
            if not np.all(np.isfinite(adaptive_result.references)) or not np.all(
                np.isfinite(adaptive_result.responsibilities)
            ):
                raise RuntimeError("adaptive fixed run produced non-finite values")
            base = args.output.with_suffix("")
            adaptive_metrics = result_metrics(
                adaptive_result,
                initial_assignments=input_assignments,
                initial_poses=input_poses,
                initial_references=input_references,
                seconds=seconds,
                minimum_halfset_weight=10.0,
                pixel_size=pixel_size,
            )
            adaptive_metrics["artifacts"] = save_result(
                base, "adaptive-fixed", adaptive_result, pixel_size=pixel_size
            )
            adaptive = {
                "poses": adaptive_result.poses,
                "references": adaptive_result.references,
            }
            adaptive_source = "executed"
        else:
            if reused_adaptive_report.get("status") not in {
                "completed",
                "completed_with_failed_gates",
            }:
                raise ValueError("reused adaptive report did not complete")
            if reused_adaptive_report["parameters"]["adaptive_config"] != asdict(
                config
            ):
                raise ValueError("reused adaptive config does not match this A/B")
            previous_hashes = reused_adaptive_report["inputs"].get("sha256", {})
            for name, digest in report["inputs"]["sha256"].items():
                if name in previous_hashes and previous_hashes[name] != digest:
                    raise ValueError(f"reused adaptive input hash changed for {name}")
            adaptive_metrics = reused_adaptive_report["runs"]["adaptive_fixed"]
            artifacts = adaptive_metrics["artifacts"]
            adaptive = load_saved_alignment(
                Path(artifacts["result"]),
                Path(artifacts["references"]),
                particle_count=len(images),
                component_count=len(input_references),
            )
            if not np.array_equal(adaptive["assignments"], input_assignments):
                raise ValueError("reused adaptive fixed result changed assignments")
            report["inputs"]["sha256"].update(
                adaptive_result=file_sha256(Path(artifacts["result"])),
                adaptive_references=file_sha256(Path(artifacts["references"])),
            )
            adaptive_source = f"reused:{args.reuse_adaptive_report}"

        print("EVAL RELION-compatible fixed-feedback subset", flush=True)
        accuracy = {
            "input": alignment_accuracy(
                input_poses,
                input_references,
                images,
                relion_poses,
                subsets,
            ),
            "adaptive_posterior": alignment_accuracy(
                adaptive["poses"],
                adaptive["references"],
                images,
                relion_poses,
                subsets,
            ),
            "quadratic_refine": alignment_accuracy(
                quadratic["poses"],
                quadratic["references"],
                images,
                relion_poses,
                subsets,
            ),
        }
        report["runs"] = {
            "adaptive_fixed": adaptive_metrics,
            "quadratic_fixed_existing": quadratic_report["runs"]["fixed"],
            "quadratic_corrective_existing": quadratic_report["runs"]["corrective"],
        }
        report["adaptive_source"] = adaptive_source
        report["relion_pose_accuracy"] = accuracy
        report["gate"] = scientific_gate(
            accuracy["input"],
            accuracy["adaptive_posterior"],
            accuracy["quadratic_refine"],
        )
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
    parser.add_argument("--stack", type=Path, default=DEFAULT_STACK)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--rf-result", type=Path, default=DEFAULT_RF_RESULT)
    parser.add_argument("--prepared-star", type=Path, default=DEFAULT_PREPARED_STAR)
    parser.add_argument("--relion-star", type=Path, default=DEFAULT_RELION_STAR)
    parser.add_argument(
        "--quadratic-report", type=Path, default=DEFAULT_QUADRATIC_REPORT
    )
    parser.add_argument(
        "--quadratic-result", type=Path, default=DEFAULT_QUADRATIC_RESULT
    )
    parser.add_argument(
        "--quadratic-references", type=Path, default=DEFAULT_QUADRATIC_REFERENCES
    )
    parser.add_argument("--backend", choices=("cuda", "cupy"), default="cuda")
    parser.add_argument("--minimum-relion-confidence", type=float, default=0.5)
    parser.add_argument("--minimum-component-particles", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--profile-execution", action="store_true")
    parser.add_argument(
        "--reuse-adaptive-report",
        type=Path,
        help="Reuse adaptive artifacts from an earlier completed A/B report.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
