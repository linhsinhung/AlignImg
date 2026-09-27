#!/usr/bin/env python3
"""Compare 12-iteration fast/precise schedules on the frozen 3,050-particle K=1 workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import traceback
from typing import Any

import numpy as np

import alignimg as ai

try:
    from tools.fast_hard_validation import (
        git_revision,
        runtime_identity,
        sha256,
        source_manifest,
    )
    from tools.fast_to_precise_validation import (
        compare_schedules,
        run_schedule,
        schedule_configs,
    )
    from tools.quadratic_stage5_validation import correlation, load_images, pose_delta
    from tools.server_validation import environment_info, utc_now, write_report
except ModuleNotFoundError:
    from fast_hard_validation import (
        git_revision,
        runtime_identity,
        sha256,
        source_manifest,
    )
    from fast_to_precise_validation import (
        compare_schedules,
        run_schedule,
        schedule_configs,
    )
    from quadratic_stage5_validation import correlation, load_images, pose_delta
    from server_validation import environment_info, utc_now, write_report


SCHEMA = "alignimg.fast-hard-known-reference-validation.v1"
DEFAULT_PARTICLES = Path("data/local/test_align.mrcs")
DEFAULT_REFERENCE = Path("data/local/mu_aligned_mean.mrc")
DEFAULT_ACCEPTED_BASELINE = Path(
    "validation-results/performance/quadratic-refine/"
    "dev5-cuda-local-n3050-ab.quadratic_refine.result.npz"
)
EXPECTED_SHA256 = {
    "particles": "228978d659e01b7a8d30fb05e6d84a158dc9c7ef0d6f35fcbae5b4f3e4cec253",
    "reference": "7e70a74660bd5e2e6f67227b8a70bd90092d11bf2a70169d9aee505b0664dc72",
    "accepted_baseline": (
        "8f6835e5dd7374063943b3f67ae36278d09ec637cf530fba7956563f3a893e48"
    ),
}
ACCEPTED_RAW_CORRELATION = 0.9543693463625517
RAW_CORRELATION_TOLERANCE = 0.005


def _poses(values: dict[str, np.ndarray]) -> ai.PoseSet:
    return ai.PoseSet(
        values["angle_deg"],
        values["shift_y_px"],
        values["shift_x_px"],
        values["mirror"],
    )


def load_accepted_baseline(path: Path) -> dict[str, np.ndarray]:
    if sha256(path) != EXPECTED_SHA256["accepted_baseline"]:
        raise ValueError("accepted 2.2 baseline SHA-256 does not match")
    with np.load(path, allow_pickle=False) as saved:
        values = {name: saved[name].copy() for name in saved.files}
    required = {
        "angle_deg",
        "shift_y_px",
        "shift_x_px",
        "mirror",
        "soft_references",
        "class_averages",
    }
    if missing := required.difference(values):
        raise ValueError(f"accepted 2.2 baseline is missing arrays: {sorted(missing)}")
    if values["angle_deg"].shape != (3050,):
        raise ValueError("accepted 2.2 baseline does not contain 3,050 poses")
    return values


def _accuracy_metrics(
    result: ai.AlignmentResult,
    supplied_reference: np.ndarray,
    accepted: dict[str, np.ndarray],
) -> dict[str, Any]:
    accepted_poses = _poses(accepted)
    return {
        "pose_delta_to_accepted_2_2": pose_delta(accepted_poses, result.poses),
        "soft_reference_correlation_to_supplied": correlation(
            result.references[0], supplied_reference
        ),
        "class_average_correlation_to_supplied": correlation(
            result.class_averages[0], supplied_reference
        ),
        "soft_reference_correlation_to_accepted_2_2": correlation(
            result.references[0], accepted["soft_references"][0]
        ),
        "class_average_correlation_to_accepted_2_2": correlation(
            result.class_averages[0], accepted["class_averages"][0]
        ),
        "soft_reference_to_class_average_correlation": correlation(
            result.references[0], result.class_averages[0]
        ),
        "mean_expected_fourier_ncc": float(
            result.diagnostics[-1]["mean_expected_fourier_ncc"]
        ),
        "reference_relative_change": float(
            result.diagnostics[-1]["reference_relative_change"]
        ),
        "class_average_estimator": result.metadata["class_average_estimator"],
        "all_assignments_are_zero": bool(np.all(result.reference_assignments == 0)),
    }


def stage_accuracy_analysis(
    results: list[ai.AlignmentResult],
    supplied_reference: np.ndarray,
    accepted: dict[str, np.ndarray],
) -> dict[str, Any]:
    metrics = [
        _accuracy_metrics(result, supplied_reference, accepted) for result in results
    ]
    analysis: dict[str, Any] = {"stages": metrics}
    if len(results) == 2:
        before, after = metrics
        analysis["precise_change_from_fast"] = {
            "pose_delta": pose_delta(results[0].poses, results[1].poses),
            "changed_assignment_count": int(
                np.count_nonzero(
                    results[0].reference_assignments != results[1].reference_assignments
                )
            ),
            "soft_reference_correlation_to_supplied_delta": (
                after["soft_reference_correlation_to_supplied"]
                - before["soft_reference_correlation_to_supplied"]
            ),
            "class_average_correlation_to_supplied_delta": (
                after["class_average_correlation_to_supplied"]
                - before["class_average_correlation_to_supplied"]
            ),
            "mean_expected_fourier_ncc_delta": (
                after["mean_expected_fourier_ncc"] - before["mean_expected_fourier_ncc"]
            ),
        }
    return analysis


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=Path, default=DEFAULT_PARTICLES)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--accepted-baseline", type=Path, default=DEFAULT_ACCEPTED_BASELINE
    )
    parser.add_argument("--backend", choices=("cuda", "cupy", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--global-iterations", type=int, default=10)
    parser.add_argument("--precise-iterations", type=int, default=2)
    parser.add_argument("--deterministic-repeats", type=int, default=3)
    parser.add_argument("--profile-execution", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; reports are immutable")
    if (
        min(
            args.batch_size,
            args.global_iterations,
            args.precise_iterations,
            args.deterministic_repeats,
        )
        < 1
    ):
        parser.error("batch size, iteration counts, and repeats must be positive")
    if not 0.0 < args.memory_fraction <= 1.0:
        parser.error("memory fraction must be in (0, 1]")
    if (args.global_iterations, args.precise_iterations) != (10, 2):
        parser.error("the 3,050-particle investigation freezes a 10+2 schedule")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "stage": 5,
        "status": "running",
        "started_at_utc": utc_now(),
        "parameters": vars(args),
        "accuracy_interpretation": (
            "The user stack has no pose truth. Accuracy is evaluated against the "
            "supplied known reference and the accepted AlignImg 2.2 quadratic result; "
            "pose deltas are differences, not absolute errors."
        ),
        "gates": {
            "accepted_2_2_raw_correlation": ACCEPTED_RAW_CORRELATION,
            "maximum_allowed_drop": RAW_CORRELATION_TOLERANCE,
            "minimum_10plus2_raw_correlation": (
                ACCEPTED_RAW_CORRELATION - RAW_CORRELATION_TOLERANCE
            ),
        },
        "schedules": {},
    }
    write_report(args.output, report)
    try:
        input_hashes = {
            "particles": sha256(args.particles),
            "reference": sha256(args.reference),
            "accepted_baseline": sha256(args.accepted_baseline),
        }
        if input_hashes != EXPECTED_SHA256:
            raise ValueError(
                f"3,050-particle frozen input SHA-256 mismatch: {input_hashes}"
            )
        images, pixel_size = load_images(args.particles)
        references, _ = load_images(args.reference)
        if images.shape != (3050, 100, 100):
            raise ValueError("expected the frozen 3050x100x100 particle stack")
        if references.shape != (1, 100, 100):
            raise ValueError("expected one 100x100 supplied reference")
        accepted = load_accepted_baseline(args.accepted_baseline)
        case = {
            "images": images,
            "references": references,
            "component": np.zeros(len(images), dtype=np.int32),
            "priors": np.empty((0, 0), dtype=np.float32),
            "mirror_search": np.asarray(False, dtype=np.bool_),
            "truth_angle_deg": np.empty(0, dtype=np.float32),
            "truth_shift_y_px": np.empty(0, dtype=np.float32),
            "truth_shift_x_px": np.empty(0, dtype=np.float32),
            "truth_mirror": np.empty(0, dtype=np.bool_),
        }
        report.update(
            environment=environment_info(args.backend, args.device),
            runtime_identity=runtime_identity(args.backend),
            git=git_revision(),
            source=source_manifest(),
            inputs={
                "particle_count": len(images),
                "image_shape": list(images.shape[1:]),
                "reference_count": len(references),
                "pixel_size_angstrom": pixel_size,
                "sha256": input_hashes,
            },
        )
        write_report(args.output, report)
        configs = schedule_configs(
            global_iterations=args.global_iterations,
            precise_iterations=args.precise_iterations,
            batch_size=args.batch_size,
            memory_fraction=args.memory_fraction,
        )

        def analyze(stages: list[ai.AlignmentResult]) -> dict[str, Any]:
            return stage_accuracy_analysis(stages, references[0], accepted)

        for schedule, schedule_config in configs.items():
            print(
                f"RUN  known3050:k1 {schedule}: warm-up + "
                f"{args.deterministic_repeats} measured",
                flush=True,
            )
            report["schedules"][schedule] = run_schedule(
                "known3050:k1",
                case,
                schedule=schedule,
                configs=schedule_config,
                backend=args.backend,
                measured_repeats=args.deterministic_repeats,
                profile_execution=args.profile_execution,
                output=args.output,
                stage_analysis=analyze,
            )
            write_report(args.output, report)
        report["comparisons"] = compare_schedules(report["schedules"])
        ladder = report["schedules"]["fast_to_precise"]
        ladder_metrics = ladder["unprofiled_stage_analysis"]["stages"][-1]
        raw_correlation = ladder_metrics["class_average_correlation_to_supplied"]
        report["accuracy_gate"] = {
            "passed": bool(
                raw_correlation >= ACCEPTED_RAW_CORRELATION - RAW_CORRELATION_TOLERANCE
            ),
            "10plus2_raw_correlation": raw_correlation,
            "delta_from_accepted_2_2": raw_correlation - ACCEPTED_RAW_CORRELATION,
        }
        report["status"] = "completed"
        report["completed_at_utc"] = utc_now()
        write_report(args.output, report)
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["completed_at_utc"] = utc_now()
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        write_report(args.output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
