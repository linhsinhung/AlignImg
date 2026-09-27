#!/usr/bin/env python3
"""Measure the frozen 3/5/8/10/12 fast-only K=1 accuracy curve."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np

import alignimg as ai

try:
    from tools.fast_hard_known_reference_validation import (
        ACCEPTED_RAW_CORRELATION,
        DEFAULT_ACCEPTED_BASELINE,
        DEFAULT_PARTICLES,
        DEFAULT_REFERENCE,
        EXPECTED_SHA256,
        RAW_CORRELATION_TOLERANCE,
        load_accepted_baseline,
        stage_accuracy_analysis,
    )
    from tools.fast_hard_validation import (
        git_revision,
        runtime_identity,
        sha256,
        source_manifest,
    )
    from tools.fast_to_precise_validation import run_schedule
    from tools.quadratic_stage5_validation import load_images
    from tools.server_validation import environment_info, utc_now, write_report
except ModuleNotFoundError:
    from fast_hard_known_reference_validation import (
        ACCEPTED_RAW_CORRELATION,
        DEFAULT_ACCEPTED_BASELINE,
        DEFAULT_PARTICLES,
        DEFAULT_REFERENCE,
        EXPECTED_SHA256,
        RAW_CORRELATION_TOLERANCE,
        load_accepted_baseline,
        stage_accuracy_analysis,
    )
    from fast_hard_validation import (
        git_revision,
        runtime_identity,
        sha256,
        source_manifest,
    )
    from fast_to_precise_validation import run_schedule
    from quadratic_stage5_validation import load_images
    from server_validation import environment_info, utc_now, write_report


SCHEMA = "alignimg.fast-hard-iteration-sweep.v1"
ITERATIONS = (3, 5, 8, 10, 12)
PRACTICAL_RAW_CORRELATION = 0.95
DEFAULT_COMPARISON_REPORT = Path(
    "validation-results/fast-hard/stage-5/known3050-accuracy0-server-cuda-10plus2.json"
)
EXPECTED_COMPARISON_REPORT_SHA256 = (
    "1a78d77b218eaf578c0e636255507c5bb1560e553eb2ccc5601e8a2219032e46"
)


def load_comparison_report(path: Path) -> dict[str, Any]:
    if sha256(path) != EXPECTED_COMPARISON_REPORT_SHA256:
        raise ValueError("known3050 comparison report SHA-256 does not match")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "completed":
        raise ValueError("known3050 comparison report is not completed")
    if report.get("inputs", {}).get("sha256") != EXPECTED_SHA256:
        raise ValueError("known3050 comparison report uses different inputs")
    return report


def summarize_sweep(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    points = []
    previous_correlation = None
    for iterations in ITERATIONS:
        run = runs[str(iterations)]
        metrics = run["unprofiled_stage_analysis"]["stages"][0]
        correlation = metrics["class_average_correlation_to_supplied"]
        points.append(
            {
                "iterations": iterations,
                "raw_correlation": correlation,
                "raw_correlation_delta_from_previous_point": (
                    None
                    if previous_correlation is None
                    else correlation - previous_correlation
                ),
                "meets_practical_0_95": correlation >= PRACTICAL_RAW_CORRELATION,
                "meets_accepted_2_2_floor": correlation
                >= ACCEPTED_RAW_CORRELATION - RAW_CORRELATION_TOLERANCE,
                "median_seconds": run["unprofiled_median_seconds"],
            }
        )
        previous_correlation = correlation
    practical = [
        point["iterations"] for point in points if point["meets_practical_0_95"]
    ]
    accepted = [
        point["iterations"] for point in points if point["meets_accepted_2_2_floor"]
    ]
    return {
        "points": points,
        "minimum_iterations_meeting_practical_0_95": (
            min(practical) if practical else None
        ),
        "minimum_iterations_meeting_accepted_2_2_floor": (
            min(accepted) if accepted else None
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=Path, default=DEFAULT_PARTICLES)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--accepted-baseline", type=Path, default=DEFAULT_ACCEPTED_BASELINE
    )
    parser.add_argument(
        "--comparison-report", type=Path, default=DEFAULT_COMPARISON_REPORT
    )
    parser.add_argument("--backend", choices=("cuda", "cupy", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--deterministic-repeats", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; reports are immutable")
    if args.batch_size < 1 or args.deterministic_repeats != 3:
        parser.error("batch size must be positive and repeats are frozen at 3")
    if not 0.0 < args.memory_fraction <= 1.0:
        parser.error("memory fraction must be in (0, 1]")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "stage": 5,
        "status": "running",
        "started_at_utc": utc_now(),
        "parameters": vars(args),
        "iterations": ITERATIONS,
        "thresholds": {
            "practical_raw_correlation": PRACTICAL_RAW_CORRELATION,
            "accepted_2_2_raw_correlation": ACCEPTED_RAW_CORRELATION,
            "accepted_2_2_maximum_drop": RAW_CORRELATION_TOLERANCE,
            "accepted_2_2_floor": (
                ACCEPTED_RAW_CORRELATION - RAW_CORRELATION_TOLERANCE
            ),
        },
        "runs": {},
    }
    write_report(args.output, report)
    try:
        input_hashes = {
            "particles": sha256(args.particles),
            "reference": sha256(args.reference),
            "accepted_baseline": sha256(args.accepted_baseline),
        }
        if input_hashes != EXPECTED_SHA256:
            raise ValueError(f"frozen input SHA-256 mismatch: {input_hashes}")
        comparison = load_comparison_report(args.comparison_report)
        images, pixel_size = load_images(args.particles)
        references, _ = load_images(args.reference)
        if images.shape != (3050, 100, 100) or references.shape != (1, 100, 100):
            raise ValueError("expected the frozen 3050x100x100 K=1 workload")
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
        prior_balanced = comparison["schedules"]["balanced_soft"]
        prior_fast12 = comparison["schedules"]["fast_only"]
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
            frozen_comparison={
                "path": str(args.comparison_report),
                "sha256": EXPECTED_COMPARISON_REPORT_SHA256,
                "balanced_12_raw_correlation": prior_balanced[
                    "unprofiled_stage_analysis"
                ]["stages"][0]["class_average_correlation_to_supplied"],
                "balanced_12_median_seconds": prior_balanced[
                    "unprofiled_median_seconds"
                ],
                "fast_12_raw_correlation": prior_fast12["unprofiled_stage_analysis"][
                    "stages"
                ][0]["class_average_correlation_to_supplied"],
                "fast_12_median_seconds": prior_fast12["unprofiled_median_seconds"],
            },
        )
        write_report(args.output, report)

        def analyze(stages: list[ai.AlignmentResult]) -> dict[str, Any]:
            return stage_accuracy_analysis(stages, references[0], accepted)

        for iterations in ITERATIONS:
            config = replace(
                ai.AlignmentConfig.preset("fast_hard"),
                max_iterations=iterations,
                batch_size=args.batch_size,
                memory_fraction=args.memory_fraction,
                apply_final_pose_to_raw=True,
            ).normalized(workflow="global")
            if iterations == 12 and asdict(config) != prior_fast12["configs"][0]:
                raise AssertionError(
                    "12-iteration config differs from frozen comparison"
                )
            print(
                f"RUN  known3050:k1 fast-only {iterations}: warm-up + "
                f"{args.deterministic_repeats} measured",
                flush=True,
            )
            current = run_schedule(
                "known3050:k1",
                case,
                schedule=f"fast_only_i{iterations}",
                configs=(config,),
                backend=args.backend,
                measured_repeats=args.deterministic_repeats,
                profile_execution=False,
                output=args.output,
                stage_analysis=analyze,
            )
            current_metrics = current["unprofiled_stage_analysis"]["stages"][0]
            current["comparison_to_frozen_balanced_12"] = {
                "raw_correlation_delta": (
                    current_metrics["class_average_correlation_to_supplied"]
                    - report["frozen_comparison"]["balanced_12_raw_correlation"]
                ),
                "wall_speedup": (
                    report["frozen_comparison"]["balanced_12_median_seconds"]
                    / current["unprofiled_median_seconds"]
                ),
            }
            report["runs"][str(iterations)] = current
            write_report(args.output, report)
        report["summary"] = summarize_sweep(report["runs"])
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
