#!/usr/bin/env python3
"""Run the first controlled RF ablation against the RELION K=10 benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

if __package__:
    from tools import re2dc_70s_relion_benchmark as relion_benchmark
    from tools import re2dc_70s_rf_validation as rf_validation
else:
    import re2dc_70s_relion_benchmark as relion_benchmark
    import re2dc_70s_rf_validation as rf_validation


VARIANTS: dict[str, dict[str, int | float]] = {
    "baseline": {},
    "top_l_8": {"top_l": 8},
    "proposal_angles_8": {"proposal_angles_per_reference": 8},
    "temperature_end_0_05": {"temperature_end": 0.05},
}


def compact_rf_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "seconds",
        "particle_iterations_per_second",
        "hard_assignment_counts",
        "hard_occupancy_normalized_entropy",
        "mean_max_responsibility",
        "effective_component_weight",
        "reliable_median_stable_frc_resolution_angstrom",
        "component_weight_cv_trajectory",
        "maximum_reference_correlation_trajectory",
        "mean_expected_fourier_ncc_trajectory",
        "mean_max_responsibility_trajectory",
        "reference_relative_change_trajectory",
        "artifacts",
    )
    return {key: summary[key] for key in keys}


def run(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "alignimg.re2dc-70s-rf-ablation.v1",
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "principle": "one-factor-at-a-time relative to the reference-free preset",
            "fixed": {
                "particle_count": 1000,
                "component_count": 10,
                "seed": args.seed,
                "iterations": args.iterations,
                "anneal_iterations": args.anneal_iterations,
                "angle_samples": args.angle_samples,
                "translation_range": args.translation_range,
            },
            "variants": VARIANTS,
        },
        "input": {
            "stack": str(args.stack),
            "prepared_star": str(args.prepared_star),
            "relion_star": str(args.relion_star),
        },
        "backend": args.backend,
        "runs": {},
    }
    rf_validation.write_json(args.output, report)
    for name, overrides in VARIANTS.items():
        print(f"ABLATION {name}", flush=True)
        validation_output = args.output.with_suffix(f".{name}.json")
        validation_args = argparse.Namespace(
            stack=args.stack,
            output=validation_output,
            backend=args.backend,
            components=10,
            iterations=args.iterations,
            anneal_iterations=args.anneal_iterations,
            seeds=[args.seed],
            angle_samples=args.angle_samples,
            top_l=overrides.get("top_l"),
            proposal_angles_per_reference=overrides.get(
                "proposal_angles_per_reference"
            ),
            temperature_start=overrides.get("temperature_start"),
            temperature_end=overrides.get("temperature_end"),
            translation_range=args.translation_range,
            batch_size=args.batch_size,
            memory_fraction=args.memory_fraction,
            minimum_frc_halfset_weight=args.minimum_frc_halfset_weight,
        )
        try:
            rf_validation.main(validation_args)
            result_path = Path(
                f"{validation_output.with_suffix(f'.seed-{args.seed}')}.result.npz"
            )
            benchmark_output = validation_output.with_suffix(".relion.json")
            benchmark_args = argparse.Namespace(
                relion_star=args.relion_star,
                prepared_star=args.prepared_star,
                rf_results=[result_path],
                output=benchmark_output,
                confidence_thresholds=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
            )
            benchmark_report = relion_benchmark.build_report(benchmark_args)
            rf_validation.write_json(benchmark_output, benchmark_report)
            validation_report = json.loads(
                validation_output.read_text(encoding="utf-8")
            )
            benchmark_run = benchmark_report["runs"][0]
            report["runs"][name] = {
                "status": "completed",
                "overrides": overrides,
                "rf_validation_report": str(validation_output),
                "relion_benchmark_report": str(benchmark_output),
                "rf_summary": compact_rf_summary(
                    validation_report["runs"][str(args.seed)]
                ),
                "relion_overall": benchmark_run["overall"],
                "relion_confidence_strata": benchmark_run[
                    "relion_confidence_strata"
                ],
            }
        except Exception as error:
            report["runs"][name] = {
                "status": "failed",
                "overrides": overrides,
                "error": f"{type(error).__name__}: {error}",
            }
            report["status"] = "failed"
            rf_validation.write_json(args.output, report)
            raise
        rf_validation.write_json(args.output, report)
    report["status"] = "completed"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    rf_validation.write_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stack",
        type=Path,
        default=Path(
            "data/re2dc_70s_testdata/prepared/re2dc_70s_n1000_s128.mrcs"
        ),
    )
    parser.add_argument(
        "--prepared-star",
        type=Path,
        default=Path(
            "data/re2dc_70s_testdata/prepared/re2dc_70s_n1000_s128.star"
        ),
    )
    parser.add_argument(
        "--relion-star",
        type=Path,
        default=relion_benchmark.DEFAULT_RELION_STAR,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation-results/re2dc-70s-rf-ablation.json"),
    )
    parser.add_argument(
        "--backend", choices=("cpu", "cuda", "cupy", "gpu", "auto"), default="cuda"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--anneal-iterations", type=int, default=10)
    parser.add_argument("--angle-samples", type=int, default=128)
    parser.add_argument("--translation-range", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--minimum-frc-halfset-weight", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)
    print(f"Ablation report saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
