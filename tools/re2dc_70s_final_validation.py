#!/usr/bin/env python3
"""Run the final 1000-particle RF-to-feedback scientific validation gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from typing import Any

import mrcfile
import numpy as np

import alignimg as ai

if __package__:
    from tools import re2dc_70s_feedback_validation as feedback_validation
    from tools import re2dc_70s_relion_benchmark as relion_benchmark
    from tools import re2dc_70s_rf_validation as rf_validation
else:
    import re2dc_70s_feedback_validation as feedback_validation
    import re2dc_70s_relion_benchmark as relion_benchmark
    import re2dc_70s_rf_validation as rf_validation


DEFAULT_STACK = Path(
    "data/re2dc_70s_testdata/prepared/re2dc_70s_n1000_s128.mrcs"
)
DEFAULT_PREPARED_STAR = Path(
    "data/re2dc_70s_testdata/prepared/re2dc_70s_n1000_s128.star"
)
DEFAULT_RELION_STAR = Path(
    "data/re2dc_70s_testdata/particles_Relion2Dclassification.star"
)
DEFAULT_OUTPUT = Path("validation-results/re2dc-70s-final-n1000.json")
PRODUCTION_PATH = {
    "candidate_scoring": "fourier",
    "score_model": "fourier_ncc",
    "reference_update": "fourier",
}


def derived_path(output: Path, suffix: str) -> Path:
    return Path(f"{output.with_suffix('')}.{suffix}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_result_bundle(
    result_path: Path,
    reference_path: Path,
    *,
    particle_count: int,
    component_count: int,
    probability_tolerance: float = 2e-5,
) -> dict[str, Any]:
    if not result_path.is_file() or not reference_path.is_file():
        raise RuntimeError(f"missing result artifacts for {result_path}")
    with mrcfile.mmap(reference_path, permissive=True, mode="r") as mrc:
        references = np.asarray(mrc.data)
        references_finite = bool(np.all(np.isfinite(references)))
    if references.shape[0] != component_count or not references_finite:
        raise RuntimeError(f"invalid reference artifact {reference_path}")

    with np.load(result_path, allow_pickle=False) as saved:
        required = (
            "angle_deg",
            "shift_y_px",
            "shift_x_px",
            "assignments",
            "responsibilities",
            "inlier_weights",
        )
        missing = [name for name in required if name not in saved.files]
        if missing:
            raise RuntimeError(f"{result_path} is missing arrays: {missing}")
        finite = all(bool(np.all(np.isfinite(saved[name]))) for name in required)
        assignments = np.asarray(saved["assignments"])
        responsibilities = np.asarray(saved["responsibilities"])
    if not finite:
        raise RuntimeError(f"{result_path} contains non-finite values")
    if assignments.shape != (particle_count,):
        raise RuntimeError(f"{result_path} has invalid assignment shape")
    if responsibilities.shape != (particle_count, component_count):
        raise RuntimeError(f"{result_path} has invalid responsibility shape")
    if np.any(assignments < 0) or np.any(assignments >= component_count):
        raise RuntimeError(f"{result_path} contains out-of-range assignments")
    responsibility_error = float(
        np.max(np.abs(responsibilities.sum(axis=1) - 1.0))
    )
    if responsibility_error > probability_tolerance:
        raise RuntimeError(
            f"{result_path} responsibility error {responsibility_error:g} exceeds "
            f"{probability_tolerance:g}"
        )
    return {
        "result": str(result_path),
        "references": str(reference_path),
        "all_required_arrays_finite": True,
        "references_finite": True,
        "responsibility_sum_max_error": responsibility_error,
    }


def require_production_path(report: dict[str, Any], location: str) -> None:
    config = report[location]["config"]
    actual = {name: config[name] for name in PRODUCTION_PATH}
    if actual != PRODUCTION_PATH:
        raise RuntimeError(
            f"validation did not use the fixed production path: {actual}"
        )


def compact_relion(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "result": run["result"],
            "hard_assignment_counts": run["alignimg"]["hard_assignment_counts"],
            "mean_max_responsibility": run["alignimg"].get(
                "mean_max_responsibility"
            ),
            "adjusted_rand_index": run["overall"]["adjusted_rand_index"],
            "normalized_mutual_information": run["overall"][
                "normalized_mutual_information"
            ],
            "optimal_label_agreement": run["overall"][
                "optimal_label_agreement"
            ],
        }
        for run in report["runs"]
    ]


def main(args: argparse.Namespace) -> None:
    for path in (args.stack, args.prepared_star, args.relion_star):
        if not path.is_file():
            raise FileNotFoundError(path)
    with mrcfile.mmap(args.stack, permissive=True, mode="r") as mrc:
        particle_count = int(mrc.data.shape[0])

    rf_output = derived_path(args.output, "rf.json")
    feedback_output = derived_path(args.output, "feedback.json")
    relion_output = derived_path(args.output, "relion.json")
    rf_base = rf_output.with_suffix(f".seed-{args.seed}")
    rf_result = Path(f"{rf_base}.result.npz")
    rf_references = Path(f"{rf_base}.references.mrcs")
    feedback_base = feedback_output.with_suffix("")

    report: dict[str, Any] = {
        "schema": "alignimg.re2dc-70s-final-validation.v1",
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "alignimg_version": ai.__version__,
        "scope_note": (
            "RELION comparison metrics are scientific observations, not pass/fail "
            "thresholds and not claims of biological classification accuracy."
        ),
        "input": {
            "stack": str(args.stack),
            "prepared_star": str(args.prepared_star),
            "relion_star": str(args.relion_star),
            "particle_count": particle_count,
        },
        "spec": {
            "backend": args.backend,
            "component_count": args.components,
            "production_path": PRODUCTION_PATH,
            "rf_iterations": args.rf_iterations,
            "rf_anneal_iterations": args.rf_anneal_iterations,
            "rf_seed": args.seed,
            "feedback_iterations": args.feedback_iterations,
            "feedback_anneal_iterations": args.feedback_anneal_iterations,
            "corrective_trust": args.corrective_trust,
            "batch_size": args.batch_size,
            "memory_fraction": args.memory_fraction,
        },
        "phase": "rf",
        "artifacts": {
            "rf_report": str(rf_output),
            "feedback_report": str(feedback_output),
            "relion_report": str(relion_output),
        },
    }
    rf_validation.write_json(args.output, report)

    try:
        print("PHASE 1/3  reference-free alignment", flush=True)
        rf_validation.main(
            argparse.Namespace(
                stack=args.stack,
                output=rf_output,
                backend=args.backend,
                components=args.components,
                iterations=args.rf_iterations,
                anneal_iterations=args.rf_anneal_iterations,
                seeds=[args.seed],
                angle_samples=args.angle_samples,
                candidate_scoring=PRODUCTION_PATH["candidate_scoring"],
                score_model=PRODUCTION_PATH["score_model"],
                reference_update=PRODUCTION_PATH["reference_update"],
                top_l=None,
                proposal_angles_per_reference=None,
                temperature_start=None,
                temperature_end=None,
                translation_range=args.translation_range,
                batch_size=args.batch_size,
                memory_fraction=args.memory_fraction,
                minimum_frc_halfset_weight=args.minimum_frc_halfset_weight,
            )
        )
        rf_report = read_json(rf_output)
        if rf_report["status"] != "completed":
            raise RuntimeError("RF report did not complete")
        require_production_path(rf_report, "parameters")
        rf_check = validate_result_bundle(
            rf_result,
            rf_references,
            particle_count=particle_count,
            component_count=args.components,
        )
        rf_summary = rf_report["runs"][str(args.seed)]
        report["rf"] = {
            "seconds": rf_summary["seconds"],
            "particle_iterations_per_second": rf_summary[
                "particle_iterations_per_second"
            ],
            "hard_assignment_counts": rf_summary["hard_assignment_counts"],
            "hard_occupancy_normalized_entropy": rf_summary[
                "hard_occupancy_normalized_entropy"
            ],
            "mean_max_responsibility": rf_summary["mean_max_responsibility"],
            "reliable_median_stable_frc_resolution_angstrom": rf_summary[
                "reliable_median_stable_frc_resolution_angstrom"
            ],
            "technical_checks": rf_check,
        }
        report["phase"] = "feedback"
        rf_validation.write_json(args.output, report)

        print("PHASE 2/3  fixed and corrective feedback", flush=True)
        feedback_validation.main(
            argparse.Namespace(
                stack=args.stack,
                references=rf_references,
                rf_result=rf_result,
                output=feedback_output,
                backend=args.backend,
                iterations=args.feedback_iterations,
                anneal_iterations=args.feedback_anneal_iterations,
                candidate_scoring=PRODUCTION_PATH["candidate_scoring"],
                score_model=PRODUCTION_PATH["score_model"],
                reference_update=PRODUCTION_PATH["reference_update"],
                corrective_trust=args.corrective_trust,
                corrective_prior_source="responsibilities",
                minimum_frc_halfset_weight=args.minimum_frc_halfset_weight,
                batch_size=args.batch_size,
                memory_fraction=args.memory_fraction,
            )
        )
        feedback_report = read_json(feedback_output)
        if feedback_report["status"] != "completed":
            raise RuntimeError("feedback report did not complete")
        require_production_path(feedback_report, "parameters")
        feedback_checks: dict[str, Any] = {}
        feedback_results: list[Path] = []
        for mode in ("fixed", "corrective"):
            result_path = Path(f"{feedback_base}.{mode}.result.npz")
            reference_path = Path(f"{feedback_base}.{mode}.references.mrcs")
            feedback_results.append(result_path)
            feedback_checks[mode] = validate_result_bundle(
                result_path,
                reference_path,
                particle_count=particle_count,
                component_count=args.components,
            )
        with np.load(rf_result, allow_pickle=False) as initial, np.load(
            feedback_results[0], allow_pickle=False
        ) as fixed:
            if not np.array_equal(initial["assignments"], fixed["assignments"]):
                raise RuntimeError("fixed feedback changed RF assignments")
        report["feedback"] = {
            "fixed": feedback_report["runs"]["fixed"],
            "corrective": feedback_report["runs"]["corrective"],
            "technical_checks": feedback_checks,
        }
        report["phase"] = "relion_comparison"
        rf_validation.write_json(args.output, report)

        print("PHASE 3/3  RELION partition comparison", flush=True)
        relion_args = argparse.Namespace(
            relion_star=args.relion_star,
            prepared_star=args.prepared_star,
            rf_results=[rf_result, *feedback_results],
            output=relion_output,
            confidence_thresholds=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
        )
        relion_report = relion_benchmark.build_report(relion_args)
        rf_validation.write_json(relion_output, relion_report)
        report["relion_comparison"] = compact_relion(relion_report)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        rf_validation.write_json(args.output, report)
        raise

    report["status"] = "completed"
    report["phase"] = "completed"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    rf_validation.write_json(args.output, report)
    print(f"PASS final 1000-particle gate: {args.output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", type=Path, default=DEFAULT_STACK)
    parser.add_argument("--prepared-star", type=Path, default=DEFAULT_PREPARED_STAR)
    parser.add_argument("--relion-star", type=Path, default=DEFAULT_RELION_STAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--backend", choices=("cpu", "cuda", "cupy", "gpu", "auto"), default="cuda"
    )
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--rf-iterations", type=int, default=15)
    parser.add_argument("--rf-anneal-iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feedback-iterations", type=int, default=5)
    parser.add_argument("--feedback-anneal-iterations", type=int, default=3)
    parser.add_argument("--corrective-trust", type=float, default=0.9)
    parser.add_argument("--angle-samples", type=int, default=128)
    parser.add_argument("--translation-range", type=float, default=4.0)
    parser.add_argument("--minimum-frc-halfset-weight", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
