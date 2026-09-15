#!/usr/bin/env python3
"""Compare AlignImg RF assignments with a RELION 2D classification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

if __package__:
    from tools.prepare_re2dc_70s_benchmark import read_particle_loop
    from tools.re2dc_70s_rf_validation import adjusted_rand_index, write_json
else:
    from prepare_re2dc_70s_benchmark import read_particle_loop
    from re2dc_70s_rf_validation import adjusted_rand_index, write_json


DEFAULT_RELION_STAR = Path(
    "data/re2dc_70s_testdata/particles_Relion2Dclassification.star"
)
DEFAULT_PREPARED_STAR = Path(
    "data/re2dc_70s_testdata/prepared/re2dc_70s_n5000_s128.star"
)
DEFAULT_OUTPUT = Path("validation-results/re2dc-70s-relion-benchmark.json")
DEFAULT_RESULTS = tuple(
    Path(
        f"validation-results/re2dc-70s-rf-n5000-k10-i20.seed-{seed}.result.npz"
    )
    for seed in range(3)
)


def read_star_columns(path: Path) -> dict[str, np.ndarray]:
    labels, rows = read_particle_loop(path)
    values = np.asarray(rows, dtype=str)
    return {label: values[:, index] for index, label in enumerate(labels)}


def integer_column(columns: dict[str, np.ndarray], label: str) -> np.ndarray:
    if label not in columns:
        raise ValueError(f"STAR is missing required field {label}")
    values = np.asarray(columns[label], dtype=np.float64)
    if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise ValueError(f"{label} must contain finite integers")
    return values.astype(np.int64)


def optional_float_column(
    columns: dict[str, np.ndarray], label: str, particle_indices: np.ndarray
) -> np.ndarray | None:
    if label not in columns:
        return None
    values = np.asarray(columns[label], dtype=np.float64)[particle_indices]
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain finite values")
    return values


def align_relion_metadata(
    relion_star: Path, prepared_star: Path
) -> dict[str, np.ndarray | None]:
    """Return RELION metadata in the exact particle order of the prepared stack."""
    relion = read_star_columns(relion_star)
    prepared = read_star_columns(prepared_star)
    relion_ids = integer_column(relion, "_rlnImageId")
    prepared_ids = integer_column(prepared, "_rlnImageId")
    if len(np.unique(relion_ids)) != len(relion_ids):
        raise ValueError("RELION STAR contains duplicate _rlnImageId values")
    if len(np.unique(prepared_ids)) != len(prepared_ids):
        raise ValueError("prepared STAR contains duplicate _rlnImageId values")
    row_by_id = {int(image_id): row for row, image_id in enumerate(relion_ids)}
    missing = [int(image_id) for image_id in prepared_ids if int(image_id) not in row_by_id]
    if missing:
        preview = ", ".join(str(value) for value in missing[:8])
        raise ValueError(f"RELION STAR is missing prepared particle ImageId values: {preview}")
    indices = np.asarray([row_by_id[int(image_id)] for image_id in prepared_ids])
    class_number = integer_column(relion, "_rlnClassNumber")[indices]
    if np.any(class_number < 1):
        raise ValueError("_rlnClassNumber values must be one-based positive integers")
    enabled = optional_float_column(relion, "_rlnEnabled", indices)
    if enabled is not None and np.any(enabled != 1.0):
        raise ValueError("RELION benchmark contains disabled prepared particles")
    return {
        "image_id": prepared_ids,
        "class_number": class_number,
        "group_number": (
            integer_column(relion, "_rlnGroupNumber")[indices]
            if "_rlnGroupNumber" in relion
            else None
        ),
        "micrograph_id": (
            integer_column(relion, "_rlnMicrographId")[indices]
            if "_rlnMicrographId" in relion
            else None
        ),
        "max_value_probability": optional_float_column(
            relion, "_rlnMaxValueProbDistribution", indices
        ),
        "significant_samples": optional_float_column(
            relion, "_rlnNrOfSignificantSamples", indices
        ),
        "angle_psi_deg": optional_float_column(relion, "_rlnAnglePsi", indices),
        "origin_x_angstrom": optional_float_column(
            relion, "_rlnOriginXAngst", indices
        ),
        "origin_y_angstrom": optional_float_column(
            relion, "_rlnOriginYAngst", indices
        ),
    }


def contingency_table(
    reference: np.ndarray, predicted: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_labels, reference_inverse = np.unique(reference, return_inverse=True)
    predicted_labels, predicted_inverse = np.unique(predicted, return_inverse=True)
    table = np.zeros((len(reference_labels), len(predicted_labels)), dtype=np.int64)
    np.add.at(table, (reference_inverse, predicted_inverse), 1)
    return table, reference_labels, predicted_labels


def normalized_mutual_information(reference: np.ndarray, predicted: np.ndarray) -> float:
    table, _, _ = contingency_table(reference, predicted)
    total = float(np.sum(table))
    if total == 0.0:
        raise ValueError("cannot compare empty assignments")
    joint = table / total
    reference_probability = np.sum(joint, axis=1)
    predicted_probability = np.sum(joint, axis=0)
    expected = reference_probability[:, None] * predicted_probability[None, :]
    selected = joint > 0.0
    mutual_information = float(np.sum(joint[selected] * np.log(joint[selected] / expected[selected])))
    reference_entropy = float(
        -np.sum(reference_probability[reference_probability > 0.0] * np.log(reference_probability[reference_probability > 0.0]))
    )
    predicted_entropy = float(
        -np.sum(predicted_probability[predicted_probability > 0.0] * np.log(predicted_probability[predicted_probability > 0.0]))
    )
    denominator = 0.5 * (reference_entropy + predicted_entropy)
    if denominator == 0.0:
        return 1.0
    return mutual_information / denominator


def optimal_label_mapping(
    reference: np.ndarray, predicted: np.ndarray
) -> tuple[dict[int, int], float, np.ndarray, np.ndarray, np.ndarray]:
    table, reference_labels, predicted_labels = contingency_table(reference, predicted)
    row_indices, column_indices = linear_sum_assignment(-table)
    mapping = {
        int(predicted_labels[column]): int(reference_labels[row])
        for row, column in zip(row_indices, column_indices)
    }
    matched = sum(int(table[row, column]) for row, column in zip(row_indices, column_indices))
    return mapping, matched / len(reference), table, reference_labels, predicted_labels


def mapped_agreement(
    reference: np.ndarray, predicted: np.ndarray, mapping: dict[int, int]
) -> float:
    mapped = np.asarray([mapping.get(int(value), -1) for value in predicted])
    return float(np.mean(mapped == reference))


def comparison_metrics(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    global_mapping: dict[int, int] | None = None,
    include_confusion: bool = True,
) -> dict[str, Any]:
    if len(reference) != len(predicted):
        raise ValueError("reference and predicted assignments must have equal length")
    if len(reference) == 0:
        raise ValueError("cannot compare empty assignments")
    mapping, agreement, table, reference_labels, predicted_labels = optimal_label_mapping(
        reference, predicted
    )
    result: dict[str, Any] = {
        "particle_count": len(reference),
        "adjusted_rand_index": adjusted_rand_index(reference, predicted),
        "normalized_mutual_information": normalized_mutual_information(
            reference, predicted
        ),
        "optimal_label_agreement": agreement,
        "optimal_predicted_to_relion_mapping": mapping,
    }
    if include_confusion:
        result.update(
            relion_class_labels=reference_labels,
            alignimg_class_labels=predicted_labels,
            confusion_matrix_relion_rows_alignimg_columns=table,
            confusion_matrix_row_normalized=table
            / np.maximum(np.sum(table, axis=1, keepdims=True), 1),
        )
    if global_mapping is not None:
        result["agreement_using_global_mapping"] = mapped_agreement(
            reference, predicted, global_mapping
        )
    return result


def confidence_strata(
    reference: np.ndarray,
    predicted: np.ndarray,
    probability: np.ndarray | None,
    global_mapping: dict[int, int],
    thresholds: list[float],
) -> dict[str, Any] | None:
    if probability is None:
        return None
    strata: dict[str, Any] = {}
    for threshold in thresholds:
        selected = probability >= threshold
        key = f"greater_or_equal_{threshold:g}"
        strata[key] = (
            comparison_metrics(
                reference[selected],
                predicted[selected],
                global_mapping=global_mapping,
                include_confusion=False,
            )
            if np.any(selected)
            else {"particle_count": 0}
        )
    below = probability < thresholds[0]
    strata[f"less_than_{thresholds[0]:g}"] = (
        comparison_metrics(
            reference[below],
            predicted[below],
            global_mapping=global_mapping,
            include_confusion=False,
        )
        if np.any(below)
        else {"particle_count": 0}
    )
    return strata


def significant_sample_strata(
    reference: np.ndarray,
    predicted: np.ndarray,
    significant_samples: np.ndarray | None,
    global_mapping: dict[int, int],
) -> dict[str, Any] | None:
    if significant_samples is None:
        return None
    bins = {
        "1": significant_samples == 1,
        "2": significant_samples == 2,
        "3_to_4": (significant_samples >= 3) & (significant_samples <= 4),
        "5_to_8": (significant_samples >= 5) & (significant_samples <= 8),
        "9_or_more": significant_samples >= 9,
    }
    return {
        name: comparison_metrics(
            reference[selected],
            predicted[selected],
            global_mapping=global_mapping,
            include_confusion=False,
        )
        if np.any(selected)
        else {"particle_count": 0}
        for name, selected in bins.items()
    }


def load_assignments(path: Path, expected_count: int) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as result:
        if "assignments" not in result:
            raise ValueError(f"{path} does not contain assignments")
        raw_assignments = np.asarray(result["assignments"])
        if raw_assignments.ndim != 1 or len(raw_assignments) != expected_count:
            raise ValueError(
                f"{path} assignments must have shape ({expected_count},), got {raw_assignments.shape}"
            )
        assignments = raw_assignments.astype(np.int64)
        if not np.all(np.isfinite(raw_assignments)) or not np.all(raw_assignments == assignments):
            raise ValueError(f"{path} assignments must contain finite integers")
        if np.any(assignments < 0):
            raise ValueError(f"{path} assignments must be zero-based non-negative integers")
        diagnostics: dict[str, Any] = {
            "class_count": int(np.max(assignments)) + 1,
            "hard_assignment_counts": np.bincount(assignments),
        }
        if "responsibilities" in result:
            responsibilities = np.asarray(result["responsibilities"], dtype=np.float64)
            if responsibilities.shape[0] != expected_count:
                raise ValueError(f"{path} responsibilities have the wrong particle count")
            diagnostics["mean_max_responsibility"] = float(
                np.mean(np.max(responsibilities, axis=1))
            )
    return assignments, diagnostics


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    metadata = align_relion_metadata(args.relion_star, args.prepared_star)
    relion_class = np.asarray(metadata["class_number"], dtype=np.int64)
    max_probability = metadata["max_value_probability"]
    significant_samples = metadata["significant_samples"]
    group_number = metadata["group_number"]
    micrograph_id = metadata["micrograph_id"]
    report: dict[str, Any] = {
        "schema": "alignimg.re2dc-70s-relion-benchmark.v1",
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope_note": (
            "RELION assignments are an external reference partition, not biological "
            "ground truth. Label-permutation-invariant metrics are primary."
        ),
        "input": {
            "relion_star": str(args.relion_star),
            "prepared_star": str(args.prepared_star),
            "rf_results": [str(path) for path in args.rf_results],
            "particle_count": len(relion_class),
            "mapping_key": "_rlnImageId",
        },
        "relion": {
            "class_counts": {
                int(label): int(count)
                for label, count in zip(*np.unique(relion_class, return_counts=True))
            },
            "group_counts": (
                {
                    int(label): int(count)
                    for label, count in zip(*np.unique(group_number, return_counts=True))
                }
                if group_number is not None
                else None
            ),
            "group_number_equals_micrograph_id": (
                bool(np.array_equal(group_number, micrograph_id))
                if group_number is not None and micrograph_id is not None
                else None
            ),
            "max_value_probability_quantiles": (
                dict(
                    zip(
                        ("minimum", "p05", "median", "p95", "maximum"),
                        np.quantile(max_probability, (0.0, 0.05, 0.5, 0.95, 1.0)),
                    )
                )
                if max_probability is not None
                else None
            ),
            "significant_samples_quantiles": (
                dict(
                    zip(
                        ("minimum", "p05", "median", "p95", "maximum"),
                        np.quantile(significant_samples, (0.0, 0.05, 0.5, 0.95, 1.0)),
                    )
                )
                if significant_samples is not None
                else None
            ),
        },
        "runs": [],
    }
    for result_path in args.rf_results:
        assignments, diagnostics = load_assignments(result_path, len(relion_class))
        overall = comparison_metrics(relion_class, assignments)
        global_mapping = overall["optimal_predicted_to_relion_mapping"]
        report["runs"].append(
            {
                "result": str(result_path),
                "alignimg": diagnostics,
                "overall": overall,
                "relion_confidence_strata": confidence_strata(
                    relion_class,
                    assignments,
                    max_probability,
                    global_mapping,
                    args.confidence_thresholds,
                ),
                "relion_significant_sample_strata": significant_sample_strata(
                    relion_class,
                    assignments,
                    significant_samples,
                    global_mapping,
                ),
            }
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relion-star", type=Path, default=DEFAULT_RELION_STAR)
    parser.add_argument("--prepared-star", type=Path, default=DEFAULT_PREPARED_STAR)
    parser.add_argument("--rf-results", type=Path, nargs="+", default=list(DEFAULT_RESULTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--confidence-thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
    )
    args = parser.parse_args()
    if any(value < 0.0 or value > 1.0 for value in args.confidence_thresholds):
        parser.error("confidence thresholds must be in [0, 1]")
    args.confidence_thresholds = sorted(set(args.confidence_thresholds))
    return args


def main(args: argparse.Namespace) -> None:
    report = build_report(args)
    write_json(args.output, report)
    for run in report["runs"]:
        metrics = run["overall"]
        print(
            f"{Path(run['result']).name}: "
            f"ARI={metrics['adjusted_rand_index']:.4f} "
            f"NMI={metrics['normalized_mutual_information']:.4f} "
            f"agreement={metrics['optimal_label_agreement']:.2%}"
        )
    print(f"Report saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main(parse_args())
