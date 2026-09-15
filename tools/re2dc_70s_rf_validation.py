#!/usr/bin/env python3
"""Run reference-free AlignImg validation on the prepared RE2DC 70S stack."""

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
from typing import Any

import mrcfile
import numpy as np
from scipy.optimize import linear_sum_assignment

import alignimg as ai
from alignimg._geometry import integer_center


DEFAULT_STACK = Path(
    "data/re2dc_70s_testdata/prepared/re2dc_70s_n1000_s128.mrcs"
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


def normalized_entropy(counts: np.ndarray) -> float:
    probabilities = counts / max(float(np.sum(counts)), 1.0)
    selected = probabilities > 0.0
    return float(-np.sum(probabilities[selected] * np.log(probabilities[selected])) / np.log(len(counts)))


def adjusted_rand_index(first: np.ndarray, second: np.ndarray) -> float:
    first_values, first_inverse = np.unique(first, return_inverse=True)
    second_values, second_inverse = np.unique(second, return_inverse=True)
    table = np.zeros((len(first_values), len(second_values)), dtype=np.int64)
    np.add.at(table, (first_inverse, second_inverse), 1)
    choose_two = lambda values: values * (values - 1) / 2
    agreement = float(np.sum(choose_two(table)))
    first_pairs = float(np.sum(choose_two(table.sum(axis=1))))
    second_pairs = float(np.sum(choose_two(table.sum(axis=0))))
    total_pairs = float(choose_two(np.asarray([len(first)]))[0])
    expected = first_pairs * second_pairs / max(total_pairs, 1.0)
    maximum = 0.5 * (first_pairs + second_pairs)
    if maximum == expected:
        return 1.0
    return (agreement - expected) / (maximum - expected)


def radial_profiles(references: np.ndarray) -> np.ndarray:
    size = references.shape[1]
    center = integer_center(size)
    y, x = np.indices((size, size), dtype=np.float32)
    rings = np.floor(np.sqrt((y - center) ** 2 + (x - center) ** 2)).astype(np.int32)
    profiles = np.empty((len(references), size // 2), dtype=np.float64)
    for index, reference in enumerate(references):
        magnitude = np.abs(np.fft.fftshift(np.fft.fft2(reference)))
        profiles[index] = [np.mean(magnitude[rings == ring]) for ring in range(size // 2)]
    profiles -= profiles.mean(axis=1, keepdims=True)
    profiles /= np.maximum(np.linalg.norm(profiles, axis=1, keepdims=True), 1e-12)
    return profiles


def run_metrics(
    result: ai.AlignmentResult,
    seconds: float,
    component_count: int,
    minimum_halfset_weight: float,
    pixel_size: float | None,
) -> dict[str, Any]:
    hard_counts = np.bincount(result.reference_assignments, minlength=component_count)
    bootstrap_counts = np.bincount(
        result.metadata["bootstrap_labels"], minlength=component_count
    )
    final = result.diagnostics[-1]
    halfset_weights = np.asarray(final["halfset_effective_weight"])
    stable_cutoff = np.asarray(final["frc_0143_stable_cutoff_cyc_per_px"])
    reliable_frc = np.min(halfset_weights, axis=0) >= minimum_halfset_weight
    reliable_values = stable_cutoff[reliable_frc & (stable_cutoff > 0.0)]
    return {
        "seconds": seconds,
        "particles_per_second": len(result.poses) / max(seconds, 1e-12),
        "particle_iterations_per_second": (
            len(result.poses) * len(result.diagnostics) / max(seconds, 1e-12)
        ),
        "bootstrap_counts": bootstrap_counts,
        "bootstrap_occupancy_normalized_entropy": normalized_entropy(bootstrap_counts),
        "hard_assignment_counts": hard_counts,
        "hard_occupancy_normalized_entropy": normalized_entropy(hard_counts),
        "mean_max_responsibility": float(
            np.mean(np.max(result.responsibilities, axis=1))
        ),
        "responsibility_sum_max_error": float(
            np.max(np.abs(result.responsibilities.sum(axis=1) - 1.0))
        ),
        "effective_component_weight": final["effective_component_weight"],
        "frc_0143_cutoff_cyc_per_px": final["frc_0143_cutoff_cyc_per_px"],
        "frc_0143_stable_cutoff_cyc_per_px": stable_cutoff,
        "frc_reliable_component_mask": reliable_frc,
        "frc_minimum_required_effective_weight_per_half": minimum_halfset_weight,
        "reliable_median_stable_frc_cutoff_cyc_per_px": (
            float(np.median(reliable_values)) if len(reliable_values) else None
        ),
        "reliable_median_stable_frc_resolution_angstrom": (
            float(pixel_size / np.median(reliable_values))
            if pixel_size is not None and len(reliable_values)
            else None
        ),
        "component_weight_cv_trajectory": [
            item["component_weight_cv"] for item in result.diagnostics
        ],
        "minimum_component_weight_to_mean_trajectory": [
            item["minimum_component_weight_to_mean"] for item in result.diagnostics
        ],
        "maximum_reference_correlation_trajectory": [
            item["maximum_offdiagonal_reference_correlation"]
            for item in result.diagnostics
        ],
        "mean_expected_fourier_ncc_trajectory": [
            item["mean_expected_fourier_ncc"] for item in result.diagnostics
        ],
        "temperature_trajectory": [
            item["temperature"] for item in result.diagnostics
        ],
        "reference_relative_change_trajectory": [
            item["reference_relative_change"] for item in result.diagnostics
        ],
        "mean_max_responsibility_trajectory": [
            item["mean_max_responsibility"] for item in result.diagnostics
        ],
        "median_stable_frc_0143_cutoff_cyc_per_px_trajectory": [
            float(np.median(item["frc_0143_stable_cutoff_cyc_per_px"]))
            for item in result.diagnostics
        ],
        "reseeded_component_count_trajectory": [
            len(item["reseeded_components"]) for item in result.diagnostics
        ],
        "metadata": result.metadata,
    }


def save_run(
    path: Path,
    result: ai.AlignmentResult,
    *,
    pixel_size: float | None,
    source_stack: Path,
    backend: str,
    config: ai.AlignmentConfig,
    seed: int,
    seconds: float,
    summary: dict[str, Any],
) -> dict[str, str]:
    reference_path = Path(f"{path}.references.mrcs")
    result_path = Path(f"{path}.result.npz")
    report_path = Path(f"{path}.report.json")
    with mrcfile.new(reference_path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(result.references, dtype=np.float32))
        if pixel_size is not None:
            mrc.voxel_size = pixel_size
        mrc.update_header_stats()
    history = (
        np.asarray(result.reference_history, dtype=np.float32)
        if result.reference_history
        else np.asarray(result.references[None], dtype=np.float32)
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
        bootstrap_labels=result.metadata["bootstrap_labels"],
    )
    write_json(
        report_path,
        {
            "schema": "alignimg.re2dc-70s-rf-seed.v1",
            "status": "completed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "input": {
                "stack": str(source_stack),
                "particle_count": len(result.poses),
                "image_shape": list(result.references.shape[1:]),
                "pixel_size_angstrom": pixel_size,
                "reference_count": len(result.references),
            },
            "spec": {
                "workflow": "reference_free",
                "backend": backend,
                "n_components": len(result.references),
                "random_seed": seed,
                "config": asdict(config),
            },
            "summary": summary,
            "diagnostics": result.diagnostics,
            "metadata": result.metadata,
            "artifacts": {
                "references": reference_path.name,
                "result": result_path.name,
            },
        },
    )
    return {
        "references": str(reference_path),
        "result": str(result_path),
        "gui_report": str(report_path),
    }


def main(args: argparse.Namespace) -> None:
    with mrcfile.mmap(args.stack, permissive=True, mode="r") as mrc:
        images = np.asarray(mrc.data, dtype=np.float32).copy()
        pixel_size_value = float(mrc.voxel_size.x)
    pixel_size = pixel_size_value if pixel_size_value > 0.0 else None
    if images.ndim != 3 or images.shape[1:] != (128, 128):
        raise ValueError(f"expected a (N, 128, 128) stack, got {images.shape}")
    config_values = asdict(ai.AlignmentConfig.preset("reference_free"))
    config_values.update(
        max_iterations=args.iterations,
        temperature_anneal_iterations=args.anneal_iterations,
        angle_samples=args.angle_samples,
        translation_range=args.translation_range,
        candidate_scoring=getattr(args, "candidate_scoring", "raster"),
        score_model=getattr(args, "score_model", "fourier_ncc"),
        reference_update=getattr(args, "reference_update", "spatial"),
        batch_size=args.batch_size,
        memory_fraction=args.memory_fraction,
        halfset_diagnostics=True,
    )
    for name in (
        "top_l",
        "proposal_angles_per_reference",
        "temperature_start",
        "temperature_end",
    ):
        value = getattr(args, name)
        if value is not None:
            config_values[name] = value
    report: dict[str, Any] = {
        "schema": "alignimg.re2dc-70s-rf-validation.v1",
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "alignimg_version": ai.__version__,
            "backends": ai.available_alignment_backends(),
        },
        "input": {"stack": str(args.stack), "shape": list(images.shape)},
        "scope_note": (
            "Components are alignment/view components, not validated biological classes. "
            "Occupancy imbalance and attraction are reported but are not repaired here."
        ),
        "parameters": {
            "backend": args.backend,
            "component_count": args.components,
            "seeds": args.seeds,
            "config": dict(config_values),
            "minimum_frc_halfset_weight": args.minimum_frc_halfset_weight,
        },
        "runs": {},
    }
    write_json(args.output, report)
    results: list[ai.AlignmentResult] = []
    for seed in args.seeds:
        print(f"RUN  reference_free seed={seed}", flush=True)
        run_config_values = dict(config_values)
        run_config_values["random_seed"] = seed
        config = ai.AlignmentConfig(**run_config_values)
        started = time.perf_counter()
        result = ai.reference_free_align(
            images,
            n_components=args.components,
            config=config,
            backend=args.backend,
        )
        seconds = time.perf_counter() - started
        results.append(result)
        run_path = args.output.with_suffix(f".seed-{seed}")
        metrics = run_metrics(
            result,
            seconds,
            args.components,
            args.minimum_frc_halfset_weight,
            pixel_size,
        )
        compact_summary = {
            key: metrics[key]
            for key in (
                "seconds",
                "particles_per_second",
                "particle_iterations_per_second",
                "hard_assignment_counts",
                "mean_max_responsibility",
                "effective_component_weight",
                "reliable_median_stable_frc_resolution_angstrom",
            )
        }
        metrics["artifacts"] = save_run(
            run_path,
            result,
            pixel_size=pixel_size,
            source_stack=args.stack,
            backend=args.backend,
            config=config,
            seed=seed,
            seconds=seconds,
            summary=compact_summary,
        )
        report["runs"][str(seed)] = metrics
        write_json(args.output, report)
        print(f"DONE seed={seed}: {seconds:.3f} s", flush=True)

    if len(results) >= 2:
        baseline = results[0]
        baseline_profiles = radial_profiles(baseline.references)
        comparisons: dict[str, Any] = {}
        for seed, result in zip(args.seeds[1:], results[1:]):
            similarity = baseline_profiles @ radial_profiles(result.references).T
            rows, columns = linear_sum_assignment(-similarity)
            comparisons[str(seed)] = {
                "against_seed": args.seeds[0],
                "bootstrap_adjusted_rand_index": adjusted_rand_index(
                    baseline.metadata["bootstrap_labels"],
                    result.metadata["bootstrap_labels"],
                ),
                "final_assignment_adjusted_rand_index": adjusted_rand_index(
                    baseline.reference_assignments, result.reference_assignments
                ),
                "radial_profile_reference_matching": {
                    "baseline_indices": rows,
                    "comparison_indices": columns,
                    "matched_correlations": similarity[rows, columns],
                    "mean_matched_correlation": float(np.mean(similarity[rows, columns])),
                },
            }
        report["cross_seed_comparisons"] = comparisons
    report["status"] = "completed"
    write_json(args.output, report)
    print(f"Report saved to: {args.output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", type=Path, default=DEFAULT_STACK)
    parser.add_argument("--output", type=Path, default=Path("validation-results/re2dc-70s-rf.json"))
    parser.add_argument("--backend", choices=("cpu", "cuda", "cupy", "gpu", "auto"), default="cuda")
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--anneal-iterations", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--angle-samples", type=int, default=128)
    parser.add_argument(
        "--candidate-scoring", choices=("raster", "fourier"), default="raster"
    )
    parser.add_argument(
        "--score-model",
        choices=("fourier_ncc", "whitened_fourier_ncc"),
        default="fourier_ncc",
    )
    parser.add_argument(
        "--reference-update", choices=("spatial", "fourier"), default="spatial"
    )
    parser.add_argument("--top-l", type=int)
    parser.add_argument("--proposal-angles-per-reference", type=int)
    parser.add_argument("--temperature-start", type=float)
    parser.add_argument("--temperature-end", type=float)
    parser.add_argument("--translation-range", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--minimum-frc-halfset-weight", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
