#!/usr/bin/env python3
"""A/B validate adaptive and continuous quadratic refinement on RELION classes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

import alignimg as ai

try:
    from tools.re2dc_70s_pose_benchmark import (
        DEFAULT_BENCHMARK,
        compose_pose_gauge,
        environment_info,
        fit_pose_gauge,
        fixed_mra_k1_equivalence,
        image_correlation,
        load_benchmark,
        reproducibility_metrics,
        subset_poses,
        wrapped_angle,
    )
    from tools.server_validation import jsonable
except ModuleNotFoundError:
    from re2dc_70s_pose_benchmark import (
        DEFAULT_BENCHMARK,
        compose_pose_gauge,
        environment_info,
        fit_pose_gauge,
        fixed_mra_k1_equivalence,
        image_correlation,
        load_benchmark,
        reproducibility_metrics,
        subset_poses,
        wrapped_angle,
    )
    from server_validation import jsonable


DEFAULT_OUTPUT = Path(
    "validation-results/performance/quadratic-refine/dev4-relion-ab.json"
)
RELION_ANGLE_SAMPLING_STEP_DEG = 5.625
RELION_ANGLE_HALF_STEP_DEG = RELION_ANGLE_SAMPLING_STEP_DEG / 2.0


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def timed(callable_):
    started = time.perf_counter()
    result = callable_()
    return result, time.perf_counter() - started


def pose_error_arrays(
    predicted: ai.PoseSet, truth: ai.PoseSet
) -> tuple[np.ndarray, np.ndarray, ai.PoseSet]:
    gauge = fit_pose_gauge(predicted, truth)
    expected = compose_pose_gauge(truth, gauge)
    angle = np.abs(wrapped_angle(predicted.angle_deg - expected.angle_deg))
    shift = np.hypot(
        predicted.shift_y_px - expected.shift_y_px,
        predicted.shift_x_px - expected.shift_x_px,
    )
    return angle, shift, gauge


def quantiles(values: np.ndarray) -> dict[str, float]:
    result = np.quantile(np.asarray(values, dtype=np.float64), (0.5, 0.95, 1.0))
    return dict(zip(("median", "p95", "maximum"), result, strict=True))


def relion_angle_lattice_metrics(
    angle_deg: np.ndarray,
    *,
    sampling_step_deg: float = RELION_ANGLE_SAMPLING_STEP_DEG,
) -> dict[str, float]:
    """Measure the common phase and residual of a RELION angle lattice."""
    angles = np.asarray(angle_deg, dtype=np.float64)
    step = float(sampling_step_deg)
    phase = float(
        np.angle(np.mean(np.exp(2j * np.pi * angles / step)))
        * step
        / (2.0 * np.pi)
    )
    residual = (angles - phase + 0.5 * step) % step - 0.5 * step
    return {
        "sampling_step_deg": step,
        "half_step_deg": 0.5 * step,
        "phase_deg": phase,
        "maximum_absolute_residual_deg": float(np.max(np.abs(residual))),
    }


def strategy_config(
    strategy: str,
    *,
    iterations: int,
    batch_size: int,
    memory_fraction: float,
    profile_execution: bool,
) -> ai.AlignmentConfig:
    common = dict(
        max_iterations=iterations,
        top_l=8,
        candidate_scoring="fourier",
        score_model="fourier_ncc",
        reference_update="fourier",
        temperature_start=0.08,
        temperature_end=0.05,
        pose_angle_sigma=5.0,
        pose_shift_sigma=2.0,
        robust_weighting=True,
        halfset_diagnostics=True,
        center_references=False,
        batch_size=batch_size,
        memory_fraction=memory_fraction,
        profile_execution=profile_execution,
    )
    if strategy == "adaptive_posterior":
        return ai.AlignmentConfig(
            search_strategy=strategy,
            local_angle_range=15.0,
            coarse_angle_step=6.0,
            local_shift_range=3.0,
            coarse_shift_step=1.0,
            adaptive_fraction=0.999,
            oversampling_order=1,
            **common,
        ).normalized(workflow="refine")
    return ai.AlignmentConfig(
        search_strategy="quadratic_refine",
        local_angle_range=7.0,
        coarse_angle_step=1.0,
        local_shift_range=3.0,
        **common,
    ).normalized(workflow="refine")


def make_initial_state(
    particles: np.ndarray,
    references: np.ndarray,
    component: np.ndarray,
    *,
    backend: str,
    batch_size: int,
    memory_fraction: float,
) -> tuple[np.ndarray, ai.PoseSet, list[dict[str, Any]]]:
    angle = np.empty(len(particles), dtype=np.float32)
    shift_y = np.empty(len(particles), dtype=np.float32)
    shift_x = np.empty(len(particles), dtype=np.float32)
    mirror = np.zeros(len(particles), dtype=np.bool_)
    initial_references = np.empty_like(references)
    records = []
    config = ai.AlignmentConfig(
        max_iterations=1,
        top_l=8,
        angle_samples=128,
        proposal_angles_per_reference=8,
        translation_range=4.0,
        candidate_scoring="fourier",
        reference_update="fourier",
        halfset_diagnostics=False,
        center_references=False,
        batch_size=batch_size,
        memory_fraction=memory_fraction,
    )
    for index in range(len(references)):
        selected = np.flatnonzero(component == index)
        result, seconds = timed(
            lambda selected=selected, index=index: ai.align_to_references(
                particles[selected],
                references[index],
                config=config,
                backend=backend,
            )
        )
        angle[selected] = result.poses.angle_deg
        shift_y[selected] = result.poses.shift_y_px
        shift_x[selected] = result.poses.shift_x_px
        mirror[selected] = result.poses.mirror
        initial_references[index] = result.references[0]
        records.append(
            {
                "component_index": index,
                "particle_count": len(selected),
                "seconds": seconds,
                "metadata": result.metadata,
            }
        )
    return (
        initial_references,
        ai.PoseSet(angle, shift_y, shift_x, mirror),
        records,
    )


def component_metrics(
    result: ai.AlignmentResult,
    particles: np.ndarray,
    oracle_references: np.ndarray,
    truth_poses: ai.PoseSet,
    component: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    angle_chunks = []
    shift_chunks = []
    per_component = []
    for index in range(len(oracle_references)):
        selected = np.flatnonzero(component == index)
        predicted = subset_poses(result.poses, selected)
        truth = subset_poses(truth_poses, selected)
        angle, shift, gauge = pose_error_arrays(predicted, truth)
        gauged_reference = ai.transform_images(
            oracle_references[index][None], gauge, backend="cpu"
        )[0]
        angle_chunks.append(angle)
        shift_chunks.append(shift)
        per_component.append(
            {
                "component_index": index,
                "particle_count": len(selected),
                "angle_error_deg": quantiles(angle),
                "shift_error_px": quantiles(shift),
                "result_reference_correlation": image_correlation(
                    result.references[index], gauged_reference
                ),
            }
        )
    all_angle = np.concatenate(angle_chunks)
    all_shift = np.concatenate(shift_chunks)
    return (
        {
            "angle_error_deg": quantiles(all_angle),
            "shift_error_px": quantiles(all_shift),
            "per_component": per_component,
            "diagnostics": result.diagnostics,
            "metadata": result.metadata,
        },
        all_angle,
        all_shift,
    )


def pose_accuracy_metrics(
    predicted: ai.PoseSet,
    truth: ai.PoseSet,
    component: np.ndarray,
    component_count: int,
) -> dict[str, Any]:
    """Summarize pose errors without requiring an AlignmentResult."""
    angle_chunks = []
    shift_chunks = []
    per_component = []
    for index in range(component_count):
        selected = np.flatnonzero(component == index)
        angle, shift, _ = pose_error_arrays(
            subset_poses(predicted, selected), subset_poses(truth, selected)
        )
        angle_chunks.append(angle)
        shift_chunks.append(shift)
        per_component.append(
            {
                "component_index": index,
                "particle_count": len(selected),
                "angle_error_deg": quantiles(angle),
                "shift_error_px": quantiles(shift),
            }
        )
    return {
        "angle_error_deg": quantiles(np.concatenate(angle_chunks)),
        "shift_error_px": quantiles(np.concatenate(shift_chunks)),
        "per_component": per_component,
    }


def accuracy_gate(
    adaptive_metrics: dict[str, Any],
    quadratic_metrics: dict[str, Any],
    initial_metrics: dict[str, Any],
    *,
    angle_half_step_deg: float = RELION_ANGLE_HALF_STEP_DEG,
) -> dict[str, Any]:
    adaptive_angle = adaptive_metrics["angle_error_deg"]
    adaptive_shift = adaptive_metrics["shift_error_px"]
    quadratic_angle = quadratic_metrics["angle_error_deg"]
    quadratic_shift = quadratic_metrics["shift_error_px"]
    correlations_ok = all(
        current["result_reference_correlation"]
        >= baseline["result_reference_correlation"] - 0.005
        for baseline, current in zip(
            adaptive_metrics["per_component"],
            quadratic_metrics["per_component"],
            strict=True,
        )
    )
    component_angles_improved = all(
        current["angle_error_deg"]["median"]
        < initial["angle_error_deg"]["median"]
        for initial, current in zip(
            initial_metrics["per_component"],
            quadratic_metrics["per_component"],
            strict=True,
        )
    )
    checks = {
        "component_median_angles_improved_from_initial": (
            component_angles_improved
        ),
        "median_angle_within_relion_half_step_of_adaptive": (
            quadratic_angle["median"]
            <= adaptive_angle["median"] + float(angle_half_step_deg)
        ),
        "median_shift_strictly_improved": (
            quadratic_shift["median"] < adaptive_shift["median"]
        ),
        "angle_p95_within_10_percent": (
            quadratic_angle["p95"] <= 1.1 * adaptive_angle["p95"]
        ),
        "shift_p95_within_10_percent": (
            quadratic_shift["p95"] <= 1.1 * adaptive_shift["p95"]
        ),
        "component_reference_correlations_within_0_005": correlations_ok,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "criteria": {
            "relion_angle_half_step_deg": float(angle_half_step_deg),
        },
        "observations": {
            "raw_median_angle_strictly_improved": (
                quadratic_angle["median"] < adaptive_angle["median"]
            ),
            "adaptive_median_angle_error_deg": adaptive_angle["median"],
            "quadratic_median_angle_error_deg": quadratic_angle["median"],
        },
    }


def run_strategy(
    strategy: str,
    particles: np.ndarray,
    references: np.ndarray,
    initial_poses: ai.PoseSet,
    truth_poses: ai.PoseSet,
    component: np.ndarray,
    oracle_references: np.ndarray,
    *,
    backend: str,
    config: ai.AlignmentConfig,
    class_priors: np.ndarray | None,
    deterministic_repeats: int,
) -> tuple[ai.AlignmentResult, dict[str, Any]]:
    call = lambda: ai.refine_alignment(
        particles,
        references,
        initial_poses,
        class_priors=class_priors,
        config=config,
        backend=backend,
    )
    result, seconds = timed(call)
    if result.metadata["backend"] != backend:
        raise AssertionError(f"{strategy} unexpectedly fell back from {backend}")
    if (
        strategy == "quadratic_refine"
        and backend == "cuda"
        and result.metadata.get("quadratic_peak_backend") != "native_cuda"
    ):
        raise AssertionError("quadratic CUDA run did not use the native peak kernel")
    metrics, _, _ = component_metrics(
        result, particles, oracle_references, truth_poses, component
    )
    metrics["seconds"] = seconds
    metrics["seconds_per_iteration"] = seconds / config.max_iterations
    metrics["particle_iterations_per_second"] = (
        len(particles) * config.max_iterations / max(seconds, 1e-12)
    )
    metrics["config"] = asdict(config)
    repeats = []
    for repeat_index in range(1, deterministic_repeats):
        repeated, repeated_seconds = timed(call)
        repeats.append(
            {
                "repeat_index": repeat_index,
                "seconds": repeated_seconds,
                "comparison": reproducibility_metrics(result, repeated),
            }
        )
    metrics["reproducibility_repeats"] = repeats
    metrics["reproducible"] = all(
        item["comparison"]["assignment_agreement"] == 1.0
        and item["comparison"]["mirror_agreement"] == 1.0
        and item["comparison"]["angle_absolute_delta_deg"]["maximum"] <= 1e-6
        and item["comparison"]["shift_delta_px"]["maximum"] <= 1e-6
        and item["comparison"]["responsibility_max_absolute_delta"] <= 2e-7
        and np.min(item["comparison"]["reference_correlations"]) >= 0.999999
        for item in repeats
    )
    return result, metrics


def main(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {
        "schema": "alignimg.quadratic-refine-ab.v2",
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_info(),
        "parameters": vars(args),
        "runs": {},
        "gates": {},
    }
    write_json(args.output, report)
    try:
        manifest, particles, oracle_references, truth, truth_poses = load_benchmark(
            args.benchmark
        )
        component = np.asarray(truth["component_index"], dtype=np.int32)
        initial_references, initial_poses, initializer_records = make_initial_state(
            particles,
            oracle_references,
            component,
            backend=args.backend,
            batch_size=args.batch_size,
            memory_fraction=args.memory_fraction,
        )
        lattice = relion_angle_lattice_metrics(truth_poses.angle_deg)
        if lattice["maximum_absolute_residual_deg"] > 1e-3:
            raise AssertionError(
                "RELION angle truth no longer matches the documented 5.625-degree "
                "sampling lattice"
            )
        initial_pose_accuracy = pose_accuracy_metrics(
            initial_poses,
            truth_poses,
            component,
            len(oracle_references),
        )
        input_path = args.output.with_suffix("").with_suffix(".inputs.npz")
        np.savez_compressed(
            input_path,
            references=initial_references,
            angle_deg=initial_poses.angle_deg,
            shift_y_px=initial_poses.shift_y_px,
            shift_x_px=initial_poses.shift_x_px,
            mirror=initial_poses.mirror,
            component=component,
        )
        report["inputs"] = {
            "benchmark": str(args.benchmark),
            "benchmark_manifest": manifest,
            "particle_count": len(particles),
            "component_count": len(oracle_references),
            "initializer": initializer_records,
            "initial_pose_accuracy": initial_pose_accuracy,
            "relion_angle_lattice": lattice,
            "frozen_input_npz": str(input_path),
            "hashes": {
                "particles": array_sha256(particles),
                "references": array_sha256(initial_references),
                "initial_angle_deg": array_sha256(initial_poses.angle_deg),
                "initial_shift_y_px": array_sha256(initial_poses.shift_y_px),
                "initial_shift_x_px": array_sha256(initial_poses.shift_x_px),
                "component": array_sha256(component),
            },
        }
        write_json(args.output, report)

        configs = {
            strategy: strategy_config(
                strategy,
                iterations=args.iterations,
                batch_size=args.batch_size,
                memory_fraction=args.memory_fraction,
                profile_execution=args.profile_execution,
            )
            for strategy in ("adaptive_posterior", "quadratic_refine")
        }
        per_class_results: dict[str, list[ai.AlignmentResult]] = {
            strategy: [] for strategy in configs
        }
        k1_runs: dict[str, list[dict[str, Any]]] = {
            strategy: [] for strategy in configs
        }
        for strategy, config in configs.items():
            for index in range(len(oracle_references)):
                print(f"RUN  {strategy} K=1 component={index}", flush=True)
                selected = np.flatnonzero(component == index)
                local_component = np.zeros(len(selected), dtype=np.int32)
                result, metrics = run_strategy(
                    strategy,
                    particles[selected],
                    initial_references[index][None],
                    subset_poses(initial_poses, selected),
                    subset_poses(truth_poses, selected),
                    local_component,
                    oracle_references[index][None],
                    backend=args.backend,
                    config=config,
                    class_priors=None,
                    deterministic_repeats=args.deterministic_repeats,
                )
                per_class_results[strategy].append(result)
                metrics["component_index"] = index
                k1_runs[strategy].append(metrics)
                report["runs"]["k1"] = k1_runs
                write_json(args.output, report)

        def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
            angle = np.concatenate(
                [
                    np.asarray(
                        [
                            run["per_component"][0]["angle_error_deg"]["median"]
                        ]
                    )
                    for run in items
                ]
            )
            shift = np.asarray(
                [run["per_component"][0]["shift_error_px"]["median"] for run in items]
            )
            return {
                "median_of_component_median_angle_error_deg": float(np.median(angle)),
                "median_of_component_median_shift_error_px": float(np.median(shift)),
                "seconds": float(sum(run["seconds"] for run in items)),
            }

        report["runs"]["k1_summary"] = {
            strategy: aggregate(k1_runs[strategy]) for strategy in configs
        }
        k1_gate_metrics = {}
        for strategy in configs:
            angle_chunks = []
            shift_chunks = []
            per_component = []
            for index, result in enumerate(per_class_results[strategy]):
                selected = np.flatnonzero(component == index)
                angle, shift, _ = pose_error_arrays(
                    result.poses, subset_poses(truth_poses, selected)
                )
                angle_chunks.append(angle)
                shift_chunks.append(shift)
                per_component.append(k1_runs[strategy][index]["per_component"][0])
            k1_gate_metrics[strategy] = {
                "angle_error_deg": quantiles(np.concatenate(angle_chunks)),
                "shift_error_px": quantiles(np.concatenate(shift_chunks)),
                "per_component": per_component,
            }
        report["gates"]["k1_accuracy"] = accuracy_gate(
            k1_gate_metrics["adaptive_posterior"],
            k1_gate_metrics["quadratic_refine"],
            initial_pose_accuracy,
            angle_half_step_deg=lattice["half_step_deg"],
        )
        report["gates"]["k1_reproducibility"] = {
            "passed": all(
                run["reproducible"] for run in k1_runs["quadratic_refine"]
            )
        }
        write_json(args.output, report)

        if args.suite in {"fixed_mra", "all"}:
            priors = ai.make_class_priors(
                assignments=component, n_components=len(oracle_references), trust=1.0
            )
            joint_results = {}
            joint_metrics = {}
            for strategy, config in configs.items():
                print(f"RUN  {strategy} fixed K={len(oracle_references)}", flush=True)
                result, metrics = run_strategy(
                    strategy,
                    particles,
                    initial_references,
                    initial_poses,
                    truth_poses,
                    component,
                    oracle_references,
                    backend=args.backend,
                    config=config,
                    class_priors=priors,
                    deterministic_repeats=args.deterministic_repeats,
                )
                if not np.array_equal(result.reference_assignments, component):
                    raise AssertionError("fixed MRA changed class assignments")
                joint_results[strategy] = result
                joint_metrics[strategy] = metrics
                report["runs"]["fixed_mra"] = joint_metrics
                write_json(args.output, report)
            equivalence = fixed_mra_k1_equivalence(
                joint_results["quadratic_refine"],
                per_class_results["quadratic_refine"],
                component,
            )
            joint_metrics["quadratic_refine"]["k1_equivalence"] = equivalence
            equivalence_checks = {
                "maximum_angle_delta_within_1e_3_deg": all(
                    item["maximum_angle_delta_deg"] <= 1e-3
                    for item in equivalence
                ),
                "maximum_shift_delta_within_1e_3_px": all(
                    item["maximum_shift_delta_px"] <= 1e-3
                    for item in equivalence
                ),
                "maximum_responsibility_delta_within_2e_4": all(
                    item["maximum_responsibility_delta"] <= 2e-4
                    for item in equivalence
                ),
                "matched_reference_correlation_at_least_0_9999": all(
                    item["reference_correlation"] >= 0.9999
                    for item in equivalence
                ),
            }
            report["gates"]["fixed_mra_accuracy"] = accuracy_gate(
                joint_metrics["adaptive_posterior"],
                joint_metrics["quadratic_refine"],
                initial_pose_accuracy,
                angle_half_step_deg=lattice["half_step_deg"],
            )
            report["gates"]["fixed_mra_k1_equivalence"] = {
                "passed": all(equivalence_checks.values()),
                "checks": equivalence_checks,
                "per_component": equivalence,
            }
            report["gates"]["fixed_mra_reproducibility"] = {
                "passed": joint_metrics["quadratic_refine"]["reproducible"]
            }

        report["status"] = (
            "completed"
            if all(item.get("passed", False) for item in report["gates"].values())
            else "completed_with_failed_gates"
        )
        write_json(args.output, report)
        print(f"Report saved to: {args.output.resolve()}", flush=True)
        return 0 if report["status"] == "completed" else 1
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
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--suite", choices=("k1", "fixed_mra", "all"), default="k1")
    parser.add_argument(
        "--backend", choices=("cpu", "cuda", "cupy"), default="cuda"
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--deterministic-repeats", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--profile-execution", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
