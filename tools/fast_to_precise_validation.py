#!/usr/bin/env python3
"""Compare equal-iteration balanced, fast-hard, and fast-to-precise schedules."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import time
import traceback
from typing import Any, Callable

import numpy as np

import alignimg as ai

try:
    from tools.fast_hard_validation import (
        ROOT,
        _case_hashes,
        _check_hard_result,
        _check_result,
        _json_hash,
        _pose_quality,
        freeze_inputs,
        git_revision,
        polar_result_arrays,
        result_hashes,
        runtime_identity,
        sha256,
        source_manifest,
    )
    from tools.performance_validation import compare_arrays, synchronize
    from tools.server_validation import environment_info, utc_now, write_report
except ModuleNotFoundError:
    from fast_hard_validation import (
        ROOT,
        _case_hashes,
        _check_hard_result,
        _check_result,
        _json_hash,
        _pose_quality,
        freeze_inputs,
        git_revision,
        polar_result_arrays,
        result_hashes,
        runtime_identity,
        sha256,
        source_manifest,
    )
    from performance_validation import compare_arrays, synchronize
    from server_validation import environment_info, utc_now, write_report


SCHEMA = "alignimg.fast-to-precise-validation.v2"
SUITES = ("synthetic", "homogeneous", "mra1000", "representative")
FROZEN_INPUTS = ROOT / "validation-results/fast-hard/stage-0/frozen-inputs"
REPRESENTATIVE_CASES = {
    "synthetic": ("k3_noiseless", "k3_snr_0_2"),
    "homogeneous": ("fixed_k3",),
    "mra1000": ("open_k10",),
}


def schedule_configs(
    *,
    global_iterations: int,
    precise_iterations: int,
    batch_size: int,
    memory_fraction: float,
    mirror_search: bool = False,
    profile_execution: bool = False,
) -> dict[str, tuple[ai.AlignmentConfig, ...]]:
    """Build schedules with equal total inference iterations."""
    total_iterations = global_iterations + precise_iterations
    common = dict(
        batch_size=batch_size,
        memory_fraction=memory_fraction,
        mirror_search=mirror_search,
        profile_execution=profile_execution,
    )
    balanced = replace(
        ai.AlignmentConfig.preset("global_balanced"),
        max_iterations=total_iterations,
        halfset_diagnostics=True,
        apply_final_pose_to_raw=True,
        **common,
    ).normalized(workflow="global")
    fast_only = replace(
        ai.AlignmentConfig.preset("fast_hard"),
        max_iterations=total_iterations,
        apply_final_pose_to_raw=True,
        **common,
    ).normalized(workflow="global")
    fast_global = replace(
        ai.AlignmentConfig.preset("fast_hard"),
        max_iterations=global_iterations,
        apply_final_pose_to_raw=False,
        **common,
    ).normalized(workflow="global")
    precise = replace(
        ai.AlignmentConfig.preset("refine"),
        max_iterations=precise_iterations,
        halfset_diagnostics=True,
        apply_final_pose_to_raw=True,
        **common,
    ).normalized(workflow="refine")
    return {
        "balanced_soft": (balanced,),
        "fast_only": (fast_only,),
        "fast_to_precise": (fast_global, precise),
    }


def _execute_schedule(
    case: dict[str, np.ndarray],
    configs: tuple[ai.AlignmentConfig, ...],
    backend: str,
) -> tuple[ai.AlignmentResult, list[ai.AlignmentResult]]:
    priors = case["priors"] if case["priors"].size else None
    global_result = ai.align_to_references(
        case["images"],
        case["references"],
        class_priors=priors,
        config=configs[0],
        backend=backend,
    )
    _check_result(global_result, backend, case)
    if configs[0].search_strategy == "polar_hard":
        _check_hard_result(global_result)
    if len(configs) == 1:
        return global_result, [global_result]

    fixed_priors = ai.make_class_priors(
        assignments=global_result.reference_assignments,
        n_components=len(global_result.references),
        trust=1.0,
    )
    final = ai.refine_alignment(
        case["images"],
        global_result.references,
        global_result.poses,
        class_priors=fixed_priors,
        config=configs[1],
        backend=backend,
    )
    _check_result(final, backend, {**case, "priors": fixed_priors})
    if not np.array_equal(
        final.reference_assignments, global_result.reference_assignments
    ):
        raise AssertionError("fixed-class precise refinement changed assignments")
    return final, [global_result, final]


def _quality_summary(
    result: ai.AlignmentResult,
    case: dict[str, np.ndarray],
    backend: str,
) -> dict[str, Any] | None:
    """Report truth-based quality when the frozen workload provides truth."""
    quality = _pose_quality(result, case, backend)
    if quality is None:
        return None
    component = np.asarray(case["component"])
    quality["assignment_accuracy"] = float(
        np.mean(result.reference_assignments == component)
    )
    return quality


def _stage_summary(
    results: list[ai.AlignmentResult],
    case: dict[str, np.ndarray],
    backend: str,
) -> list[dict[str, Any]]:
    return [
        {
            "search_strategy": result.metadata["search_strategy"],
            "iterations": len(result.diagnostics),
            "quality": _quality_summary(result, case, backend),
            "occupancy": np.bincount(
                result.reference_assignments, minlength=len(result.references)
            ),
            "iteration_seconds": [
                float(item["seconds"]) for item in result.diagnostics
            ],
            "candidate_inference_seconds": [
                float(item["candidate_inference_seconds"])
                for item in result.diagnostics
            ],
            "reference_update_seconds": [
                float(item["reference_update_seconds"]) for item in result.diagnostics
            ],
            "centering_seconds": [
                float(item["centering_seconds"]) for item in result.diagnostics
            ],
            "frc_diagnostics_seconds": [
                float(item["frc_diagnostics_seconds"]) for item in result.diagnostics
            ],
            "raw_average_seconds": float(result.metadata["final_raw_average_seconds"]),
        }
        for result in results
    ]


def _resource_summary(result: ai.AlignmentResult) -> dict[str, Any]:
    return {
        name: result.metadata.get(name)
        for name in (
            "engine",
            "backend",
            "gpu_device",
            "gpu_memory_plan_at_completion",
            "gpu_memory_plans",
            "gpu_workspace",
            "polar_sampler_backend",
            "polar_peak_backend",
            "polar_full_correlation_map_d2h",
            "gpu_policy",
        )
    }


def run_schedule(
    qualified_name: str,
    case: dict[str, np.ndarray],
    *,
    schedule: str,
    configs: tuple[ai.AlignmentConfig, ...],
    backend: str,
    measured_repeats: int,
    profile_execution: bool,
    output: Path,
    stage_analysis: Callable[[list[ai.AlignmentResult]], Any] | None = None,
) -> dict[str, Any]:
    synchronize(backend)
    started = time.perf_counter()
    warmup, _ = _execute_schedule(case, configs, backend)
    synchronize(backend)
    warmup_seconds = time.perf_counter() - started

    measured: list[ai.AlignmentResult] = []
    measured_stages: list[list[ai.AlignmentResult]] = []
    seconds = []
    for _ in range(measured_repeats):
        synchronize(backend)
        started = time.perf_counter()
        result, stages = _execute_schedule(case, configs, backend)
        synchronize(backend)
        seconds.append(time.perf_counter() - started)
        measured.append(result)
        measured_stages.append(stages)

    primary = measured[0]
    hashes = [result_hashes(result) for result in measured]
    repeat_parity = [
        compare_arrays(polar_result_arrays(primary), polar_result_arrays(result))
        for result in measured[1:]
    ]

    profiled = None
    profiled_stages = None
    profile_parity = None
    if profile_execution:
        profile_configs = tuple(
            replace(config, profile_execution=True) for config in configs
        )
        synchronize(backend)
        profiled, profiled_stages = _execute_schedule(case, profile_configs, backend)
        synchronize(backend)
        profile_parity = compare_arrays(
            polar_result_arrays(primary), polar_result_arrays(profiled)
        )

    safe_name = qualified_name.replace(":", "-")
    result_path = output.with_suffix("").with_name(
        f"{output.stem}.{safe_name}.{schedule}.result.npz"
    )
    if result_path.exists():
        raise FileExistsError(f"result already exists: {result_path}")
    np.savez_compressed(result_path, **polar_result_arrays(primary))
    occupancy = np.bincount(
        primary.reference_assignments, minlength=len(primary.references)
    )
    first_stage = measured_stages[0][0]
    return {
        "schedule": schedule,
        "configs": [asdict(config) for config in configs],
        "config_sha256": _json_hash([asdict(config) for config in configs]),
        "total_iterations": sum(config.max_iterations for config in configs),
        "warmup_seconds": warmup_seconds,
        "unprofiled_wall_seconds": seconds,
        "unprofiled_median_seconds": float(np.median(seconds)),
        "unprofiled_stage_timings": _stage_summary(measured_stages[0], case, backend),
        "unprofiled_stage_analysis": (
            None if stage_analysis is None else stage_analysis(measured_stages[0])
        ),
        "deterministic_hashes": hashes,
        "deterministic_exact_match": len({item["combined"] for item in hashes}) == 1,
        "repeat_parity": repeat_parity,
        "profile_parity": profile_parity,
        "profiled_stages": (
            None
            if profiled_stages is None
            else _stage_summary(profiled_stages, case, backend)
        ),
        "profiled_performance": (
            None
            if profiled_stages is None
            else [stage.metadata.get("performance") for stage in profiled_stages]
        ),
        "input_sha256": _case_hashes(case),
        "responsibility_row_sum_max_error": float(
            np.max(np.abs(primary.responsibilities.sum(axis=1) - 1.0))
        ),
        "assignments": primary.reference_assignments,
        "occupancy": occupancy,
        "quality": _quality_summary(primary, case, backend),
        "fixed_assignment_preserved": (
            True
            if len(measured_stages[0]) == 1
            else np.array_equal(
                primary.reference_assignments,
                first_stage.reference_assignments,
            )
        ),
        "resources": [_resource_summary(stage) for stage in measured_stages[0]],
        "result": str(result_path),
        "result_sha256": sha256(result_path),
    }


def compare_schedules(schedules: dict[str, dict[str, Any]]) -> dict[str, Any]:
    balanced = schedules["balanced_soft"]
    return {
        name: {
            "wall_speedup_vs_balanced": float(
                balanced["unprofiled_median_seconds"]
                / value["unprofiled_median_seconds"]
            ),
            "same_total_iterations": (
                value["total_iterations"] == balanced["total_iterations"]
            ),
        }
        for name, value in schedules.items()
        if name != "balanced_soft"
    }


def load_frozen_cases(
    suite: str, inputs_dir: Path, inputs: Path | None = None
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    suites = REPRESENTATIVE_CASES if suite == "representative" else {suite: None}
    cases: dict[str, dict[str, np.ndarray]] = {}
    manifests = {}
    for source_suite, selected_names in suites.items():
        path = (
            inputs if inputs is not None else inputs_dir / f"{source_suite}.inputs.npz"
        )
        if path is None or not path.exists():
            raise FileNotFoundError(f"frozen input does not exist: {path}")
        frozen = freeze_inputs(source_suite, path)
        selected = set(frozen["cases"] if selected_names is None else selected_names)
        missing = selected.difference(frozen["cases"])
        if missing:
            raise ValueError(f"missing frozen {source_suite} cases: {sorted(missing)}")
        for name, case in frozen["cases"].items():
            if name in selected:
                cases[f"{source_suite}:{name}"] = case
        manifests[source_suite] = {
            name: value for name, value in frozen.items() if name != "cases"
        }
    return cases, manifests


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "cupy"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--global-iterations", type=int, default=3)
    parser.add_argument("--precise-iterations", type=int, default=2)
    parser.add_argument("--deterministic-repeats", type=int, default=2)
    parser.add_argument("--profile-execution", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--inputs-dir", type=Path, default=FROZEN_INPUTS)
    parser.add_argument(
        "--only",
        help="Comma-separated suite:case names; single suites also accept case",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
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
    if args.suite == "representative" and args.inputs is not None:
        parser.error("--inputs cannot be used with the representative suite")
    if args.output.exists():
        parser.error("output already exists; reports are immutable")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases, manifests = load_frozen_cases(args.suite, args.inputs_dir, args.inputs)
    if args.only:
        requested = set(args.only.split(","))
        if args.suite != "representative":
            requested = {
                name if ":" in name else f"{args.suite}:{name}" for name in requested
            }
        unknown = requested.difference(cases)
        if unknown:
            raise ValueError(f"unknown cases: {sorted(unknown)}")
        cases = {name: case for name, case in cases.items() if name in requested}

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "stage": 5,
        "status": "running",
        "started_at_utc": utc_now(),
        "parameters": vars(args),
        "frozen_inputs": manifests,
        "cases": {},
    }
    write_report(args.output, report)
    try:
        report["environment"] = environment_info(args.backend, args.device)
        report["runtime_identity"] = runtime_identity(args.backend)
        report["git"] = git_revision()
        report["source"] = source_manifest()
        measured_repeats = max(3, args.deterministic_repeats)
        for qualified_name, case in cases.items():
            mirror_search = bool(np.asarray(case["mirror_search"]).item())
            configs = schedule_configs(
                global_iterations=args.global_iterations,
                precise_iterations=args.precise_iterations,
                batch_size=args.batch_size,
                memory_fraction=args.memory_fraction,
                mirror_search=mirror_search,
            )
            case_report = {
                "particle_count": len(case["images"]),
                "reference_count": len(case["references"]),
                "image_shape": list(case["images"].shape[1:]),
                "schedules": {},
            }
            report["cases"][qualified_name] = case_report
            write_report(args.output, report)
            for schedule, schedule_config in configs.items():
                print(
                    f"RUN  {qualified_name} {schedule}: warm-up + "
                    f"{measured_repeats} measured",
                    flush=True,
                )
                case_report["schedules"][schedule] = run_schedule(
                    qualified_name,
                    case,
                    schedule=schedule,
                    configs=schedule_config,
                    backend=args.backend,
                    measured_repeats=measured_repeats,
                    profile_execution=args.profile_execution,
                    output=args.output,
                )
                write_report(args.output, report)
            case_report["comparisons"] = compare_schedules(case_report["schedules"])
            write_report(args.output, report)
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
