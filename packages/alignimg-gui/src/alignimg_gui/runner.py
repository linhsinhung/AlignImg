"""Two-stage alignment worker implementation without any Qt dependency."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
import traceback
from typing import Any, Callable

import numpy as np

import alignimg as ai

from .artifacts import (
    base_report,
    load_mrc_stack,
    load_previous_result,
    save_result,
    write_json,
)
from .spec import RunSpec


EventSink = Callable[[dict[str, Any]], None]


def _summary(
    result: ai.AlignmentResult,
    seconds: float,
    diagnostics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    items = diagnostics if diagnostics is not None else result.diagnostics
    counts = np.bincount(
        result.reference_assignments, minlength=len(result.references)
    ).astype(np.int64)
    return {
        "seconds": seconds,
        "particles_per_second": len(result.poses) / max(seconds, 1e-12),
        "hard_assignment_counts": counts,
        "mean_max_responsibility": float(
            np.mean(np.max(result.responsibilities, axis=1))
        ),
        "mean_inlier_weight": float(np.mean(result.inlier_weights)),
        "mean_expected_fourier_ncc_trajectory": [
            item.get("mean_expected_fourier_ncc") for item in items
        ],
        "reference_relative_change_trajectory": [
            item.get("reference_relative_change") for item in items
        ],
        "temperature_trajectory": [item.get("temperature") for item in items],
    }


def _stage_report(
    run_directory: Path,
    stage: str,
    result: ai.AlignmentResult,
    seconds: float,
    pixel_size: float | None,
) -> dict[str, Any]:
    stage_directory = run_directory / stage
    stage_directory.mkdir(exist_ok=True)
    payload = {
        "schema": "alignimg-gui.stage.v1",
        "status": "completed",
        "stage": stage,
        "summary": _summary(result, seconds),
        "diagnostics": result.diagnostics,
        "metadata": result.metadata,
        "artifacts": save_result(stage_directory, result, pixel_size=pixel_size),
    }
    write_json(stage_directory / "report.json", payload)
    payload["artifacts"] = {
        name: str(Path(stage) / value) for name, value in payload["artifacts"].items()
    }
    payload["report"] = str(Path(stage) / "report.json")
    return payload


def _refinement_priors(
    policy: str,
    assignments: np.ndarray,
    responsibilities: np.ndarray | None,
    reference_count: int,
    trust: float,
) -> np.ndarray | None:
    if policy == "free":
        return None
    if policy == "fixed":
        return ai.make_class_priors(
            assignments=assignments,
            n_components=reference_count,
            trust=1.0,
        )
    if policy == "soft" and responsibilities is not None:
        if responsibilities.shape != (len(assignments), reference_count):
            raise ValueError(
                "previous responsibilities do not match image/reference counts."
            )
        return ai.make_class_priors(
            responsibilities=responsibilities,
            trust=trust,
        )
    return ai.make_class_priors(
        assignments=assignments,
        n_components=reference_count,
        trust=trust,
    )


def _combined_history(
    global_result: ai.AlignmentResult | None,
    refine_result: ai.AlignmentResult | None,
) -> np.ndarray:
    def history(result: ai.AlignmentResult) -> np.ndarray:
        if result.reference_history:
            return np.asarray(result.reference_history, dtype=np.float32)
        return np.asarray(result.references[None], dtype=np.float32)

    if global_result is None:
        assert refine_result is not None
        return history(refine_result)
    global_history = history(global_result)
    if refine_result is None:
        return global_history
    refine_history = history(refine_result)
    if refine_result.reference_history:
        refine_history = refine_history[1:]
    return np.concatenate((global_history, refine_history), axis=0)


def execute_run(spec: RunSpec, emit: EventSink | None = None) -> dict[str, Any]:
    spec = spec.normalized()
    run_directory = spec.run_directory
    run_directory.mkdir(parents=True, exist_ok=True)
    report_path = run_directory / "report.json"
    report = base_report(spec.to_dict())
    report["stages"] = {}
    write_json(report_path, report)

    def notify(event: str, message: str, **values: Any) -> None:
        if emit is not None:
            emit({"event": event, "message": message, **values})

    try:
        notify("started", "Loading particle stack")
        images, pixel_size = load_mrc_stack(spec.particles)
        references = None
        if spec.references:
            references, _ = load_mrc_stack(spec.references, allow_single=True)
            if references.shape[1:] != images.shape[1:]:
                raise ValueError("particle and reference image sizes do not match.")

        previous = None
        if not spec.run_global and spec.previous_result:
            previous = load_previous_result(
                spec.previous_result, len(images), images.shape[-1]
            )

        pipeline_started = time.perf_counter()
        global_result: ai.AlignmentResult | None = None
        refine_result: ai.AlignmentResult | None = None
        if spec.run_global:
            notify(
                "running",
                "Running Reference-Free global stage"
                if spec.mode == "reference_free"
                else "Running reference-based global stage",
                stage="global",
                particle_count=len(images),
            )
            started = time.perf_counter()
            global_config = ai.AlignmentConfig(**spec.global_config)
            if spec.mode == "reference_free":
                global_result = ai.reference_free_align(
                    images,
                    n_components=spec.n_components,
                    config=global_config,
                    backend=spec.backend,
                )
            else:
                assert references is not None
                global_result = ai.align_to_references(
                    images,
                    references,
                    config=global_config,
                    backend=spec.backend,
                )
            seconds = time.perf_counter() - started
            report["stages"]["global"] = _stage_report(
                run_directory, "global", global_result, seconds, pixel_size
            )
            write_json(report_path, report)
            notify(
                "stage_completed", f"Global stage completed in {seconds:.3f} seconds"
            )

        if spec.refine_enabled:
            if global_result is not None:
                stage_references = global_result.references
                poses = global_result.poses
                assignments = global_result.reference_assignments
                responsibilities = global_result.responsibilities
            else:
                assert references is not None and previous is not None
                stage_references = references
                poses, assignments, responsibilities = previous
            if np.any(assignments < 0) or np.any(assignments >= len(stage_references)):
                raise ValueError(
                    "previous assignments are outside the reference range."
                )
            priors = _refinement_priors(
                spec.refine_class_policy,
                assignments,
                responsibilities,
                len(stage_references),
                spec.corrective_trust,
            )
            notify(
                "running",
                f"Running local refinement ({spec.refine_class_policy} class policy)",
                stage="refine",
                particle_count=len(images),
            )
            started = time.perf_counter()
            refine_result = ai.refine_alignment(
                images,
                stage_references,
                poses,
                class_priors=priors,
                config=ai.AlignmentConfig(**spec.refine_config),
                backend=spec.backend,
            )
            seconds = time.perf_counter() - started
            report["stages"]["refine"] = _stage_report(
                run_directory, "refine", refine_result, seconds, pixel_size
            )
            write_json(report_path, report)
            notify(
                "stage_completed", f"Refine stage completed in {seconds:.3f} seconds"
            )

        final_result = refine_result or global_result
        assert final_result is not None
        combined_history = _combined_history(global_result, refine_result)
        combined_diagnostics: list[dict[str, Any]] = []
        for stage, result in (("global", global_result), ("refine", refine_result)):
            if result is None:
                continue
            for item in result.diagnostics:
                combined_diagnostics.append(
                    {
                        **item,
                        "stage": stage,
                        "pipeline_iteration": len(combined_diagnostics),
                    }
                )
        seconds = time.perf_counter() - pipeline_started
        notify("saving", "Saving combined pipeline result")
        report.update(
            {
                "status": "completed",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "final_stage": "refine" if refine_result is not None else "global",
                "input": {
                    "particle_count": len(images),
                    "image_shape": list(images.shape[1:]),
                    "pixel_size_angstrom": pixel_size,
                    "reference_count": len(final_result.references),
                },
                "summary": _summary(
                    final_result, seconds, diagnostics=combined_diagnostics
                ),
                "diagnostics": combined_diagnostics,
                "metadata": final_result.metadata,
                "artifacts": save_result(
                    run_directory,
                    final_result,
                    pixel_size=pixel_size,
                    reference_history=combined_history,
                ),
            }
        )
        write_json(report_path, report)
        notify(
            "completed", f"Pipeline completed in {seconds:.3f} seconds", seconds=seconds
        )
        return report
    except Exception as error:
        report.update(
            {
                "status": "failed",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        write_json(report_path, report)
        notify("failed", f"{type(error).__name__}: {error}")
        raise
