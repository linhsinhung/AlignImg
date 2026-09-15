#!/usr/bin/env python3
"""Compare accepted dev9 raw-output path with dev10 GPU accumulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASELINE_SOURCE_SHA256 = (
    "b39fde39e8e3606b320b539b77989cfd3b72677be87da78b80e9177535915829"
)
BASELINE_VERSION = "2.1.0.dev9"
CURRENT_VERSION = "2.1.0.dev10"
STAGE_CASES = {
    "smoke": ("adaptive",),
    "representative": ("local_refine",),
}


def result_path(report_path: Path, case_name: str) -> Path:
    return report_path.with_name(f"{report_path.stem}.{case_name}.result.npz")


def validate_baseline_report(path: Path, selected: tuple[str, ...]) -> dict:
    from tools.performance_validation import sha256

    report = json.loads(path.read_text())
    if report.get("status") != "completed":
        raise ValueError("baseline performance report is incomplete")
    if report.get("source", {}).get("source_sha256") != BASELINE_SOURCE_SHA256:
        raise ValueError("baseline report must use the accepted dev9 source")
    if report.get("environment", {}).get("alignimg_module_version") != BASELINE_VERSION:
        raise ValueError("baseline report must use AlignImg dev9")
    if report.get("native_build", {}).get("version") != BASELINE_VERSION:
        raise ValueError("baseline report must use the dev9 native CUDA build")
    for name in selected:
        case = report.get("cases", {}).get(name)
        if not case or case.get("status") != "passed":
            raise ValueError(f"baseline case is unavailable: {name}")
        arrays = result_path(path, name)
        if not arrays.is_file():
            raise ValueError(
                f"baseline result is missing: {arrays}; preserve the dev9 current NPZ files"
            )
        if sha256(arrays) != case.get("result_sha256"):
            raise ValueError(f"baseline result hash mismatch: {name}")
    return report


def validate_raw_gpu_accumulation_contract(
    name: str, baseline: dict, current: dict
) -> dict:
    metadata = current.get("class_average_metadata") or {}
    if metadata.get("accumulation") != "cupy_fp64_device":
        raise AssertionError(f"{name}: final raw average did not use GPU accumulation")
    if metadata.get("transform_backend") != "cuda":
        raise AssertionError(f"{name}: final raw transform did not use native CUDA")
    memory_plan = metadata.get("gpu_memory_plan")
    memory_events = metadata.get("gpu_memory_events") or []
    if not memory_plan or memory_plan["batch_size"] < 1:
        raise AssertionError(f"{name}: final raw VRAM plan is missing")
    if any(record.get("event") == "oom_retry" for record in memory_events):
        raise AssertionError(f"{name}: unexpected final raw OOM retry")

    before_stage = baseline["performance"]["stages"]["workflow/final_raw_average"]
    after_stage = current["performance"]["stages"]["workflow/final_raw_average"]
    before = before_stage["counters"]
    after = after_stage["counters"]
    particles = int(current["particle_count"])
    pixels = int(current["image_shape"][0]) * int(current["image_shape"][1])
    components = int(metadata.get("component_count") or 0)
    if components < 1:
        raise AssertionError(f"{name}: final raw component count is missing")
    avoided = int(after.get("final_raw_aligned_d2h_bytes_avoided", 0))
    output_bytes = int(after.get("final_raw_output_d2h_bytes", 0))
    if int(after.get("final_raw_gpu_accumulation_particles", 0)) != particles:
        raise AssertionError(f"{name}: final raw particle accounting is inconsistent")
    if avoided != particles * pixels * 4:
        raise AssertionError(f"{name}: avoided aligned-stack bytes are inconsistent")
    if output_bytes != components * pixels * 4:
        raise AssertionError(f"{name}: final raw output bytes are inconsistent")
    if int(after.get("d2h_calls", 0)) != 1 or int(after.get("d2h_bytes", 0)) != output_bytes:
        raise AssertionError(f"{name}: final raw D2H was not reduced to K images")
    if int(before.get("d2h_bytes", 0)) != avoided:
        raise AssertionError(f"{name}: dev9 baseline did not download the aligned stack")
    before_transfer = int(before.get("h2d_bytes", 0)) + int(before.get("d2h_bytes", 0))
    after_transfer = int(after.get("h2d_bytes", 0)) + int(after.get("d2h_bytes", 0))
    if after_transfer >= before_transfer:
        raise AssertionError(f"{name}: total final raw transfer bytes did not decrease")

    old_peak = int(
        baseline["performance"].get("gpu_memory", {}).get(
            "tracked_pool_live_bytes_peak", 0
        )
    )
    new_peak = int(
        current["performance"].get("gpu_memory", {}).get(
            "tracked_pool_live_bytes_peak", 0
        )
    )
    return {
        "final_raw_gpu_accumulation_batches": int(
            after.get("final_raw_gpu_accumulation_batches", 0)
        ),
        "final_raw_gpu_accumulation_particles": particles,
        "final_raw_aligned_d2h_bytes_avoided": avoided,
        "final_raw_output_d2h_bytes": output_bytes,
        "final_raw_transfer_bytes_before": before_transfer,
        "final_raw_transfer_bytes_after": after_transfer,
        "final_raw_transfer_bytes_saved": before_transfer - after_transfer,
        "final_raw_stage_seconds_before": float(before_stage["wall_seconds"]),
        "final_raw_stage_seconds_after": float(after_stage["wall_seconds"]),
        "final_raw_stage_current_over_baseline_seconds": float(
            after_stage["wall_seconds"] / max(before_stage["wall_seconds"], 1e-12)
        ),
        "tracked_pool_live_bytes_before": old_peak,
        "tracked_pool_live_bytes_after": new_peak,
        "tracked_pool_live_bytes_saved": old_peak - new_peak,
        "d2h_calls_saved": int(before.get("d2h_calls", 0)) - 1,
        "d2h_bytes_saved": int(before.get("d2h_bytes", 0)) - output_bytes,
        "gpu_memory_plan": memory_plan,
    }


def compare_reports(
    baseline_path: Path, current_path: Path, selected: tuple[str, ...]
) -> dict:
    import numpy as np
    from tools.performance_validation import compare_arrays, sha256

    old = json.loads(baseline_path.read_text())
    new = json.loads(current_path.read_text())
    if new.get("environment", {}).get("alignimg_module_version") != CURRENT_VERSION:
        raise RuntimeError("current report did not use AlignImg dev10")
    if new.get("native_build", {}).get("version") != CURRENT_VERSION:
        raise RuntimeError("current report did not use the dev10 native CUDA build")
    output = {}
    for name in selected:
        baseline = old["cases"][name]
        current = new["cases"][name]
        if current.get("status") != "passed":
            raise AssertionError(f"{name}: current performance case failed")
        if baseline["frozen_inputs_sha256"] != current["frozen_inputs_sha256"]:
            raise AssertionError(f"{name}: baseline and current inputs differ")
        arrays = []
        for report_path, case in (
            (baseline_path, baseline),
            (current_path, current),
        ):
            path = result_path(report_path, name)
            if sha256(path) != case["result_sha256"]:
                raise AssertionError(f"{name}: result hash mismatch")
            with np.load(path, allow_pickle=False) as data:
                arrays.append(dict(data))
        ratio = (
            current["unprofiled_median_seconds"]
            / baseline["unprofiled_median_seconds"]
        )
        output[name] = {
            "numeric_errors": compare_arrays(*arrays),
            "current_over_baseline_seconds": ratio,
            "timing_review": "repeat_required" if ratio > 1.05 else "within_gate",
            "baseline_counters": baseline["performance"]["counters"],
            "current_counters": current["performance"]["counters"],
            **validate_raw_gpu_accumulation_contract(name, baseline, current),
        }
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda",), default="cuda")
    parser.add_argument("--suite", choices=tuple(STAGE_CASES), default="smoke")
    parser.add_argument("--only")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    selected = tuple(args.only.split(",")) if args.only else STAGE_CASES[args.suite]
    if not selected or not set(selected) <= set(STAGE_CASES[args.suite]):
        parser.error("unknown case")
    if args.repeats < 1 or args.batch_size < 2 or not 0 < args.memory_fraction <= 1:
        parser.error("invalid repeats, batch size, or memory fraction")
    baseline_path = args.baseline_report or ROOT / (
        "validation-results/performance/stage-5/"
        f"dev9-cuda-fused-{args.suite}-ab-clean-repeat.current.json"
    )
    current_path = args.output.with_suffix(".current.json")
    if args.output.exists() or current_path.exists():
        parser.error("output exists; choose a new report name")

    from tools import performance_validation as perf

    report = {
        "stage": 5,
        "increment": "gpu_final_raw_average_accumulation",
        "status": "running",
        "backend": args.backend,
        "baseline": str(baseline_path),
        "current": str(current_path),
        "cases": {},
    }
    perf.write_report(args.output, report)
    try:
        if perf.ai.__version__ != CURRENT_VERSION:
            raise RuntimeError(f"install AlignImg {CURRENT_VERSION} before Stage 5 A/B")
        validate_baseline_report(baseline_path, selected)
        common = [
            "--backend", "cuda",
            "--suite", args.suite,
            "--only", ",".join(selected),
            "--batch-size", str(args.batch_size),
            "--memory-fraction", str(args.memory_fraction),
            "--device", str(args.device),
            "--repeats", str(args.repeats),
            "--capture-candidates",
            "--output", str(current_path),
        ]
        if perf.main(common):
            raise RuntimeError("current engine validation failed; inspect current JSON")
        report["cases"] = compare_reports(baseline_path, current_path, selected)
        report["status"] = (
            "repeat_required"
            if any(
                case["timing_review"] == "repeat_required"
                for case in report["cases"].values()
            )
            else "passed"
        )
        report["scientific_review"] = "pending"
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
    perf.write_report(args.output, report)
    print(f"Stage 5 GPU raw average: {report['status']}; report: {args.output}")
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
