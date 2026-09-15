#!/usr/bin/env python3
"""Scale A/B for dev9 host and dev10 GPU final raw-average accumulation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CURRENT_VERSION = "2.1.0.dev10"
MATERIAL_BENEFIT_RATIO = 0.95
REGRESSION_RATIO = 1.05


def parse_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(sorted(set(int(item) for item in value.split(","))))
    except ValueError as error:
        raise argparse.ArgumentTypeError("counts must be comma-separated integers") from error
    if not counts or counts[0] < 1:
        raise argparse.ArgumentTypeError("counts must contain positive integers")
    return counts


def timing_review(ratio: float) -> str:
    if ratio <= MATERIAL_BENEFIT_RATIO:
        return "material_benefit"
    if ratio > REGRESSION_RATIO:
        return "regression"
    return "no_material_benefit"


def profile_call(function, cp):
    from alignimg._profiling import ExecutionProfile, _ACTIVE

    profile = ExecutionProfile()
    token = _ACTIVE.set(profile)
    try:
        profile.attach_cuda(cp)
        with profile.stage("final_raw_average", cuda=True):
            value = function()
    finally:
        try:
            profile.close()
        finally:
            _ACTIVE.reset(token)
    return value, profile.asdict()


def dev9_host_accumulation(
    images, poses, assignments, weights, references, config, *, transform_batch
):
    """Reproduce the accepted dev9 GPU-transform/CPU-accumulation path."""
    import alignimg_gpu.backend as gpu_backend
    from alignimg._fourier import soft_circular_mask

    reference_count = len(references)
    sums = np.zeros((reference_count, *images.shape[1:]), dtype=np.float64)
    total_weights = np.zeros(reference_count, dtype=np.float64)
    batch_size = int(config.batch_size or 512)
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        aligned = gpu_backend._ashost(
            transform_batch(
                images[start:stop],
                poses.angle_deg[start:stop],
                poses.shift_y_px[start:stop],
                poses.shift_x_px[start:stop],
                poses.mirror[start:stop],
            )
        ).astype(np.float32, copy=False)
        batch_assignments = assignments[start:stop]
        batch_weights = weights[start:stop]
        for reference_index in np.unique(batch_assignments):
            selected = batch_assignments == reference_index
            selected_weights = batch_weights[selected]
            sums[reference_index] += np.tensordot(
                selected_weights,
                aligned[selected].astype(np.float64),
                axes=(0, 0),
            )
            total_weights[reference_index] += float(np.sum(selected_weights))

    averages = np.asarray(references, dtype=np.float32).copy()
    nonempty = total_weights > 1e-8
    averages[nonempty] = (
        sums[nonempty] / total_weights[nonempty, None, None]
    ).astype(np.float32)
    averages[nonempty] *= soft_circular_mask(
        images.shape[-1], config.mask_radius, config.mask_soft_edge
    )
    return averages


def compare_outputs(expected: np.ndarray, actual: np.ndarray) -> dict:
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    maximum = float(np.max(np.abs(delta), initial=0.0))
    relative = float(
        np.linalg.norm(delta) / max(np.linalg.norm(expected.astype(np.float64)), 1e-12)
    )
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=3e-5)
    return {"maximum_absolute_error": maximum, "relative_l2_error": relative}


def transfer_summary(
    count: int,
    size: int,
    components: int,
    baseline_profile: dict,
    current_profile: dict,
) -> dict:
    before = baseline_profile["stages"]["final_raw_average"]["counters"]
    after = current_profile["stages"]["final_raw_average"]["counters"]
    aligned_bytes = count * size * size * 4
    output_bytes = components * size * size * 4
    if int(before.get("d2h_bytes", 0)) != aligned_bytes:
        raise AssertionError("dev9 baseline did not download the complete aligned stack")
    if int(after.get("d2h_calls", 0)) != 1:
        raise AssertionError("dev10 must perform exactly one final raw-average D2H")
    if int(after.get("d2h_bytes", 0)) != output_bytes:
        raise AssertionError("dev10 must download exactly K class averages")
    if int(after.get("final_raw_aligned_d2h_bytes_avoided", 0)) != aligned_bytes:
        raise AssertionError("dev10 avoided-byte accounting is inconsistent")
    before_total = int(before.get("h2d_bytes", 0)) + int(before.get("d2h_bytes", 0))
    after_total = int(after.get("h2d_bytes", 0)) + int(after.get("d2h_bytes", 0))
    if after_total >= before_total:
        raise AssertionError("dev10 did not reduce total explicit transfer bytes")
    return {
        "baseline_h2d_bytes": int(before.get("h2d_bytes", 0)),
        "baseline_d2h_calls": int(before.get("d2h_calls", 0)),
        "baseline_d2h_bytes": int(before.get("d2h_bytes", 0)),
        "current_h2d_bytes": int(after.get("h2d_bytes", 0)),
        "current_d2h_calls": int(after.get("d2h_calls", 0)),
        "current_d2h_bytes": int(after.get("d2h_bytes", 0)),
        "total_transfer_bytes_saved": before_total - after_total,
        "aligned_d2h_bytes_avoided": aligned_bytes,
    }


def synchronize(cp):
    cp.cuda.get_current_stream().synchronize()


def time_call(function, cp) -> tuple[float, np.ndarray]:
    synchronize(cp)
    started = time.perf_counter()
    value = function()
    synchronize(cp)
    return time.perf_counter() - started, value


def run_case(count, values, config, repeats, cp, gpu_backend, backend):
    images = values["images"][:count]
    assignments = values["assignments"][:count]
    weights = values["weights"][:count]
    poses = type(values["poses"])(
        values["poses"].angle_deg[:count],
        values["poses"].shift_y_px[:count],
        values["poses"].shift_x_px[:count],
        values["poses"].mirror[:count],
    )
    references = values["references"]
    transform = getattr(gpu_backend, f"_transform_batch_{backend}")
    current = getattr(gpu_backend, f"_final_raw_class_averages_{backend}")

    def baseline_call():
        return dev9_host_accumulation(
            images,
            poses,
            assignments,
            weights,
            references,
            config,
            transform_batch=transform,
        )

    def current_call():
        return current(
            images, poses, assignments, weights, references, config
        )

    baseline_call()
    current_call()
    synchronize(cp)

    baseline_seconds = []
    current_seconds = []
    expected = actual = None
    for repeat in range(repeats):
        order = (("baseline", baseline_call), ("current", current_call))
        if repeat % 2:
            order = tuple(reversed(order))
        for name, function in order:
            seconds, result = time_call(function, cp)
            if isinstance(result, tuple):
                result = result[0]
            if name == "baseline":
                baseline_seconds.append(seconds)
                expected = result
            else:
                current_seconds.append(seconds)
                actual = result

    profiled_baseline, baseline_profile = profile_call(baseline_call, cp)
    profiled_current, current_profile = profile_call(current_call, cp)
    current_output, current_metadata = profiled_current
    numeric = compare_outputs(profiled_baseline, current_output)
    compare_outputs(expected, actual)
    ratio = float(np.median(current_seconds) / np.median(baseline_seconds))
    return {
        "status": "passed",
        "particle_count": count,
        "baseline_wall_seconds": baseline_seconds,
        "current_wall_seconds": current_seconds,
        "baseline_median_seconds": float(np.median(baseline_seconds)),
        "current_median_seconds": float(np.median(current_seconds)),
        "current_over_baseline_seconds": ratio,
        "timing_review": timing_review(ratio),
        "numeric_errors": numeric,
        "transfer": transfer_summary(
            count,
            images.shape[-1],
            len(references),
            baseline_profile,
            current_profile,
        ),
        "baseline_performance": baseline_profile,
        "current_performance": current_profile,
        "current_metadata": {
            key: value
            for key, value in current_metadata.items()
            if key.startswith("class_average_")
        },
    }


def make_values(count: int, size: int, components: int, seed: int):
    import alignimg as ai

    rng = np.random.default_rng(seed)
    images = rng.standard_normal((count, size, size), dtype=np.float32)
    references = rng.standard_normal((components, size, size), dtype=np.float32)
    assignments = np.arange(count, dtype=np.int32) % components
    weights = rng.uniform(0.5, 1.0, count).astype(np.float64)
    poses = ai.PoseSet(
        angle_deg=rng.uniform(-180.0, 180.0, count).astype(np.float32),
        shift_y_px=rng.uniform(-4.0, 4.0, count).astype(np.float32),
        shift_x_px=rng.uniform(-4.0, 4.0, count).astype(np.float32),
        mirror=np.zeros(count, dtype=bool),
    )
    return {
        "images": images,
        "references": references,
        "assignments": assignments,
        "weights": weights,
        "poses": poses,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "cupy"), default="cuda")
    parser.add_argument("--counts", type=parse_counts, default=parse_counts("1000,3000,5000"))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        args.size < 2
        or args.size % 2
        or args.components < 1
        or args.components > min(args.counts)
        or args.batch_size < 1
        or args.repeats < 1
        or not 0.0 < args.memory_fraction <= 1.0
    ):
        parser.error("invalid size, components, batch size, repeats, or memory fraction")
    if args.output.exists():
        parser.error("output already exists; choose a new report name")

    from dataclasses import replace
    import alignimg as ai
    import alignimg_gpu
    import alignimg_gpu.backend as gpu_backend
    import cupy as cp
    from tools.performance_fixtures import source_manifest
    from tools.performance_validation import installed_source_info
    from tools.server_validation import environment_info, utc_now, write_report

    cp.cuda.Device(args.device).use()
    report = {
        "schema": "alignimg.stage5.raw-average-scale.v1",
        "stage": 5,
        "increment": "gpu_final_raw_average_accumulation",
        "status": "running",
        "started_at_utc": utc_now(),
        "parameters": vars(args),
        "cases": {},
    }
    write_report(args.output, report)
    try:
        if ai.__version__ != CURRENT_VERSION or alignimg_gpu.__version__ != CURRENT_VERSION:
            raise RuntimeError(f"install AlignImg core and GPU {CURRENT_VERSION}")
        if args.backend == "cuda":
            native = gpu_backend._native_module()
            if native is None or native.__version__ != CURRENT_VERSION:
                raise RuntimeError("native CUDA build is missing or stale")
            report["native_build"] = {
                "version": native.__version__,
                **native.runtime_info(),
            }
        source = source_manifest()
        report["source"] = source
        report["environment"] = environment_info(args.backend, args.device)
        report["installed_core"] = installed_source_info(
            ai, "src/alignimg/", source["files"]
        )
        report["installed_gpu"] = installed_source_info(
            alignimg_gpu,
            "packages/alignimg-gpu/src/alignimg_gpu/",
            source["files"],
        )
        config = replace(
            ai.AlignmentConfig(),
            apply_final_pose_to_raw=True,
            batch_size=args.batch_size,
            memory_fraction=args.memory_fraction,
        )
        report["config"] = asdict(config)
        values = make_values(max(args.counts), args.size, args.components, args.seed)
        for count in args.counts:
            print(f"RUN  raw_average count={count}", flush=True)
            report["cases"][str(count)] = run_case(
                count,
                values,
                config,
                args.repeats,
                cp,
                gpu_backend,
                args.backend,
            )
            print(
                f"PASS raw_average count={count}: ratio="
                f"{report['cases'][str(count)]['current_over_baseline_seconds']:.3f}",
                flush=True,
            )
            write_report(args.output, report)
        largest = report["cases"][str(max(args.counts))]
        report["summary"] = {
            "largest_particle_count": max(args.counts),
            "largest_current_over_baseline_seconds": largest[
                "current_over_baseline_seconds"
            ],
            "timing_review": largest["timing_review"],
            "acceptance_rule": (
                "Accept only when the largest case is at least 5% faster; "
                "repeat a regression on an otherwise idle GPU before rollback."
            ),
        }
        report["status"] = (
            "passed"
            if largest["timing_review"] == "material_benefit"
            else largest["timing_review"]
        )
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
    report["completed_at_utc"] = utc_now()
    write_report(args.output, report)
    print(f"Stage 5 raw-average scale: {report['status']}; report: {args.output}")
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
