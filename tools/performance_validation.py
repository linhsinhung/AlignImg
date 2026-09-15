#!/usr/bin/env python3
"""Profiling parity and separate warmed, unprofiled throughput runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
from unittest.mock import patch

import numpy as np

import alignimg as ai

try:
    from tools.performance_fixtures import (
        DEFAULT_BASELINE,
        ROOT,
        capture_iteration,
        fixture_config,
        poses_from_inputs,
        sha256,
        source_manifest,
        synthetic_inputs,
    )
    from tools.server_validation import environment_info, utc_now, write_report
except ModuleNotFoundError:
    from performance_fixtures import (
        DEFAULT_BASELINE,
        ROOT,
        capture_iteration,
        fixture_config,
        poses_from_inputs,
        sha256,
        source_manifest,
        synthetic_inputs,
    )
    from server_validation import environment_info, utc_now, write_report


SMOKE_CASES = ("global", "adaptive", "reference_free", "global_mirror", "adaptive_whitened")
REAL_CASES = ("pose_global", "pose_fixed_mra", "local_global", "local_refine")


def installed_source_info(module, prefix, expected_files):
    directory = Path(module.__file__).resolve().parent
    files = {
        str(path.relative_to(directory)): sha256(path)
        for path in directory.rglob("*.py")
    }
    expected = {
        key[len(prefix) :]: value
        for key, value in expected_files.items()
        if key.startswith(prefix) and key.endswith(".py")
    }
    if files != expected:
        raise RuntimeError(
            f"installed {module.__name__} sources differ from this checkout; reinstall"
        )
    return {"directory": str(directory), "sha256": files}


def result_arrays(result):
    values = {
        "references": result.references,
        "class_averages": result.class_averages,
        "assignments": result.reference_assignments,
        "responsibilities": result.responsibilities,
        "inlier_weights": result.inlier_weights,
        "angle_deg": result.poses.angle_deg,
        "shift_y_px": result.poses.shift_y_px,
        "shift_x_px": result.poses.shift_x_px,
        "mirror": result.poses.mirror,
        "reference_history": np.asarray(result.reference_history),
    }
    values.update(
        {f"candidate_{name}": value for name, value in vars(result.candidates).items()}
    )
    for index, diagnostic in enumerate(result.diagnostics):
        for name in (
            "effective_component_weight",
            "frc",
            "frc_0143_cutoff_cyc_per_px",
            "frc_0143_stable_cutoff_cyc_per_px",
            "halfset_effective_weight",
        ):
            if name in diagnostic:
                values[f"iteration_{index}_{name}"] = diagnostic[name]
    return values


def compare_arrays(expected: dict, actual: dict) -> dict:
    if set(expected) != set(actual):
        raise AssertionError("result fields differ")
    errors = {}
    for name, reference in expected.items():
        reference, value = np.asarray(reference), np.asarray(actual[name])
        if reference.shape != value.shape:
            raise AssertionError(f"{name}: shape changed")
        if reference.dtype.kind in "biu":
            np.testing.assert_array_equal(value, reference, err_msg=name)
            continue
        # Rescue gain is deliberately NaN when no rescue was attempted.
        if name == "e__rescue_score_gain" or name.endswith("_e__rescue_score_gain"):
            np.testing.assert_array_equal(
                np.isnan(reference), np.isnan(value), err_msg=name
            )
            reference = reference[~np.isnan(reference)]
            value = value[~np.isnan(value)]
        # Padded candidate scores may be -inf. Other NaNs are never valid.
        if np.isnan(reference).any() or np.isnan(value).any():
            raise AssertionError(f"{name}: NaN")
        np.testing.assert_array_equal(
            np.isfinite(value), np.isfinite(reference), err_msg=name
        )
        finite = np.isfinite(reference)
        np.testing.assert_array_equal(value[~finite], reference[~finite], err_msg=name)
        delta = value[finite].astype(np.float64) - reference[finite].astype(np.float64)
        if name == "angle_deg":
            delta = (delta + 180.0) % 360.0 - 180.0
        if name in {"angle_deg", "shift_y_px", "shift_x_px"}:
            np.testing.assert_allclose(delta, 0.0, atol=1e-3, rtol=0, err_msg=name)
        else:
            np.testing.assert_allclose(
                value[finite], reference[finite], atol=2e-4, rtol=3e-5, err_msg=name
            )
        errors[name] = {
            "maximum_absolute_error": float(np.max(np.abs(delta), initial=0)),
            "relative_l2_error": float(
                np.linalg.norm(delta) / max(np.linalg.norm(reference[finite]), 1e-12)
            ),
        }
    return errors


def check_result(result, backend, priors=None):
    if result.metadata["backend"] != backend:
        raise AssertionError("backend fallback is not allowed")
    for name in ("references", "class_averages", "responsibilities", "inlier_weights"):
        if not np.isfinite(getattr(result, name)).all():
            raise AssertionError(f"non-finite {name}")
    np.testing.assert_allclose(
        result.responsibilities.sum(axis=1), 1.0, atol=5e-6, rtol=0
    )
    if np.any(result.responsibilities < 0):
        raise AssertionError("negative responsibilities")
    if priors is not None:
        np.testing.assert_array_equal(
            result.reference_assignments, np.argmax(priors, axis=1)
        )


def execute(values, config, backend, workflow):
    if workflow == "reference_free":
        return ai.reference_free_align(
            values["images"], n_components=2, config=config, backend=backend
        )
    kwargs = dict(config=config, backend=backend, class_priors=values.get("priors"))
    if workflow == "refine":
        return ai.refine_alignment(
            values["images"], values["references"], poses_from_inputs(values), **kwargs
        )
    return ai.align_to_references(values["images"], values["references"], **kwargs)


def profiling_case(backend: str, batch_size: int) -> dict:
    values = synthetic_inputs()
    config = replace(
        fixture_config(True), batch_size=batch_size, apply_final_pose_to_raw=True
    )
    plain = execute(values, config, backend, "refine")
    profiled = execute(
        values, replace(config, profile_execution=True), backend, "refine"
    )
    check_result(profiled, backend, values["priors"])
    errors = compare_arrays(result_arrays(plain), result_arrays(profiled))
    performance = profiled.metadata["performance"]
    if (
        not performance["iterations"]
        or "workflow/final_raw_average" not in performance["stages"]
    ):
        raise AssertionError("incomplete profiling metadata")
    if backend != "cpu" and (
        performance["gpu_memory"] is None
        or not performance["counters"].get("h2d_bytes")
    ):
        raise AssertionError("GPU profiling was not attached")
    return {"profile_parity": errors, "performance": performance}


def load_case(name, batch_size, memory_fraction):
    adaptive = name in {"adaptive", "adaptive_whitened", "pose_fixed_mra", "local_refine"}
    config = replace(
        fixture_config(adaptive), batch_size=batch_size, memory_fraction=memory_fraction
    )
    workflow = "refine" if adaptive else "global"
    paths = []
    if name in SMOKE_CASES:
        values = synthetic_inputs()
        if name == "reference_free":
            workflow = "reference_free"
            config = replace(config, max_iterations=2)
        elif adaptive:
            config = replace(config, apply_final_pose_to_raw=True)
        if name == "global_mirror":
            config = replace(config, mirror_search=True, max_iterations=2)
        elif name == "adaptive_whitened":
            config = replace(config, score_model="whitened_fourier_ncc", max_iterations=2)
    elif name in {"pose_global", "pose_fixed_mra"}:
        try:
            from tools.re2dc_70s_pose_benchmark import load_benchmark, DEFAULT_BENCHMARK
        except ModuleNotFoundError:
            from re2dc_70s_pose_benchmark import load_benchmark, DEFAULT_BENCHMARK
        _, images, references, truth, poses = load_benchmark(ROOT / DEFAULT_BENCHMARK)
        paths = [ROOT / DEFAULT_BENCHMARK]
        values = {
            "images": images,
            "references": references,
            "priors": ai.make_class_priors(
                assignments=truth["component_index"], n_components=len(references)
            ),
            "particle_indices": truth["prepared_index_zero_based"],
        }
        values.update(
            {
                f"initial_{key}": getattr(poses, key)
                for key in ("angle_deg", "shift_y_px", "shift_x_px", "mirror")
            }
        )
        config = replace(
            config,
            angle_samples=128,
            top_l=8,
            local_angle_range=15,
            local_shift_range=3,
        )
    else:
        import mrcfile

        particle_path = ROOT / "data/local/test_align.mrcs"
        reference_path = ROOT / "data/local/mu_aligned_mean.mrc"
        paths = [particle_path, reference_path]
        with mrcfile.mmap(particle_path, mode="r") as stack:
            images = np.asarray(stack.data[:256], dtype=np.float32).copy()
        with mrcfile.open(reference_path) as stack:
            references = (
                np.asarray(stack.data, dtype=np.float32)
                .reshape(1, *images.shape[1:])
                .copy()
            )
        values = {
            "images": images,
            "references": references,
            "particle_indices": np.arange(len(images), dtype=np.int32),
        }
        config = replace(
            ai.AlignmentConfig.preset("refine" if adaptive else "global_accurate"),
            max_iterations=1,
            batch_size=batch_size,
            memory_fraction=memory_fraction,
            angle_samples=256,
            proposal_angles_per_reference=8,
            translation_range=6,
            coarse_angle_step=3,
            local_angle_range=6,
            local_shift_range=2,
        )
        if adaptive:
            pose_path = (
                ROOT / "validation-results/local-known-reference-raw-average/result.npz"
            )
            paths.append(pose_path)
            with np.load(pose_path, allow_pickle=False) as data:
                for key in ("angle_deg", "shift_y_px", "shift_x_px"):
                    values[f"initial_{key}"] = data[f"global_{key}"][
                        : len(images)
                    ].copy()
                values["references"] = data["reference_history"][1].copy()
            values["initial_mirror"] = np.zeros(len(images), dtype=bool)
            config = replace(config, apply_final_pose_to_raw=True)
    return (
        values,
        config,
        workflow,
        {str(path.relative_to(ROOT)): sha256(path) for path in paths},
    )


def frozen_fixture_check(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if sha256(directory / "source.tar.gz") != manifest["archive_sha256"]:
        raise AssertionError("frozen source archive modified")
    report = {}
    for name in ("global", "adaptive"):
        path = directory / f"{name}.npz"
        if sha256(path) != manifest["fixture_sha256"][path.name]:
            raise AssertionError(f"frozen fixture modified: {path}")
        with np.load(path, allow_pickle=False) as data:
            values = {key: data[key] for key in data.files}
        config = ai.AlignmentConfig(**json.loads(str(values.pop("config_json"))))
        actual = capture_iteration(values, config)
        expected = {
            key: value
            for key, value in values.items()
            if key not in {"images", "references", "labels", "priors"}
            and not key.startswith("initial_")
        }
        report[name] = compare_arrays(expected, actual)
    return report


def synchronize(backend):
    if backend != "cpu":
        import cupy as cp

        cp.cuda.get_current_stream().synchronize()


def run_case(name, args):
    load_started = time.perf_counter()
    values, config, workflow, input_hashes = load_case(
        name, args.batch_size, args.memory_fraction
    )
    input_seconds = time.perf_counter() - load_started
    base = args.output.parent / f"{args.output.stem}.{name}"
    np.savez_compressed(
        base.with_suffix(base.suffix + ".inputs.npz"),
        **values,
        config_json=np.asarray(json.dumps(asdict(config))),
        workflow=np.asarray(workflow),
    )
    warm_started = time.perf_counter()
    warm = execute(values, config, args.backend, workflow)
    synchronize(args.backend)
    warm_seconds = time.perf_counter() - warm_started
    check_result(
        warm,
        args.backend,
        values.get("priors") if workflow != "reference_free" else None,
    )
    expected = result_arrays(warm)
    seconds = []
    for _ in range(args.repeats):
        synchronize(args.backend)
        started = time.perf_counter()
        result = execute(values, config, args.backend, workflow)
        synchronize(args.backend)
        seconds.append(time.perf_counter() - started)
        compare_arrays(expected, result_arrays(result))
    full_candidates = {}
    if getattr(args, "capture_candidates", False):
        # Capture E-step output before centering; the extra copies occur only
        # in the diagnostic run, outside all unprofiled throughput measurements.
        from alignimg import _engine

        original_call = _engine.profile_call
        iteration = 0

        def capture(name, function, *positional, **keywords):
            nonlocal iteration
            value = original_call(name, function, *positional, **keywords)
            if name == "candidate_inference":
                full_candidates.update({
                    f"iteration_{iteration}_e_{key}": array.copy()
                    for key, array in value.items()
                })
                iteration += 1
            return value

        with patch.object(_engine, "profile_call", capture):
            profiled = execute(values, replace(config, profile_execution=True), args.backend, workflow)
    else:
        profiled = execute(
            values, replace(config, profile_execution=True), args.backend, workflow
        )
    errors = compare_arrays(expected, result_arrays(profiled))
    check_result(
        profiled,
        args.backend,
        values.get("priors") if workflow != "reference_free" else None,
    )
    output_started = time.perf_counter()
    result_path = base.with_suffix(base.suffix + ".result.npz")
    np.savez_compressed(result_path, **result_arrays(profiled), **full_candidates)
    output_seconds = time.perf_counter() - output_started
    median = float(np.median(seconds))
    class_average_metadata = {
        "estimator": profiled.metadata.get("class_average_estimator"),
        "transform_backend": profiled.metadata.get("class_average_transform_backend"),
        "accumulation": profiled.metadata.get("class_average_gpu_accumulation"),
        "batch_size": profiled.metadata.get("class_average_batch_size"),
        "component_count": profiled.metadata.get("class_average_component_count"),
        "empty_components": np.asarray(
            profiled.metadata.get("class_average_empty_components", []),
            dtype=np.int64,
        ).tolist(),
        "gpu_memory_plan": profiled.metadata.get("class_average_gpu_memory_plan"),
        "gpu_memory_events": profiled.metadata.get("class_average_gpu_memory_events"),
    }
    return {
        "config": asdict(config),
        "workflow": workflow,
        "particle_count": len(values["images"]),
        "image_shape": list(values["images"].shape[1:]),
        "input_sha256": input_hashes,
        "frozen_inputs_sha256": sha256(base.with_suffix(base.suffix + ".inputs.npz")),
        "warmup_seconds": warm_seconds,
        "unprofiled_wall_seconds": seconds,
        "unprofiled_median_seconds": median,
        "particle_iterations_per_second": len(values["images"])
        * config.max_iterations
        / median,
        "io_seconds": {
            "load_and_hash": input_seconds,
            "write_result_npz": output_seconds,
        },
        "profile_parity": errors,
        "performance": profiled.metadata["performance"],
        "gpu_memory_plans": profiled.metadata.get("gpu_memory_plans", []),
        "gpu_workspace": profiled.metadata.get("gpu_workspace"),
        "halfset_update_policy": profiled.metadata.get("halfset_update_policy"),
        "class_average_metadata": class_average_metadata,
        "result": str(result_path),
        "result_sha256": sha256(result_path),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda", "cupy"), default="cuda")
    parser.add_argument("--suite", choices=("smoke", "representative"), default="smoke")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--capture-candidates", action="store_true",
                        help="Save full pre-centering E-step arrays during the profiled run")
    parser.add_argument(
        "--only", help="Comma-separated cases; unknown names are errors"
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cases = SMOKE_CASES if args.suite == "smoke" else REAL_CASES
    selected = set(args.only.split(",")) if args.only else set(cases)
    if (
        not selected <= set(cases)
        or args.repeats < 1
        or args.batch_size < 1
        or not 0 < args.memory_fraction <= 1
    ):
        parser.error("invalid cases, repeats, batch size, or memory fraction")
    if args.output.exists():
        parser.error("output already exists; choose a new report name (no overwrite)")
    report = {
        "schema": "alignimg.performance.v1",
        "stage": 5,
        "started_at_utc": utc_now(),
        "status": "running",
        "parameters": vars(args),
        "cases": {},
    }
    write_report(args.output, report)
    try:
        report["source"] = source_manifest()
        report["environment"] = environment_info(args.backend, args.device)
        report["installed_core"] = installed_source_info(
            ai, "src/alignimg/", report["source"]["files"]
        )
        report["build_environment"] = {
            name: os.environ.get(name)
            for name in (
                "CUDACXX",
                "CMAKE_ARGS",
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        }
        if args.backend != "cpu":
            import alignimg_gpu
            from alignimg_gpu.backend import _native_module

            if alignimg_gpu.__version__ != ai.__version__:
                raise RuntimeError("core/GPU version mismatch; reinstall both packages")
            report["installed_gpu_source"] = str(alignimg_gpu.__file__)
            report["installed_gpu"] = installed_source_info(
                alignimg_gpu,
                "packages/alignimg-gpu/src/alignimg_gpu/",
                report["source"]["files"],
            )
            try:
                compiler = subprocess.run(
                    ["nvcc", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                report["nvcc_on_path"] = compiler.stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                report["nvcc_on_path"] = None
            if args.backend == "cuda":
                native = _native_module()
                if (
                    native is None
                    or getattr(native, "__version__", None) != ai.__version__
                ):
                    raise RuntimeError(
                        "native CUDA build is missing or stale; rebuild alignimg-gpu"
                    )
                report["native_build"] = {
                    "version": native.__version__,
                    "binary_sha256": sha256(Path(native.__file__)),
                    **native.runtime_info(),
                }
        report["frozen_fixture_check"] = frozen_fixture_check(args.baseline)
        report["baseline_manifest_sha256"] = sha256(args.baseline / "manifest.json")
        write_report(args.output, report)
        for name in cases:
            if name not in selected:
                continue
            print(
                f"RUN  {name}: warm-up + {args.repeats} unprofiled + 1 profiled",
                flush=True,
            )
            try:
                report["cases"][name] = {"status": "passed", **run_case(name, args)}
                print(f"PASS {name}", flush=True)
            except Exception as error:
                report["cases"][name] = {
                    "status": "failed",
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                print(f"FAIL {name}: {error}", flush=True)
            write_report(args.output, report)
        failed = sum(case["status"] == "failed" for case in report["cases"].values())
        report["status"] = "failed" if failed else "completed"
        report["summary"] = {
            "failed": failed,
            "passed": len(report["cases"]) - failed,
            "performance_acceptance": "baseline_only",
            "scientific_review": "pending",
        }
    except Exception as error:
        report.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
    report["completed_at_utc"] = utc_now()
    if platform.system() in {"Linux", "Darwin"}:
        import resource

        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        report["process_lifetime_peak_rss_bytes"] = rss * (
            1024 if platform.system() == "Linux" else 1
        )
    write_report(args.output, report)
    print(f"Report saved to: {args.output}", flush=True)
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
