#!/usr/bin/env python3
"""Freeze and benchmark the AlignImg 2.2 fast-hard comparison workloads."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any

import numpy as np

import alignimg as ai
from alignimg._transform import transform_image
from alignimg._polar_hard import polar_radius, translation_center_grid

try:
    from tools.performance_fixtures import ROOT, sha256, source_manifest
    from tools.performance_validation import compare_arrays, result_arrays, synchronize
    from tools.re2dc_70s_pose_benchmark import (
        DEFAULT_BENCHMARK,
        image_correlation,
        load_benchmark,
        wrapped_angle,
    )
    from tools.server_validation import environment_info, utc_now, write_report
except ModuleNotFoundError:
    from performance_fixtures import ROOT, sha256, source_manifest
    from performance_validation import compare_arrays, result_arrays, synchronize
    from re2dc_70s_pose_benchmark import (
        DEFAULT_BENCHMARK,
        image_correlation,
        load_benchmark,
        wrapped_angle,
    )
    from server_validation import environment_info, utc_now, write_report


SCHEMA = "alignimg.fast-hard-validation.v1"
COMPARISON_SCHEMA = "alignimg.fast-hard-validation.v2"
POLAR_COMPARISON_SCHEMA = "alignimg.fast-hard-validation.v3"
POLAR_GPU_COMPARISON_SCHEMA = "alignimg.fast-hard-validation.v4"
INPUT_SCHEMA = "alignimg.fast-hard-inputs.v1"
SUITES = ("synthetic", "homogeneous", "mra1000")
DEFAULT_MRA_STACK = ROOT / "data/re2dc_70s_testdata/prepared/re2dc_70s_n1000_s128.mrcs"
DEFAULT_MRA_REFERENCES = (
    ROOT / "validation-results/re2dc-70s-final-n1000.rf.seed-0.result.npz"
)


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def polar_result_arrays(result: ai.AlignmentResult) -> dict[str, np.ndarray]:
    values = result_arrays(result)
    if result._polar_raw_shift_y_values is not None:
        values["polar_raw_shift_y_px"] = np.asarray(
            result._polar_raw_shift_y_values, dtype=np.float32
        )
        values["polar_raw_shift_x_px"] = np.asarray(
            result._polar_raw_shift_x_values, dtype=np.float32
        )
    return values


def result_hashes(result: ai.AlignmentResult) -> dict[str, str]:
    values = polar_result_arrays(result)
    hashes = {name: _array_hash(value) for name, value in sorted(values.items())}
    hashes["combined"] = _json_hash(hashes)
    return hashes


def _case_hashes(case: dict[str, np.ndarray]) -> dict[str, str]:
    hashes = {name: _array_hash(value) for name, value in sorted(case.items())}
    hashes["combined"] = _json_hash(hashes)
    return hashes


def baseline_config(
    *, batch_size: int, memory_fraction: float, profile_execution: bool = False
) -> ai.AlignmentConfig:
    """Return the frozen Stage 0 side of the future balanced/fast A/B."""
    return replace(
        ai.AlignmentConfig.preset("global_balanced"),
        max_iterations=3,
        translation_range=4.0,
        translation_step=1.0,
        robust_weighting=False,
        halfset_diagnostics=False,
        apply_final_pose_to_raw=False,
        batch_size=batch_size,
        memory_fraction=memory_fraction,
        profile_execution=profile_execution,
    ).normalized(workflow="global")


def fast_hard_config(
    *,
    batch_size: int,
    memory_fraction: float,
    profile_execution: bool = False,
    proposal_angles_per_reference: int = 4,
) -> ai.AlignmentConfig:
    """Return the private Stage 1 config-only hard candidate."""
    return replace(
        baseline_config(
            batch_size=batch_size,
            memory_fraction=memory_fraction,
            profile_execution=profile_execution,
        ),
        top_l=1,
        proposal_angles_per_reference=proposal_angles_per_reference,
    ).normalized(workflow="global")


def polar_hard_config(
    *, batch_size: int, memory_fraction: float, profile_execution: bool = False
) -> ai.AlignmentConfig:
    """Return the private Stage 2 CPU-authoritative true-polar candidate."""
    return replace(
        baseline_config(
            batch_size=batch_size,
            memory_fraction=memory_fraction,
            profile_execution=profile_execution,
        ),
        search_strategy="polar_hard",
        candidate_scoring="polar",
        score_model="polar_ring_ccf",
        top_l=1,
        proposal_angles_per_reference=None,
    ).normalized(workflow="global")


def preset_snapshot() -> dict[str, dict[str, Any]]:
    return {
        name: asdict(ai.AlignmentConfig.preset(name))
        for name in ("global_balanced", "reference_free", "refine")
    }


def git_revision() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        return {
            "available": True,
            "commit": run("rev-parse", "HEAD"),
            "short_commit": run("rev-parse", "--short", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
            "error": None,
        }
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return {
            "available": False,
            "commit": None,
            "short_commit": None,
            "dirty": None,
            "error": str(error),
        }


def runtime_identity(backend: str) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "core": {
            "version": ai.__version__,
            "module": str(Path(ai.__file__).resolve()),
        },
        "gpu": None,
        "native": None,
    }
    if importlib.util.find_spec("alignimg_gpu") is None:
        if backend != "cpu":
            raise RuntimeError("alignimg-gpu is required for the requested backend")
        return identity
    import alignimg_gpu

    identity["gpu"] = {
        "version": alignimg_gpu.__version__,
        "module": str(Path(alignimg_gpu.__file__).resolve()),
    }
    if backend != "cpu" and alignimg_gpu.__version__ != ai.__version__:
        raise RuntimeError("core and GPU package versions differ")
    from alignimg_gpu.backend import _native_module

    native = _native_module()
    if native is not None:
        identity["native"] = {
            "version": str(native.__version__),
            "module": str(Path(native.__file__).resolve()),
            "binary_sha256": sha256(Path(native.__file__)),
            "runtime": native.runtime_info(),
        }
    if backend == "cuda" and (
        native is None or str(native.__version__) != ai.__version__
    ):
        raise RuntimeError(
            "native CUDA extension is missing or has a different version"
        )
    return identity


def _reference_stack(size: int, count: int) -> np.ndarray:
    y, x = np.indices((size, size), dtype=np.float32)
    references = []
    centers = (
        ((0.30, 0.66, 1.0), (0.68, 0.33, 0.72), (0.47, 0.48, 0.35)),
        ((0.26, 0.28, 1.0), (0.66, 0.70, 0.65), (0.72, 0.40, 0.42)),
        ((0.30, 0.72, 0.82), (0.72, 0.62, 1.0), (0.57, 0.25, 0.38)),
    )
    for component in range(count):
        value = np.zeros((size, size), dtype=np.float32)
        for cy, cx, amplitude in centers[component]:
            width = size * (0.075 + 0.012 * component)
            value += amplitude * np.exp(
                -((y - cy * size) ** 2 + (x - cx * size) ** 2) / (2.0 * width**2)
            )
        references.append(value)
    return np.stack(references).astype(np.float32)


def _inverse_pose(
    angle: float, shift_y: float, shift_x: float, mirror: bool
) -> tuple[float, float, float, bool]:
    radians = np.deg2rad(angle)
    rotation = np.asarray(
        [[np.cos(radians), np.sin(radians)], [-np.sin(radians), np.cos(radians)]],
        dtype=np.float64,
    )
    reflection = np.asarray([[-1.0, 0.0], [0.0, 1.0]])
    linear = rotation @ (reflection if mirror else np.eye(2))
    inverse_linear = linear.T
    inverse_shift = -inverse_linear @ np.asarray([shift_x, shift_y], dtype=np.float64)
    inverse_angle = angle if mirror else -angle
    return inverse_angle, float(inverse_shift[1]), float(inverse_shift[0]), mirror


def _particles_from_truth(
    references: np.ndarray,
    component: np.ndarray,
    poses: ai.PoseSet,
) -> np.ndarray:
    output = np.empty((len(component), *references.shape[1:]), dtype=np.float32)
    for index, reference_index in enumerate(component):
        inverse = _inverse_pose(
            float(poses.angle_deg[index]),
            float(poses.shift_y_px[index]),
            float(poses.shift_x_px[index]),
            bool(poses.mirror[index]),
        )
        output[index] = transform_image(
            references[int(reference_index)],
            angle_deg=inverse[0],
            shift_y_px=inverse[1],
            shift_x_px=inverse[2],
            mirror=inverse[3],
        )
    return output


def _case_values(
    images: np.ndarray,
    references: np.ndarray,
    component: np.ndarray | None = None,
    poses: ai.PoseSet | None = None,
    priors: np.ndarray | None = None,
    *,
    mirror_search: bool = False,
) -> dict[str, np.ndarray]:
    values = {
        "images": np.asarray(images, dtype=np.float32),
        "references": np.asarray(references, dtype=np.float32),
        "component": np.asarray([] if component is None else component, dtype=np.int32),
        "priors": np.asarray([] if priors is None else priors, dtype=np.float32),
        "mirror_search": np.asarray(mirror_search, dtype=np.bool_),
    }
    if poses is None:
        values.update(
            truth_angle_deg=np.asarray([], dtype=np.float32),
            truth_shift_y_px=np.asarray([], dtype=np.float32),
            truth_shift_x_px=np.asarray([], dtype=np.float32),
            truth_mirror=np.asarray([], dtype=np.bool_),
        )
    else:
        values.update(
            truth_angle_deg=poses.angle_deg,
            truth_shift_y_px=poses.shift_y_px,
            truth_shift_x_px=poses.shift_x_px,
            truth_mirror=poses.mirror,
        )
    return values


def synthetic_cases() -> dict[str, dict[str, np.ndarray]]:
    size = 48
    k1_reference = _reference_stack(size, 1)
    angles = np.asarray([0.0, 90.0, -37.5, 143.0], dtype=np.float32)
    shift_y = np.asarray([0.0, 2.0, -1.0, 0.5], dtype=np.float32)
    shift_x = np.asarray([0.0, -1.0, 2.0, -1.5], dtype=np.float32)
    component = np.zeros(len(angles), dtype=np.int32)
    cases: dict[str, dict[str, np.ndarray]] = {}
    for mirrored in (False, True):
        poses = ai.PoseSet(
            angles,
            shift_y,
            shift_x,
            np.full(len(angles), mirrored, dtype=np.bool_),
        )
        name = "k1_mirror_on" if mirrored else "k1_mirror_off"
        cases[name] = _case_values(
            _particles_from_truth(k1_reference, component, poses),
            k1_reference,
            component,
            poses,
            mirror_search=mirrored,
        )

    references = _reference_stack(size, 3)
    rng = np.random.default_rng(20260915)
    component = np.repeat(np.arange(3, dtype=np.int32), 12)
    poses = ai.PoseSet(
        rng.uniform(-180.0, 180.0, len(component)).astype(np.float32),
        rng.uniform(-3.0, 3.0, len(component)).astype(np.float32),
        rng.uniform(-3.0, 3.0, len(component)).astype(np.float32),
        np.zeros(len(component), dtype=np.bool_),
    )
    clean = _particles_from_truth(references, component, poses)
    for name, snr in (("k3_noiseless", None), ("k3_snr_0_5", 0.5), ("k3_snr_0_2", 0.2)):
        images = clean.copy()
        if snr is not None:
            noise = rng.normal(size=images.shape).astype(np.float32)
            signal_scale = np.std(images, axis=(1, 2), keepdims=True)
            noise_scale = np.std(noise, axis=(1, 2), keepdims=True)
            images += (
                noise * signal_scale / np.maximum(noise_scale * np.sqrt(snr), 1e-8)
            )
        cases[name] = _case_values(images, references, component, poses)
    return cases


def homogeneous_cases() -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    _, particles, references, truth, poses = load_benchmark(ROOT / DEFAULT_BENCHMARK)
    component = np.asarray(truth["component_index"], dtype=np.int32)
    cases = {}
    for index in range(len(references)):
        selected = component == index
        cases[f"k1_class_{index}"] = _case_values(
            particles[selected],
            references[index : index + 1],
            np.zeros(np.sum(selected), dtype=np.int32),
            ai.PoseSet(
                poses.angle_deg[selected],
                poses.shift_y_px[selected],
                poses.shift_x_px[selected],
                poses.mirror[selected],
            ),
        )
    cases["fixed_k3"] = _case_values(
        particles,
        references,
        component,
        poses,
        ai.make_class_priors(assignments=component, n_components=len(references)),
    )
    manifest = ROOT / DEFAULT_BENCHMARK
    sources = {str(manifest.relative_to(ROOT)): sha256(manifest)}
    for value in json.loads(manifest.read_text())["artifacts"].values():
        path = ROOT / value
        sources[str(path.relative_to(ROOT))] = sha256(path)
    return cases, sources


def mra1000_cases() -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    import mrcfile

    with mrcfile.mmap(DEFAULT_MRA_STACK, permissive=True, mode="r") as mrc:
        images = np.asarray(mrc.data, dtype=np.float32).copy()
    with np.load(DEFAULT_MRA_REFERENCES, allow_pickle=False) as saved:
        references = np.asarray(saved["reference_history"][-1], dtype=np.float32)
    cases = {"open_k10": _case_values(images, references)}
    sources = {
        str(DEFAULT_MRA_STACK.relative_to(ROOT)): sha256(DEFAULT_MRA_STACK),
        str(DEFAULT_MRA_REFERENCES.relative_to(ROOT)): sha256(DEFAULT_MRA_REFERENCES),
    }
    return cases, sources


def _encode_cases(
    suite: str, cases: dict[str, dict[str, np.ndarray]], sources: dict[str, str]
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {
        "schema": np.asarray(INPUT_SCHEMA),
        "suite": np.asarray(suite),
        "case_names_json": np.asarray(json.dumps(list(cases))),
        "source_sha256_json": np.asarray(json.dumps(sources, sort_keys=True)),
    }
    for case_name, case in cases.items():
        for name, value in case.items():
            values[f"{case_name}__{name}"] = value
    return values


def _decode_cases(values: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    names = json.loads(str(np.asarray(values["case_names_json"]).item()))
    return {
        case_name: {
            name.split("__", 1)[1]: value
            for name, value in values.items()
            if name.startswith(f"{case_name}__")
        }
        for case_name in names
    }


def freeze_inputs(suite: str, path: Path) -> dict[str, Any]:
    """Create a workload archive once, or validate and reuse the existing archive."""
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            values = {name: saved[name].copy() for name in saved.files}
        if str(values["schema"].item()) != INPUT_SCHEMA:
            raise ValueError(f"unsupported frozen input schema: {path}")
        if str(values["suite"].item()) != suite:
            raise ValueError(f"frozen input suite does not match {suite}: {path}")
    else:
        if suite == "synthetic":
            cases, sources = synthetic_cases(), {}
        elif suite == "homogeneous":
            cases, sources = homogeneous_cases()
        elif suite == "mra1000":
            cases, sources = mra1000_cases()
        else:
            raise ValueError(f"unknown suite: {suite}")
        values = _encode_cases(suite, cases, sources)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **values)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "source_sha256": json.loads(str(values["source_sha256_json"].item())),
        "cases": _decode_cases(values),
    }


def _truth_poses(case: dict[str, np.ndarray]) -> ai.PoseSet | None:
    if not len(case["truth_angle_deg"]):
        return None
    return ai.PoseSet(
        case["truth_angle_deg"],
        case["truth_shift_y_px"],
        case["truth_shift_x_px"],
        case["truth_mirror"],
    )


def _fit_common_gauge(
    predicted: ai.PoseSet, expected: ai.PoseSet
) -> tuple[ai.PoseSet, ai.PoseSet]:
    """Fit and apply one orientation-preserving gauge, including mirrored poses."""
    delta = wrapped_angle(predicted.angle_deg - expected.angle_deg)
    circular_mean = np.rad2deg(
        np.arctan2(
            np.mean(np.sin(np.deg2rad(delta))),
            np.mean(np.cos(np.deg2rad(delta))),
        )
    )
    unwrapped = circular_mean + wrapped_angle(delta - circular_mean)
    gauge_angle = float(wrapped_angle(np.asarray([np.median(unwrapped)]))[0])
    radians = np.deg2rad(gauge_angle)
    rotation = np.asarray(
        [[np.cos(radians), np.sin(radians)], [-np.sin(radians), np.cos(radians)]],
        dtype=np.float64,
    )
    expected_shift = np.column_stack((expected.shift_x_px, expected.shift_y_px)).astype(
        np.float64
    )
    predicted_shift = np.column_stack(
        (predicted.shift_x_px, predicted.shift_y_px)
    ).astype(np.float64)
    residual = predicted_shift - expected_shift @ rotation.T
    gauge_shift = np.median(residual, axis=0)
    gauge = ai.PoseSet(
        np.asarray([gauge_angle], dtype=np.float32),
        np.asarray([gauge_shift[1]], dtype=np.float32),
        np.asarray([gauge_shift[0]], dtype=np.float32),
        np.asarray([False]),
    )
    composed_shift = expected_shift @ rotation.T + gauge_shift
    gauged = ai.PoseSet(
        wrapped_angle(expected.angle_deg + gauge_angle).astype(np.float32),
        composed_shift[:, 1].astype(np.float32),
        composed_shift[:, 0].astype(np.float32),
        expected.mirror,
    )
    return gauge, gauged


def _pose_quality(
    result: ai.AlignmentResult, case: dict[str, np.ndarray], backend: str
) -> dict[str, Any] | None:
    truth = _truth_poses(case)
    if truth is None:
        return None
    component = case["component"]
    angle_errors: list[np.ndarray] = []
    shift_errors: list[np.ndarray] = []
    reference_correlations = []
    raw_average_correlations = []
    for index in range(len(case["references"])):
        selected = component == index
        predicted = ai.PoseSet(
            result.poses.angle_deg[selected],
            result.poses.shift_y_px[selected],
            result.poses.shift_x_px[selected],
            result.poses.mirror[selected],
        )
        expected = ai.PoseSet(
            truth.angle_deg[selected],
            truth.shift_y_px[selected],
            truth.shift_x_px[selected],
            truth.mirror[selected],
        )
        gauge, gauged = _fit_common_gauge(predicted, expected)
        angle = np.abs(wrapped_angle(predicted.angle_deg - gauged.angle_deg))
        shift = np.hypot(
            predicted.shift_y_px - gauged.shift_y_px,
            predicted.shift_x_px - gauged.shift_x_px,
        )
        target = ai.transform_images(
            case["references"][index : index + 1], gauge, backend="cpu"
        )[0]
        angle_errors.append(angle)
        shift_errors.append(shift)
        reference_correlations.append(
            image_correlation(result.references[index], target)
        )
        aligned = ai.transform_images(
            case["images"][selected], predicted, backend=backend
        )
        raw_average_correlations.append(image_correlation(aligned.mean(axis=0), target))
    angle = np.concatenate(angle_errors)
    shift = np.concatenate(shift_errors)
    return {
        "angle_absolute_error_deg": {
            "median": float(np.median(angle)),
            "p95": float(np.quantile(angle, 0.95)),
        },
        "shift_magnitude_error_px": {
            "median": float(np.median(shift)),
            "p95": float(np.quantile(shift, 0.95)),
        },
        "reference_correlation": reference_correlations,
        "aligned_raw_class_average_correlation": raw_average_correlations,
        "mirror_accuracy": float(np.mean(result.poses.mirror == truth.mirror)),
    }


def _estimated_search_counts(
    case: dict[str, np.ndarray], config: ai.AlignmentConfig
) -> dict[str, int]:
    priors = case["priors"]
    if priors.size:
        active = np.count_nonzero(priors > 0.0, axis=1)
    else:
        active = np.full(len(case["images"]), len(case["references"]), dtype=np.int64)
    if config.search_strategy == "polar_hard":
        size = int(case["images"].shape[1])
        centers, _ = translation_center_grid(
            0.0,
            0.0,
            size=size,
            radius=polar_radius(size, config),
            translation_range=config.translation_range,
            translation_step=config.translation_step,
        )
        mirrors = 2 if config.mirror_search else 1
        polar_count = int(
            np.sum(active)
            * len(centers)
            * config.angle_samples
            * mirrors
            * config.max_iterations
        )
        return {
            "polar_proposal_count": polar_count,
            "exact_rescoring_count": 0,
            "retained_candidate_count": int(
                len(case["images"]) * config.max_iterations
            ),
        }
    angles = int(config.proposal_angles_per_reference or 0)
    mirrors = 2 if config.mirror_search else 1
    proposal_count = int(np.sum(active) * angles * mirrors * config.max_iterations)
    rescoring = int(
        np.sum([count * int(np.ceil(config.top_l / count)) for count in active])
        * angles
        * mirrors
        * config.max_iterations
    )
    return {
        "polar_proposal_count": proposal_count,
        "exact_rescoring_count": rescoring,
        "retained_candidate_count": int(
            len(case["images"]) * config.top_l * config.max_iterations
        ),
    }


def _check_result(
    result: ai.AlignmentResult, backend: str, case: dict[str, np.ndarray]
) -> None:
    if result.metadata.get("backend") != backend:
        raise AssertionError("backend fallback is not allowed")
    for name in ("references", "responsibilities", "inlier_weights"):
        if not np.all(np.isfinite(getattr(result, name))):
            raise AssertionError(f"non-finite {name}")
    row_error = np.max(np.abs(result.responsibilities.sum(axis=1) - 1.0))
    if row_error > 2e-6:
        raise AssertionError(f"responsibility row-sum error is {row_error}")
    priors = case["priors"]
    if priors.size and np.all(np.count_nonzero(priors > 0.0, axis=1) == 1):
        expected = np.argmax(priors, axis=1)
        if not np.array_equal(result.reference_assignments, expected):
            raise AssertionError("fixed one-hot priors changed reference assignments")
    if result.metadata.get("search_strategy") == "polar_hard" and backend in {
        "cuda",
        "cupy",
    }:
        expected = "native_cuda" if backend == "cuda" else "cupy"
        if result.metadata.get("polar_sampler_backend") != expected:
            raise AssertionError("polar sampler backend attribution is incorrect")
        if result.metadata.get("polar_peak_backend") != expected:
            raise AssertionError("polar peak backend attribution is incorrect")
        if result.metadata.get("polar_full_correlation_map_d2h") is not False:
            raise AssertionError("polar correlation maps must remain on device")


def _check_hard_result(result: ai.AlignmentResult) -> None:
    responsibilities = np.asarray(result.responsibilities)
    expected = np.eye(responsibilities.shape[1], dtype=responsibilities.dtype)[
        result.reference_assignments
    ]
    if not np.array_equal(responsibilities, expected):
        raise AssertionError("top_l=1 must produce one-hot responsibilities")
    if result.candidates.posterior.shape[1] != 1 or not np.array_equal(
        result.candidates.posterior,
        np.ones_like(result.candidates.posterior),
    ):
        raise AssertionError("top_l=1 must retain one posterior with weight one")


def _execute(
    case: dict[str, np.ndarray], config: ai.AlignmentConfig, backend: str
) -> ai.AlignmentResult:
    priors = case["priors"] if case["priors"].size else None
    return ai.align_to_references(
        case["images"],
        case["references"],
        class_priors=priors,
        config=config,
        backend=backend,
    )


def run_case(
    name: str,
    case: dict[str, np.ndarray],
    *,
    config: ai.AlignmentConfig,
    backend: str,
    measured_repeats: int,
    profile_execution: bool,
    output: Path,
    variant: str = "global_balanced_2.2_baseline",
    result_label: str = "baseline",
    profile_repeats: int = 1,
) -> dict[str, Any]:
    synchronize(backend)
    started = time.perf_counter()
    warmup = _execute(case, config, backend)
    synchronize(backend)
    warmup_seconds = time.perf_counter() - started
    _check_result(warmup, backend, case)
    if config.top_l == 1:
        _check_hard_result(warmup)

    results = []
    seconds = []
    hashes = []
    for _ in range(measured_repeats):
        synchronize(backend)
        started = time.perf_counter()
        current = _execute(case, config, backend)
        synchronize(backend)
        seconds.append(time.perf_counter() - started)
        _check_result(current, backend, case)
        if config.top_l == 1:
            _check_hard_result(current)
        results.append(current)
        hashes.append(result_hashes(current))
    primary = results[0]
    parity = [
        compare_arrays(polar_result_arrays(primary), polar_result_arrays(value))
        for value in results[1:]
    ]

    profiled = None
    profile_parities = []
    profiled_candidate_inference_seconds = []
    if profile_execution:
        for _ in range(profile_repeats):
            synchronize(backend)
            current = _execute(case, replace(config, profile_execution=True), backend)
            synchronize(backend)
            _check_result(current, backend, case)
            if config.top_l == 1:
                _check_hard_result(current)
            profile_parities.append(
                compare_arrays(
                    polar_result_arrays(primary), polar_result_arrays(current)
                )
            )
            performance = current.metadata.get("performance")
            if performance is None:
                raise AssertionError("profiled execution did not return performance")
            profiled_candidate_inference_seconds.append(
                float(
                    performance["stages"][
                        "workflow/alignment_engine/candidate_inference"
                    ]["wall_seconds"]
                )
            )
            if profiled is None:
                profiled = current

    result_path = output.with_suffix("").with_name(
        f"{output.stem}.{name}.{result_label}.result.npz"
    )
    if result_path.exists():
        raise FileExistsError(f"result already exists: {result_path}")
    np.savez_compressed(result_path, **polar_result_arrays(primary))
    responsibilities = primary.responsibilities
    occupancy = np.bincount(
        primary.reference_assignments, minlength=len(primary.references)
    )
    performance = None if profiled is None else profiled.metadata.get("performance")
    backend_metadata = {
        name: primary.metadata.get(name)
        for name in (
            "engine",
            "backend",
            "gpu_device",
            "polar_sampler_backend",
            "polar_peak_backend",
            "polar_full_correlation_map_d2h",
            "gpu_policy",
        )
    }
    return {
        "variant": variant,
        "config": asdict(config),
        "config_sha256": _json_hash(asdict(config)),
        "particle_count": len(case["images"]),
        "reference_count": len(case["references"]),
        "image_shape": list(case["images"].shape[1:]),
        "input_sha256": _case_hashes(case),
        "warmup_seconds": warmup_seconds,
        "unprofiled_wall_seconds": seconds,
        "unprofiled_median_seconds": float(np.median(seconds)),
        "deterministic_hashes": hashes,
        "deterministic_exact_match": len({item["combined"] for item in hashes}) == 1,
        "repeat_parity": parity,
        "profile_parity": profile_parities[0] if profile_parities else None,
        "profile_repeat_parity": profile_parities,
        "profiled_candidate_inference_seconds": (profiled_candidate_inference_seconds),
        "profiled_candidate_inference_median_seconds": (
            None
            if not profiled_candidate_inference_seconds
            else float(np.median(profiled_candidate_inference_seconds))
        ),
        "search_counts": _estimated_search_counts(case, config),
        "responsibility_row_sum_max_error": float(
            np.max(np.abs(responsibilities.sum(axis=1) - 1.0))
        ),
        "mean_max_responsibility": float(np.mean(np.max(responsibilities, axis=1))),
        "assignments": primary.reference_assignments,
        "occupancy": occupancy,
        "quality": _pose_quality(primary, case, backend),
        "iteration_seconds": [item["seconds"] for item in primary.diagnostics],
        "performance": performance,
        "backend_metadata": backend_metadata,
        "gpu_memory_plan_at_completion": primary.metadata.get(
            "gpu_memory_plan_at_completion"
        ),
        "gpu_memory_plans": primary.metadata.get("gpu_memory_plans", []),
        "gpu_workspace": primary.metadata.get("gpu_workspace"),
        "result": str(result_path),
        "result_sha256": sha256(result_path),
    }


def compare_variants(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Summarize the Stage 1 candidate relative to its same-process baseline."""
    with np.load(baseline["result"], allow_pickle=False) as saved:
        baseline_result = {name: saved[name].copy() for name in saved.files}
    with np.load(candidate["result"], allow_pickle=False) as saved:
        candidate_result = {name: saved[name].copy() for name in saved.files}

    angle_delta = (
        candidate_result["angle_deg"] - baseline_result["angle_deg"] + 180.0
    ) % 360.0 - 180.0
    shift_delta = np.hypot(
        candidate_result["shift_y_px"] - baseline_result["shift_y_px"],
        candidate_result["shift_x_px"] - baseline_result["shift_x_px"],
    )
    reference_correlations = [
        image_correlation(actual, expected)
        for actual, expected in zip(
            candidate_result["references"], baseline_result["references"]
        )
    ]
    baseline_candidate_seconds = baseline["profiled_candidate_inference_median_seconds"]
    candidate_candidate_seconds = candidate[
        "profiled_candidate_inference_median_seconds"
    ]
    return {
        "wall_speedup": baseline["unprofiled_median_seconds"]
        / candidate["unprofiled_median_seconds"],
        "candidate_inference_speedup": (
            None
            if baseline_candidate_seconds is None or candidate_candidate_seconds is None
            else baseline_candidate_seconds / candidate_candidate_seconds
        ),
        "assignment_exact_match": bool(
            np.array_equal(
                candidate_result["assignments"], baseline_result["assignments"]
            )
        ),
        "assignment_match_fraction": float(
            np.mean(candidate_result["assignments"] == baseline_result["assignments"])
        ),
        "pose_delta": {
            "angle_absolute_median_deg": float(np.median(np.abs(angle_delta))),
            "angle_absolute_p95_deg": float(np.quantile(np.abs(angle_delta), 0.95)),
            "angle_absolute_max_deg": float(np.max(np.abs(angle_delta))),
            "shift_magnitude_median_px": float(np.median(shift_delta)),
            "shift_magnitude_p95_px": float(np.quantile(shift_delta, 0.95)),
            "shift_magnitude_max_px": float(np.max(shift_delta)),
            "mirror_exact_match": bool(
                np.array_equal(candidate_result["mirror"], baseline_result["mirror"])
            ),
        },
        "reference_correlation": reference_correlations,
        "search_count_ratio": {
            name: candidate["search_counts"][name] / baseline["search_counts"][name]
            for name in baseline["search_counts"]
        },
    }


def compare_polar_backend_parity(
    reference_path: Path,
    actual_path: Path,
    case: dict[str, np.ndarray],
    config: ai.AlignmentConfig,
) -> dict[str, Any]:
    """Compare final hard-polar inference to an accepted backend result."""
    with np.load(reference_path, allow_pickle=False) as saved:
        reference = {name: saved[name].copy() for name in saved.files}
    with np.load(actual_path, allow_pickle=False) as saved:
        actual = {name: saved[name].copy() for name in saved.files}

    def angle_error(name: str) -> float:
        delta = (actual[name] - reference[name] + 180.0) % 360.0 - 180.0
        return float(np.max(np.abs(delta)))

    def absolute_error(name: str) -> float:
        return float(np.max(np.abs(actual[name] - reference[name])))

    raw_shift_names = (
        "polar_raw_shift_y_px",
        "polar_raw_shift_x_px",
    )
    if any(name not in reference or name not in actual for name in raw_shift_names):
        raise ValueError(
            "polar parity artifacts must include raw translation-center arrays"
        )

    score_error = np.abs(actual["candidate_score"] - reference["candidate_score"])
    score_scale = np.maximum(np.abs(reference["candidate_score"]), 1e-12)
    reference_indices = reference["candidate_reference_index"][:, 0]
    actual_indices = actual["candidate_reference_index"][:, 0]
    priors = case["priors"]
    if not priors.size:
        priors = np.full(
            (len(case["images"]), len(case["references"])),
            1.0 / len(case["references"]),
            dtype=np.float64,
        )
    else:
        priors = np.asarray(priors, dtype=np.float64)
        priors /= priors.sum(axis=1, keepdims=True)
    rows = np.arange(len(priors))
    anneal_iterations = (
        config.max_iterations
        if config.temperature_anneal_iterations is None
        else config.temperature_anneal_iterations
    )
    if anneal_iterations <= 1:
        final_temperature = float(config.temperature_end)
    else:
        fraction = min((config.max_iterations - 1) / (anneal_iterations - 1), 1.0)
        final_temperature = float(
            config.temperature_start
            * (config.temperature_end / config.temperature_start) ** fraction
        )
    reference_objective = reference["candidate_score"][
        :, 0
    ] / final_temperature + np.log(priors[rows, reference_indices])
    actual_objective = actual["candidate_score"][:, 0] / final_temperature + np.log(
        priors[rows, actual_indices]
    )
    tolerances = {
        "angle_deg": 1e-3,
        "shift_px": 1e-5,
        "score_absolute": 2e-5,
        "objective_absolute": 1e-3,
    }
    errors = {
        "pose_angle_max_deg": angle_error("angle_deg"),
        "candidate_angle_max_deg": angle_error("candidate_angle_deg"),
        "pose_shift_y_max_px": absolute_error("shift_y_px"),
        "pose_shift_x_max_px": absolute_error("shift_x_px"),
        "candidate_shift_y_max_px": absolute_error("candidate_shift_y_px"),
        "candidate_shift_x_max_px": absolute_error("candidate_shift_x_px"),
        "raw_shift_y_max_px": absolute_error("polar_raw_shift_y_px"),
        "raw_shift_x_max_px": absolute_error("polar_raw_shift_x_px"),
        "score_absolute_max": float(np.max(score_error)),
        "score_relative_max": float(np.max(score_error / score_scale)),
        "objective_absolute_max": float(
            np.max(np.abs(actual_objective - reference_objective))
        ),
    }
    exact = {
        "assignments": bool(
            np.array_equal(actual["assignments"], reference["assignments"])
        ),
        "pose_mirror": bool(np.array_equal(actual["mirror"], reference["mirror"])),
        "candidate_reference": bool(
            np.array_equal(
                actual["candidate_reference_index"],
                reference["candidate_reference_index"],
            )
        ),
        "candidate_mirror": bool(
            np.array_equal(actual["candidate_mirror"], reference["candidate_mirror"])
        ),
    }
    within_tolerance = (
        all(exact.values())
        and errors["pose_angle_max_deg"] <= tolerances["angle_deg"]
        and errors["candidate_angle_max_deg"] <= tolerances["angle_deg"]
        and max(errors["raw_shift_y_max_px"], errors["raw_shift_x_max_px"])
        <= tolerances["shift_px"]
        and errors["score_absolute_max"] <= tolerances["score_absolute"]
        and errors["objective_absolute_max"] <= tolerances["objective_absolute"]
    )
    return {
        "reference_result": str(reference_path),
        "reference_result_sha256": sha256(reference_path),
        "actual_result": str(actual_path),
        "actual_result_sha256": sha256(actual_path),
        "tolerances": tolerances,
        "exact": exact,
        "errors": errors,
        "within_tolerance": bool(within_tolerance),
    }


def compare_homogeneous_fixed_k3_to_k1(
    case_reports: dict[str, dict[str, Any]],
    frozen_cases: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Compare the hard fixed-K=3 result with the three independent K=1 runs."""
    required = {"fixed_k3", "k1_class_0", "k1_class_1", "k1_class_2"}
    if not required.issubset(case_reports):
        raise ValueError("homogeneous fixed-K=3/K=1 comparison needs all four cases")
    with np.load(
        case_reports["fixed_k3"]["fast_hard"]["result"], allow_pickle=False
    ) as saved:
        fixed = {name: saved[name].copy() for name in saved.files}
    component = np.asarray(frozen_cases["fixed_k3"]["component"], dtype=np.int32)
    per_class = []
    for reference_index in range(3):
        with np.load(
            case_reports[f"k1_class_{reference_index}"]["fast_hard"]["result"],
            allow_pickle=False,
        ) as saved:
            single = {name: saved[name].copy() for name in saved.files}
        selected = component == reference_index
        angle_delta = (
            fixed["angle_deg"][selected] - single["angle_deg"] + 180.0
        ) % 360.0 - 180.0
        shift_delta = np.hypot(
            fixed["shift_y_px"][selected] - single["shift_y_px"],
            fixed["shift_x_px"][selected] - single["shift_x_px"],
        )
        per_class.append(
            {
                "reference_index": reference_index,
                "particle_count": int(np.sum(selected)),
                "angle_absolute_max_deg": float(np.max(np.abs(angle_delta))),
                "shift_magnitude_max_px": float(np.max(shift_delta)),
                "mirror_exact_match": bool(
                    np.array_equal(fixed["mirror"][selected], single["mirror"])
                ),
                "assignment_contract_exact": bool(
                    np.all(fixed["assignments"][selected] == reference_index)
                    and np.all(single["assignments"] == 0)
                ),
                "reference_correlation": image_correlation(
                    fixed["references"][reference_index], single["references"][0]
                ),
            }
        )
    return {
        "per_class": per_class,
        "pose_tolerance": {"angle_deg": 1e-3, "shift_px": 1e-3},
        "pose_within_tolerance": all(
            item["angle_absolute_max_deg"] <= 1e-3
            and item["shift_magnitude_max_px"] <= 1e-3
            and item["mirror_exact_match"]
            for item in per_class
        ),
        "assignment_contract_exact": all(
            item["assignment_contract_exact"] for item in per_class
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "cupy"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--deterministic-repeats", type=int, default=2)
    parser.add_argument("--profile-execution", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument(
        "--only", help="Comma-separated case names; default runs all cases"
    )
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument(
        "--compare-fast-hard",
        action="store_true",
        help="Run the Stage 0 baseline and Stage 1 config-only hard candidate together",
    )
    parser.add_argument(
        "--compare-polar-hard",
        action="store_true",
        help="Run the baseline and Stage 2/3 true-polar candidate together",
    )
    parser.add_argument(
        "--parity-reference-report",
        type=Path,
        help="Accepted polar report whose result NPZ files are the parity reference",
    )
    parser.add_argument(
        "--proposal-ablation-2-vs-4",
        action="store_true",
        help="Run the one allowed Stage 1 hard proposal=4 versus proposal=2 ablation",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.deterministic_repeats < 1:
        parser.error("batch size and deterministic repeats must be positive")
    if not 0.0 < args.memory_fraction <= 1.0:
        parser.error("memory fraction must be in (0, 1]")
    if args.compare_fast_hard and args.compare_polar_hard:
        parser.error("choose only one comparison mode")
    comparison_requested = args.compare_fast_hard or args.compare_polar_hard
    if comparison_requested and args.freeze_only:
        parser.error("comparison modes cannot be combined with --freeze-only")
    if comparison_requested and not args.profile_execution:
        parser.error("comparison modes require --profile-execution")
    if args.parity_reference_report is not None and not args.compare_polar_hard:
        parser.error("--parity-reference-report requires --compare-polar-hard")
    if args.parity_reference_report is not None and args.backend == "cpu":
        parser.error("CPU reports cannot use --parity-reference-report")
    if (
        args.parity_reference_report is not None
        and not args.parity_reference_report.exists()
    ):
        parser.error("parity reference report does not exist")
    if args.proposal_ablation_2_vs_4 and not args.compare_fast_hard:
        parser.error("--proposal-ablation-2-vs-4 requires --compare-fast-hard")
    if args.output.exists():
        parser.error("output already exists; reports are immutable")
    if args.inputs is None:
        args.inputs = args.output.parent / "frozen-inputs" / f"{args.suite}.inputs.npz"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    comparison_requested = args.compare_fast_hard or args.compare_polar_hard
    polar_gpu = args.compare_polar_hard and args.backend != "cpu"
    parity_reference = None
    if args.parity_reference_report is not None:
        parity_reference = json.loads(args.parity_reference_report.read_text())
        if parity_reference.get("status") != "completed":
            raise ValueError("parity reference report is not completed")
    frozen = freeze_inputs(args.suite, args.inputs)
    report: dict[str, Any] = {
        "schema": (
            POLAR_GPU_COMPARISON_SCHEMA
            if polar_gpu
            else (
                POLAR_COMPARISON_SCHEMA
                if args.compare_polar_hard
                else COMPARISON_SCHEMA if args.compare_fast_hard else SCHEMA
            )
        ),
        "stage": (
            3
            if polar_gpu
            else 2 if args.compare_polar_hard else 1 if args.compare_fast_hard else 0
        ),
        "status": "frozen" if args.freeze_only else "running",
        "started_at_utc": utc_now(),
        "parameters": vars(args),
        "frozen_inputs": {
            name: value for name, value in frozen.items() if name != "cases"
        },
        "cases": {},
    }
    write_report(args.output, report)
    if args.freeze_only:
        report["completed_at_utc"] = utc_now()
        write_report(args.output, report)
        return 0
    try:
        report["environment"] = environment_info(args.backend, args.device)
        report["runtime_identity"] = runtime_identity(args.backend)
        report["git"] = git_revision()
        report["source"] = source_manifest()
        report["presets"] = preset_snapshot()
        report["preset_sha256"] = _json_hash(report["presets"])
        if args.proposal_ablation_2_vs_4:
            baseline = fast_hard_config(
                batch_size=args.batch_size,
                memory_fraction=args.memory_fraction,
                proposal_angles_per_reference=4,
            )
            baseline_variant = "config_only_fast_hard_proposal_4"
            baseline_display = "proposal_4"
        else:
            baseline = baseline_config(
                batch_size=args.batch_size,
                memory_fraction=args.memory_fraction,
            )
            baseline_variant = "global_balanced_2.2_baseline"
            baseline_display = "baseline"
        stage1_anchor = None
        if args.compare_polar_hard:
            candidate = polar_hard_config(
                batch_size=args.batch_size,
                memory_fraction=args.memory_fraction,
            )
            candidate_variant = f"polar_hard_{args.backend}"
            candidate_display = "polar_hard"
            stage1_anchor = (
                fast_hard_config(
                    batch_size=args.batch_size,
                    memory_fraction=args.memory_fraction,
                )
                if polar_gpu
                else None
            )
        else:
            candidate = fast_hard_config(
                batch_size=args.batch_size,
                memory_fraction=args.memory_fraction,
                proposal_angles_per_reference=(
                    2 if args.proposal_ablation_2_vs_4 else 4
                ),
            )
            candidate_variant = (
                "config_only_fast_hard_proposal_2"
                if args.proposal_ablation_2_vs_4
                else "config_only_fast_hard"
            )
            candidate_display = (
                "proposal_2" if args.proposal_ablation_2_vs_4 else "fast_hard"
            )
        measured_repeats = max(3, args.deterministic_repeats)
        selected = set(args.only.split(",")) if args.only else set(frozen["cases"])
        unknown = selected.difference(frozen["cases"])
        if unknown:
            raise ValueError(f"unknown cases for {args.suite}: {sorted(unknown)}")
        for name, case in frozen["cases"].items():
            if name not in selected:
                continue
            mirror_search = bool(np.asarray(case["mirror_search"]).item())
            baseline_case_config = replace(baseline, mirror_search=mirror_search)
            profile_repeats = 3 if comparison_requested else 1
            suffix = f" + {profile_repeats} profiled" if args.profile_execution else ""
            print(
                f"RUN  {name} {baseline_display}: warm-up + "
                f"{measured_repeats} measured" + suffix,
                flush=True,
            )
            baseline_result = run_case(
                name,
                case,
                config=baseline_case_config,
                backend=args.backend,
                measured_repeats=measured_repeats,
                profile_execution=args.profile_execution,
                output=args.output,
                variant=baseline_variant,
                profile_repeats=profile_repeats,
            )
            if comparison_requested:
                candidate_case_config = replace(candidate, mirror_search=mirror_search)
                anchor_result = None
                if stage1_anchor is not None:
                    anchor_case_config = replace(
                        stage1_anchor, mirror_search=mirror_search
                    )
                    print(
                        f"RUN  {name} Stage 1 hard-4 anchor: warm-up + "
                        f"{measured_repeats} measured" + suffix,
                        flush=True,
                    )
                    anchor_result = run_case(
                        name,
                        case,
                        config=anchor_case_config,
                        backend=args.backend,
                        measured_repeats=measured_repeats,
                        profile_execution=True,
                        output=args.output,
                        variant="config_only_fast_hard_stage1_anchor",
                        result_label="stage1_anchor",
                        profile_repeats=profile_repeats,
                    )
                print(
                    f"RUN  {name} {candidate_display}: warm-up + "
                    f"{measured_repeats} measured" + suffix,
                    flush=True,
                )
                candidate_result = run_case(
                    name,
                    case,
                    config=candidate_case_config,
                    backend=args.backend,
                    measured_repeats=measured_repeats,
                    profile_execution=True,
                    output=args.output,
                    variant=candidate_variant,
                    result_label="fast_hard",
                    profile_repeats=profile_repeats,
                )
                report["cases"][name] = {
                    "baseline": baseline_result,
                    "fast_hard": candidate_result,
                    "comparison": compare_variants(baseline_result, candidate_result),
                }
                if anchor_result is not None:
                    report["cases"][name].update(
                        {
                            "stage1_anchor": anchor_result,
                            "stage1_vs_baseline": compare_variants(
                                baseline_result, anchor_result
                            ),
                            "stage3_vs_stage1": compare_variants(
                                anchor_result, candidate_result
                            ),
                        }
                    )
                if parity_reference is not None:
                    reference_case = parity_reference.get("cases", {}).get(name)
                    if reference_case is None or "fast_hard" not in reference_case:
                        raise ValueError(
                            f"parity reference is missing polar case {name}"
                        )
                    reference_result_path = Path(reference_case["fast_hard"]["result"])
                    if not reference_result_path.is_absolute():
                        reference_result_path = ROOT / reference_result_path
                    report["cases"][name]["backend_parity"] = (
                        compare_polar_backend_parity(
                            reference_result_path,
                            Path(candidate_result["result"]),
                            case,
                            candidate_case_config,
                        )
                    )
            else:
                report["cases"][name] = baseline_result
            write_report(args.output, report)
        if (
            comparison_requested
            and args.suite == "homogeneous"
            and {"fixed_k3", "k1_class_0", "k1_class_1", "k1_class_2"}.issubset(
                report["cases"]
            )
        ):
            report["homogeneous_fixed_k3_vs_k1"] = compare_homogeneous_fixed_k3_to_k1(
                report["cases"], frozen["cases"]
            )
        report["status"] = "completed"
        if comparison_requested:
            measured_variants = (
                ("baseline", "stage1_anchor", "fast_hard")
                if polar_gpu
                else ("baseline", "fast_hard")
            )
            report["summary"] = {
                "case_count": len(report["cases"]),
                "all_finite_and_normalized": True,
                "all_deterministic_exact": all(
                    value[variant]["deterministic_exact_match"]
                    for value in report["cases"].values()
                    for variant in measured_variants
                ),
                "hard_contract_exact": all(
                    value[variant]["mean_max_responsibility"] == 1.0
                    for value in report["cases"].values()
                    for variant in (
                        ("stage1_anchor", "fast_hard") if polar_gpu else ("fast_hard",)
                    )
                ),
                "performance_acceptance": "pending_review",
                "scientific_review": "pending",
                "all_backend_parity_within_tolerance": (
                    all(
                        value["backend_parity"]["within_tolerance"]
                        for value in report["cases"].values()
                    )
                    if parity_reference is not None
                    else None
                ),
                "next_stage": (
                    "Review the one allowed Stage 1 proposal=2 versus proposal=4 "
                    "ablation before entering Stage 2."
                    if args.proposal_ablation_2_vs_4
                    else (
                        "Review Stage 3 GPU parity, native attribution, resources, and "
                        "throughput before accepting the native path."
                        if polar_gpu
                        else (
                            "Review Stage 2 CPU polar_hard quality and determinism before "
                            "starting any Stage 3 GPU implementation."
                            if args.compare_polar_hard
                            else "Review Stage 1 CPU/CuPy/CUDA A/B before accepting or "
                            "rejecting the config-only candidate."
                        )
                    )
                ),
            }
        else:
            report["summary"] = {
                "case_count": len(report["cases"]),
                "all_finite_and_normalized": True,
                "all_deterministic_exact": all(
                    value["deterministic_exact_match"]
                    for value in report["cases"].values()
                ),
                "performance_acceptance": "baseline_only",
                "scientific_review": "pending",
                "next_stage": (
                    "Stage 1 is blocked until all Stage 0 CPU/CuPy/CUDA reports and "
                    "frozen inputs are reviewed."
                ),
            }
    except Exception as error:
        report.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
    report["completed_at_utc"] = utc_now()
    write_report(args.output, report)
    print(f"Report saved to: {args.output}", flush=True)
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
