#!/usr/bin/env python3
"""Run staged AlignImg CPU/GPU validation and write an incremental JSON report."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time
import traceback
from typing import Any, Callable

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment

import alignimg as ai
from alignimg._adaptive import CELL_DTYPE
from alignimg._fourier import (
    _candidate_angles,
    _fourier_ncc,
    prepare_stack,
    score_weights_for_particle,
    soft_circular_mask,
)
from alignimg._fourier_native import (
    score_fourier_candidates_cpu,
    transform_fourier_cpu,
)
from alignimg._geometry import CENTER_CONVENTION


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def distribution_versions() -> dict[str, str | None]:
    names = (
        "alignimg",
        "alignimg-gpu",
        "numpy",
        "scipy",
        "opencv-python-headless",
        "cupy-cuda12x",
        "cupy-cuda13x",
    )
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def environment_info(backend: str, device: int) -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "executable": sys.executable,
        "alignimg_module_version": ai.__version__,
        "versions": distribution_versions(),
        "alignimg_backends": ai.available_alignment_backends(),
    }
    if backend != "cpu":
        import cupy as cp

        cp.cuda.Device(device).use()
        properties = cp.cuda.runtime.getDeviceProperties(device)
        name = properties.get("name", b"unknown")
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        info["gpu"] = {
            "device": device,
            "name": name,
            "compute_capability": [
                int(properties.get("major", 0)),
                int(properties.get("minor", 0)),
            ],
            "driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }
    return info


def gpu_memory_snapshot() -> dict[str, int] | None:
    try:
        import cupy as cp

        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        return {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "pool_used_bytes": int(pool.used_bytes()),
            "pool_total_bytes": int(pool.total_bytes()),
        }
    except (ImportError, RuntimeError):
        return None


def make_references(count: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.indices((size, size), dtype=np.float32)
    references = np.zeros((count, size, size), dtype=np.float32)
    for reference in range(count):
        for _ in range(5):
            cy = rng.uniform(0.22 * size, 0.78 * size)
            cx = rng.uniform(0.22 * size, 0.78 * size)
            sigma_y = rng.uniform(0.025 * size, 0.08 * size)
            sigma_x = rng.uniform(0.025 * size, 0.08 * size)
            amplitude = rng.uniform(0.35, 1.0)
            references[reference] += amplitude * np.exp(
                -0.5 * (((y - cy) / sigma_y) ** 2 + ((x - cx) / sigma_x) ** 2)
            )
    return references


def slice_poses(poses: ai.PoseSet, selected: slice | np.ndarray) -> ai.PoseSet:
    return ai.PoseSet(
        poses.angle_deg[selected],
        poses.shift_y_px[selected],
        poses.shift_x_px[selected],
        poses.mirror[selected],
    )


def make_dataset(
    particle_count: int,
    size: int,
    reference_count: int,
    snr: float | None,
    seed: int,
    angle_samples: int,
    outlier_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ai.PoseSet]:
    rng = np.random.default_rng(seed)
    references = make_references(reference_count, size, seed + 1)
    classes = np.arange(particle_count, dtype=np.int32) % reference_count
    rng.shuffle(classes)
    angle_step = 360.0 / angle_samples
    angle_index = rng.integers(0, angle_samples, size=particle_count)
    source_poses = ai.PoseSet(
        angle_deg=(angle_index * angle_step - 180.0).astype(np.float32),
        shift_y_px=rng.integers(-3, 4, size=particle_count).astype(np.float32),
        shift_x_px=rng.integers(-3, 4, size=particle_count).astype(np.float32),
        mirror=np.zeros(particle_count, dtype=np.bool_),
    )
    particles = np.empty((particle_count, size, size), dtype=np.float32)
    generation_batch = 512
    for start in range(0, particle_count, generation_batch):
        stop = min(start + generation_batch, particle_count)
        particles[start:stop] = ai.transform_images(
            references[classes[start:stop]],
            slice_poses(source_poses, slice(start, stop)),
        )
    if snr is not None:
        signal_variance = np.var(particles, axis=(1, 2), keepdims=True)
        particles += rng.normal(size=particles.shape).astype(np.float32) * np.sqrt(
            signal_variance / float(snr)
        )
    outlier_count = int(round(particle_count * outlier_fraction))
    if outlier_count:
        indices = rng.choice(particle_count, size=outlier_count, replace=False)
        scale = float(np.std(particles))
        particles[indices] = rng.normal(
            0.0, max(scale, 1e-6), size=particles[indices].shape
        ).astype(np.float32)
    return particles, references, classes, source_poses


def stack_correlations(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    a = values.reshape(len(values), -1).astype(np.float64)
    b = targets.reshape(len(targets), -1).astype(np.float64)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    numerator = np.sum(a * b, axis=1)
    denominator = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    return numerator / np.maximum(denominator, 1e-12)


def reference_correlations(values: np.ndarray, targets: np.ndarray) -> list[float]:
    return stack_correlations(values, targets).astype(float).tolist()


def classification_calibration(
    responsibilities: np.ndarray, truth: np.ndarray, bins: int = 10
) -> dict[str, float]:
    rows = np.arange(len(truth))
    true_probability = np.maximum(responsibilities[rows, truth], 1e-30)
    one_hot = np.zeros_like(responsibilities)
    one_hot[rows, truth] = 1.0
    confidence = np.max(responsibilities, axis=1)
    predicted = np.argmax(responsibilities, axis=1)
    correct = predicted == truth
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        selected = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if index == bins - 1
            else confidence < edges[index + 1]
        )
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(
                float(np.mean(confidence[selected])) - float(np.mean(correct[selected]))
            )
    return {
        "class_nll": float(np.mean(-np.log(true_probability))),
        "class_brier_score": float(
            np.mean(np.sum((responsibilities - one_hot) ** 2, axis=1))
        ),
        "class_ece_10": ece,
    }


def result_metrics(
    result: ai.AlignmentResult,
    particles: np.ndarray,
    references: np.ndarray,
    true_classes: np.ndarray,
    source_poses: ai.PoseSet,
    backend: str,
    seconds: float,
) -> dict[str, Any]:
    selected = np.arange(min(256, len(particles)))
    aligned = ai.transform_images(
        particles[selected], slice_poses(result.poses, selected), backend=backend
    )
    aligned_corr = stack_correlations(aligned, references[true_classes[selected]])
    angle_error = (
        result.poses.angle_deg + source_poses.angle_deg + 180.0
    ) % 360.0 - 180.0
    responsibilities = result.responsibilities
    candidate_class_match = result.candidates.reference_index == true_classes[:, None]
    candidate_angle_error = (
        result.candidates.angle_deg + source_poses.angle_deg[:, None] + 180.0
    ) % 360.0 - 180.0
    angle_step = 360.0 / int(result.metadata["config"]["angle_samples"])
    posterior = np.maximum(result.candidates.posterior, 1e-30)
    entropy = -np.sum(posterior * np.log(posterior), axis=1)
    entropy_scale = np.log(max(2, posterior.shape[1]))
    metrics: dict[str, Any] = {
        "seconds": seconds,
        "particles_per_second": len(particles) / max(seconds, 1e-12),
        "assignment_accuracy": float(
            np.mean(result.reference_assignments == true_classes)
        ),
        "mean_aligned_particle_correlation_sample": float(np.mean(aligned_corr)),
        "median_absolute_angle_error_deg": float(np.median(np.abs(angle_error))),
        "angle_accuracy_within_one_bin": float(
            np.mean(np.abs(angle_error) <= angle_step + 1e-6)
        ),
        "candidate_true_class_recall": float(
            np.mean(np.any(candidate_class_match, axis=1))
        ),
        "candidate_class_angle_recall": float(
            np.mean(
                np.any(
                    candidate_class_match
                    & (np.abs(candidate_angle_error) <= angle_step + 1e-6),
                    axis=1,
                )
            )
        ),
        "mean_max_responsibility": float(np.mean(np.max(responsibilities, axis=1))),
        "mean_normalized_candidate_entropy": float(np.mean(entropy / entropy_scale)),
        "responsibility_sum_max_error": float(
            np.max(np.abs(responsibilities.sum(axis=1) - 1.0))
        ),
        "reference_correlations": reference_correlations(result.references, references),
        "mean_reference_correlation": float(
            np.mean(reference_correlations(result.references, references))
        ),
        "effective_component_weight": result.diagnostics[-1][
            "effective_component_weight"
        ],
        "metadata": result.metadata,
        "gpu_memory": gpu_memory_snapshot() if backend != "cpu" else None,
    }
    metrics.update(classification_calibration(responsibilities, true_classes))
    return metrics


def proposal_recall_case() -> dict[str, Any]:
    angle_samples = 72
    proposal_count = 4
    output: dict[str, Any] = {
        "parameters": {
            "particle_count": 64,
            "size": 64,
            "reference_count": 8,
            "angle_samples": angle_samples,
            "proposal_count": proposal_count,
        },
        "metrics": {},
    }
    config = ai.AlignmentConfig(angle_samples=angle_samples)
    for snr in (0.5, 0.2):
        particles, references, classes, source_poses = make_dataset(
            64, 64, 8, snr, 123, angle_samples
        )
        prepared_particles = prepare_stack(particles, config)
        prepared_references = prepare_stack(references, config)
        minimum_errors = []
        for particle in range(len(particles)):
            proposals = _candidate_angles(
                prepared_particles.polar[particle],
                prepared_references.polar[classes[particle]],
                proposal_count,
            )
            minimum_errors.append(
                min(
                    abs(
                        (angle + source_poses.angle_deg[particle] + 180.0) % 360.0
                        - 180.0
                    )
                    for angle in proposals
                )
            )
        errors = np.asarray(minimum_errors)
        output["metrics"][f"snr_{str(snr).replace('.', '_')}"] = {
            "recall_within_one_bin": float(
                np.mean(errors <= 360.0 / angle_samples + 1e-6)
            ),
            "median_minimum_angle_error_deg": float(np.median(errors)),
            "p90_minimum_angle_error_deg": float(np.quantile(errors, 0.9)),
        }
    return output


def global_case(
    *,
    backend: str,
    particle_count: int,
    size: int,
    reference_count: int,
    snr: float,
    iterations: int,
    angle_samples: int,
    top_l: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    particles, references, classes, source_poses = make_dataset(
        particle_count, size, reference_count, snr, seed, angle_samples
    )
    config = ai.AlignmentConfig(
        max_iterations=iterations,
        top_l=top_l,
        angle_samples=angle_samples,
        translation_range=4,
        halfset_diagnostics=False,
        center_references=False,
        batch_size=batch_size,
        random_seed=seed,
    )
    started = time.perf_counter()
    result = ai.align_to_references(
        particles, references, config=config, backend=backend
    )
    seconds = time.perf_counter() - started
    return {
        "parameters": {
            "particle_count": particle_count,
            "size": size,
            "reference_count": reference_count,
            "snr": snr,
            "backend": backend,
            "config": asdict(config),
        },
        "metrics": result_metrics(
            result,
            particles,
            references,
            classes,
            source_poses,
            backend,
            seconds,
        ),
    }


def compact_alignment_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    names = (
        "seconds",
        "particles_per_second",
        "assignment_accuracy",
        "median_absolute_angle_error_deg",
        "angle_accuracy_within_one_bin",
        "candidate_true_class_recall",
        "candidate_class_angle_recall",
        "mean_aligned_particle_correlation_sample",
        "mean_reference_correlation",
        "mean_max_responsibility",
        "mean_normalized_candidate_entropy",
        "class_nll",
        "class_brier_score",
        "class_ece_10",
    )
    return {name: metrics[name] for name in names}


def low_snr_ablation_case(backend: str, batch_size: int) -> dict[str, Any]:
    particle_count = 128
    size = 96
    reference_count = 8
    angle_samples = 96
    particles, references, classes, source_poses = make_dataset(
        particle_count, size, reference_count, 0.2, 50, angle_samples
    )
    experiments = (
        ("p4_l8_t002", 4, 8, 0.02),
        ("p6_l8_t002", 6, 8, 0.02),
        ("p8_l8_t002", 8, 8, 0.02),
        ("p6_l16_t002", 6, 16, 0.02),
        ("p6_l8_t005", 6, 8, 0.05),
        ("p6_l8_t010", 6, 8, 0.10),
        ("p8_l8_t005", 8, 8, 0.05),
    )
    results: dict[str, Any] = {}
    for name, proposal_count, top_l, temperature in experiments:
        config = ai.AlignmentConfig(
            max_iterations=1,
            top_l=top_l,
            angle_samples=angle_samples,
            proposal_angles_per_reference=proposal_count,
            translation_range=4,
            temperature_start=temperature,
            temperature_end=temperature,
            halfset_diagnostics=False,
            center_references=False,
            batch_size=batch_size,
            random_seed=50,
        )
        started = time.perf_counter()
        result = ai.align_to_references(
            particles, references, config=config, backend=backend
        )
        seconds = time.perf_counter() - started
        metrics = result_metrics(
            result,
            particles,
            references,
            classes,
            source_poses,
            backend,
            seconds,
        )
        results[name] = {
            "proposal_angles_per_reference": proposal_count,
            "top_l": top_l,
            "temperature": temperature,
            "metrics": compact_alignment_metrics(metrics),
            "gpu_memory_plans": result.metadata.get("gpu_memory_plans"),
        }
    return {
        "parameters": {
            "backend": backend,
            "particle_count": particle_count,
            "size": size,
            "reference_count": reference_count,
            "snr": 0.2,
            "angle_samples": angle_samples,
        },
        "experiments": results,
    }


def rf_ablation_case(backend: str, batch_size: int) -> dict[str, Any]:
    particle_count = 192
    reference_count = 8
    particles, _, classes, _ = make_dataset(
        particle_count, 96, reference_count, 0.5, 51, 96
    )
    experiments = (
        ("p4_l4_t002", 4, 4, 0.02),
        ("p6_l8_t002", 6, 8, 0.02),
        ("p6_l8_t005", 6, 8, 0.05),
        ("p8_l16_t005", 8, 16, 0.05),
    )
    results: dict[str, Any] = {}
    for name, proposal_count, top_l, temperature in experiments:
        config = ai.AlignmentConfig(
            max_iterations=3,
            top_l=top_l,
            angle_samples=96,
            proposal_angles_per_reference=proposal_count,
            translation_range=4,
            temperature_start=temperature,
            temperature_end=temperature,
            halfset_diagnostics=True,
            center_references=False,
            batch_size=batch_size,
            random_seed=51,
        )
        started = time.perf_counter()
        result = ai.reference_free_align(
            particles, n_components=reference_count, config=config, backend=backend
        )
        seconds = time.perf_counter() - started
        weights = result.diagnostics[-1]["effective_component_weight"]
        frc = result.diagnostics[-1]["frc_0143_cutoff_cyc_per_px"]
        results[name] = {
            "proposal_angles_per_reference": proposal_count,
            "top_l": top_l,
            "temperature": temperature,
            "seconds": seconds,
            "particles_per_second": particle_count / max(seconds, 1e-12),
            "permutation_invariant_assignment_accuracy": permutation_accuracy(
                result.reference_assignments, classes, reference_count
            ),
            "bootstrap_permutation_invariant_assignment_accuracy": permutation_accuracy(
                result.metadata["bootstrap_labels"], classes, reference_count
            ),
            "component_weight_cv": float(
                np.std(weights) / max(float(np.mean(weights)), 1e-12)
            ),
            "minimum_frc_0143_cutoff_cyc_per_px": float(np.min(frc)),
            "median_frc_0143_cutoff_cyc_per_px": float(np.median(frc)),
            "mean_max_responsibility": float(
                np.mean(np.max(result.responsibilities, axis=1))
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
            "reseeded_component_count_trajectory": [
                len(item["reseeded_components"]) for item in result.diagnostics
            ],
            "gpu_memory_plans": result.metadata.get("gpu_memory_plans"),
        }
    return {
        "parameters": {
            "backend": backend,
            "particle_count": particle_count,
            "size": 96,
            "reference_count": reference_count,
            "snr": 0.5,
        },
        "experiments": results,
    }


def batch_sweep_case(backend: str) -> dict[str, Any]:
    if backend == "cpu":
        return {"skipped": True, "reason": "GPU batch sweep is not applicable to CPU."}
    particle_count = 256
    reference_count = 50
    particles, references, classes, source_poses = make_dataset(
        particle_count, 128, reference_count, 0.5, 52, 128
    )
    results: dict[str, Any] = {}
    for batch_size in (128, 256, 512, 1024):
        config = ai.AlignmentConfig(
            max_iterations=1,
            top_l=8,
            angle_samples=128,
            proposal_angles_per_reference=4,
            translation_range=4,
            halfset_diagnostics=False,
            center_references=False,
            batch_size=batch_size,
            random_seed=52,
        )
        started = time.perf_counter()
        result = ai.align_to_references(
            particles, references, config=config, backend=backend
        )
        seconds = time.perf_counter() - started
        metrics = result_metrics(
            result,
            particles,
            references,
            classes,
            source_poses,
            backend,
            seconds,
        )
        results[str(batch_size)] = {
            "seconds": seconds,
            "particles_per_second": metrics["particles_per_second"],
            "assignment_accuracy": metrics["assignment_accuracy"],
            "gpu_memory_plans": result.metadata.get("gpu_memory_plans"),
            "gpu_memory": metrics["gpu_memory"],
        }
    return {
        "parameters": {
            "backend": backend,
            "particle_count": particle_count,
            "size": 128,
            "reference_count": reference_count,
            "snr": 0.5,
        },
        "experiments": results,
    }


def parity_case(backend: str, batch_size: int) -> dict[str, Any]:
    particles, references, classes, source_poses = make_dataset(12, 32, 1, None, 10, 72)
    config = ai.AlignmentConfig(
        max_iterations=1,
        top_l=4,
        angle_samples=72,
        translation_range=4,
        halfset_diagnostics=False,
        center_references=False,
        batch_size=batch_size,
    )
    started = time.perf_counter()
    cpu = ai.align_to_references(particles, references, config=config, backend="cpu")
    cpu_seconds = time.perf_counter() - started
    output: dict[str, Any] = {
        "parameters": {"backend": backend, "config": asdict(config)},
        "cpu": result_metrics(
            cpu,
            particles,
            references,
            classes,
            source_poses,
            "cpu",
            cpu_seconds,
        ),
    }
    if backend != "cpu":
        started = time.perf_counter()
        gpu = ai.align_to_references(
            particles, references, config=config, backend=backend
        )
        gpu_seconds = time.perf_counter() - started
        angle_delta = (
            cpu.poses.angle_deg - gpu.poses.angle_deg + 180.0
        ) % 360.0 - 180.0
        output["gpu"] = result_metrics(
            gpu,
            particles,
            references,
            classes,
            source_poses,
            backend,
            gpu_seconds,
        )
        output["parity"] = {
            "assignment_match_fraction": float(
                np.mean(cpu.reference_assignments == gpu.reference_assignments)
            ),
            "max_absolute_angle_delta_deg": float(np.max(np.abs(angle_delta))),
            "max_absolute_shift_y_delta_px": float(
                np.max(np.abs(cpu.poses.shift_y_px - gpu.poses.shift_y_px))
            ),
            "max_absolute_shift_x_delta_px": float(
                np.max(np.abs(cpu.poses.shift_x_px - gpu.poses.shift_x_px))
            ),
            "responsibility_mae": float(
                np.mean(np.abs(cpu.responsibilities - gpu.responsibilities))
            ),
            "reference_correlations": reference_correlations(
                cpu.references, gpu.references
            ),
        }
    return output


def integer_center_contract_case(backend: str) -> dict[str, Any]:
    size = 32
    center = size // 2
    images = np.zeros((4, size, size), dtype=np.float32)
    images[0, center, center] = 2.0
    images[1, center, center + 3] = 1.0
    images[2, 8, 6] = 1.0
    images[3] = make_references(1, size, 91)[0]
    poses = ai.PoseSet(
        angle_deg=np.asarray([37.0, 90.0, 0.0, 31.0], dtype=np.float32),
        shift_y_px=np.asarray([0.0, 0.0, 0.0, -1.25], dtype=np.float32),
        shift_x_px=np.asarray([0.0, 0.0, 0.0, 2.5], dtype=np.float32),
        mirror=np.asarray([False, False, True, True]),
    )
    expected_peaks = [(center, center), (center - 3, center), (8, size - 6)]
    cpu = ai.transform_images(images, poses, backend="cpu")
    cpu_peaks = [
        tuple(
            int(value)
            for value in np.unravel_index(np.argmax(cpu[index]), cpu[index].shape)
        )
        for index in range(3)
    ]
    shifted_dc = tuple(
        int(value)
        for value in np.unravel_index(
            np.argmax(np.abs(np.fft.fftshift(np.fft.fft2(np.ones((size, size)))))),
            (size, size),
        )
    )
    if cpu_peaks != expected_peaks:
        raise AssertionError(
            f"CPU integer-center peaks {cpu_peaks} do not match {expected_peaks}"
        )
    if shifted_dc != (center, center):
        raise AssertionError(
            f"shifted DFT origin {shifted_dc} does not match {(center, center)}"
        )

    output: dict[str, Any] = {
        "parameters": {
            "backend": backend,
            "size": size,
            "center": center,
            "center_convention": CENTER_CONVENTION,
        },
        "cpu_peaks": cpu_peaks,
        "expected_peaks": expected_peaks,
        "shifted_dft_origin": shifted_dc,
    }
    if backend != "cpu":
        accelerated = ai.transform_images(images, poses, backend=backend)
        accelerated_peaks = [
            tuple(
                int(value)
                for value in np.unravel_index(
                    np.argmax(accelerated[index]), accelerated[index].shape
                )
            )
            for index in range(3)
        ]
        correlations = reference_correlations(accelerated, cpu)
        if accelerated_peaks != expected_peaks:
            raise AssertionError(
                f"{backend} integer-center peaks {accelerated_peaks} do not match "
                f"{expected_peaks}"
            )
        if min(correlations) < 0.999:
            raise AssertionError(
                f"{backend}/CPU transform correlation fell below 0.999: {correlations}"
            )
        output["accelerated_peaks"] = accelerated_peaks
        output["cpu_accelerated_reference_correlations"] = correlations
        output["cpu_accelerated_mean_absolute_error"] = float(
            np.mean(np.abs(accelerated - cpu))
        )
        output["cpu_accelerated_max_absolute_error"] = float(
            np.max(np.abs(accelerated - cpu))
        )
    return output


def permutation_accuracy(predicted: np.ndarray, truth: np.ndarray, count: int) -> float:
    confusion = np.zeros((count, count), dtype=np.int64)
    np.add.at(confusion, (predicted, truth), 1)
    rows, columns = linear_sum_assignment(-confusion)
    return float(confusion[rows, columns].sum() / len(truth))


def reference_free_case(
    backend: str,
    particle_count: int,
    size: int,
    reference_count: int,
    snr: float,
    iterations: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    angle_samples = 64
    particles, _, classes, _ = make_dataset(
        particle_count, size, reference_count, snr, seed, angle_samples
    )
    config = ai.AlignmentConfig(
        max_iterations=iterations,
        top_l=4,
        angle_samples=angle_samples,
        translation_range=4,
        halfset_diagnostics=True,
        center_references=True,
        batch_size=batch_size,
        random_seed=seed,
    )
    started = time.perf_counter()
    result = ai.reference_free_align(
        particles,
        n_components=reference_count,
        config=config,
        backend=backend,
    )
    seconds = time.perf_counter() - started
    return {
        "parameters": {
            "particle_count": particle_count,
            "size": size,
            "reference_count": reference_count,
            "snr": snr,
            "backend": backend,
            "config": asdict(config),
        },
        "metrics": {
            "seconds": seconds,
            "particles_per_second": particle_count / max(seconds, 1e-12),
            "permutation_invariant_assignment_accuracy": permutation_accuracy(
                result.reference_assignments, classes, reference_count
            ),
            "bootstrap_permutation_invariant_assignment_accuracy": permutation_accuracy(
                result.metadata["bootstrap_labels"], classes, reference_count
            ),
            "mean_max_responsibility": float(
                np.mean(np.max(result.responsibilities, axis=1))
            ),
            "responsibility_sum_max_error": float(
                np.max(np.abs(result.responsibilities.sum(axis=1) - 1.0))
            ),
            "effective_component_weight": result.diagnostics[-1][
                "effective_component_weight"
            ],
            "frc_0143_cutoff_cyc_per_px": result.diagnostics[-1][
                "frc_0143_cutoff_cyc_per_px"
            ],
            "bootstrap_labels": result.metadata["bootstrap_labels"],
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
            "reseeded_component_count_trajectory": [
                len(item["reseeded_components"]) for item in result.diagnostics
            ],
            "metadata": result.metadata,
            "gpu_memory": gpu_memory_snapshot() if backend != "cpu" else None,
        },
    }


def refine_case(
    backend: str,
    particle_count: int,
    size: int,
    reference_count: int,
    snr: float,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    particles, references, classes, source_poses = make_dataset(
        particle_count,
        size,
        reference_count,
        snr,
        seed,
        72,
        outlier_fraction=0.10,
    )
    global_config = ai.AlignmentConfig(
        max_iterations=1,
        top_l=4,
        angle_samples=72,
        translation_range=4,
        halfset_diagnostics=False,
        center_references=False,
        batch_size=batch_size,
        random_seed=seed,
    )
    initial = ai.align_to_references(
        particles, references, config=global_config, backend=backend
    )
    refine_config = ai.AlignmentConfig(
        max_iterations=2,
        top_l=4,
        angle_samples=144,
        translation_range=2,
        robust_weighting=True,
        halfset_diagnostics=False,
        center_references=False,
        batch_size=batch_size,
        random_seed=seed,
    )
    started = time.perf_counter()
    refined = ai.refine_alignment(
        particles,
        references,
        initial.poses,
        config=refine_config,
        backend=backend,
    )
    seconds = time.perf_counter() - started
    metrics = result_metrics(
        refined,
        particles,
        references,
        classes,
        source_poses,
        backend,
        seconds,
    )
    metrics["initial_assignment_accuracy"] = float(
        np.mean(initial.reference_assignments == classes)
    )
    metrics["inlier_weight_quantiles"] = np.quantile(
        refined.inlier_weights, [0.0, 0.1, 0.5, 0.9, 1.0]
    )
    return {
        "parameters": {
            "particle_count": particle_count,
            "size": size,
            "reference_count": reference_count,
            "snr": snr,
            "outlier_fraction": 0.10,
            "backend": backend,
            "global_config": asdict(global_config),
            "refine_config": asdict(refine_config),
        },
        "metrics": metrics,
    }


def feedback_refine_case(backend: str, batch_size: int) -> dict[str, Any]:
    references = make_references(2, 32, 71)
    truth = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int32)
    images = references[truth].copy()
    feedback = truth.copy()
    feedback[0] = 1
    initial_poses = ai.PoseSet.identity(len(images))
    config = ai.AlignmentConfig(
        max_iterations=1,
        top_l=4,
        angle_samples=36,
        proposal_angles_per_reference=4,
        translation_range=2,
        temperature_start=0.02,
        temperature_end=0.02,
        halfset_diagnostics=True,
        center_references=False,
        batch_size=batch_size,
    )
    fixed = ai.refine_alignment(
        images,
        references,
        initial_poses,
        class_priors=ai.make_class_priors(
            assignments=feedback, n_components=2, trust=1.0
        ),
        config=config,
        backend=backend,
    )
    if not np.array_equal(fixed.reference_assignments, feedback):
        raise AssertionError("fixed feedback refinement changed an assignment")
    corrective_priors = ai.make_class_priors(
        assignments=feedback, n_components=2, trust=0.5
    )
    corrective = ai.refine_alignment(
        images,
        references,
        initial_poses,
        class_priors=corrective_priors,
        config=config,
        backend=backend,
    )
    output: dict[str, Any] = {
        "parameters": {"backend": backend, "config": asdict(config)},
        "fixed_reassignment_fraction": float(
            np.mean(fixed.reference_assignments != feedback)
        ),
        "corrective_reassignment_fraction": float(
            np.mean(corrective.reference_assignments != feedback)
        ),
        "corrective_truth_accuracy": float(
            np.mean(corrective.reference_assignments == truth)
        ),
        "responsibility_sum_max_error": float(
            np.max(np.abs(corrective.responsibilities.sum(axis=1) - 1.0))
        ),
        "stable_frc_cutoff_cyc_per_px": corrective.diagnostics[-1][
            "frc_0143_stable_cutoff_cyc_per_px"
        ],
    }
    if backend != "cpu":
        cpu = ai.refine_alignment(
            images,
            references,
            initial_poses,
            class_priors=corrective_priors,
            config=config,
            backend="cpu",
        )
        output["cpu_backend_parity"] = {
            "assignment_match_fraction": float(
                np.mean(cpu.reference_assignments == corrective.reference_assignments)
            ),
            "responsibility_mae": float(
                np.mean(np.abs(cpu.responsibilities - corrective.responsibilities))
            ),
            "reference_correlations": reference_correlations(
                cpu.references, corrective.references
            ),
        }
    return output


def final_raw_average_case(backend: str, batch_size: int) -> dict[str, Any]:
    references = make_references(2, 32, 73)
    assignments = np.asarray([0, 0, 1, 1], dtype=np.int32)
    source_poses = ai.PoseSet(
        angle_deg=np.asarray([12.0, -18.0, 24.0, -30.0], dtype=np.float32),
        shift_y_px=np.asarray([1.0, -1.0, 0.0, 1.0], dtype=np.float32),
        shift_x_px=np.asarray([-1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        mirror=np.zeros(4, dtype=np.bool_),
    )
    images = ai.transform_images(references[assignments], source_poses)
    config = ai.AlignmentConfig(
        max_iterations=1,
        top_l=4,
        angle_samples=36,
        proposal_angles_per_reference=4,
        translation_range=2.0,
        temperature_start=0.02,
        temperature_end=0.02,
        robust_weighting=True,
        halfset_diagnostics=False,
        center_references=False,
        batch_size=batch_size,
        apply_final_pose_to_raw=True,
        profile_execution=True,
    )
    result = ai.align_to_references(
        images,
        references,
        class_priors=ai.make_class_priors(
            assignments=assignments,
            n_components=2,
            trust=1.0,
        ),
        config=config,
        backend=backend,
    )
    aligned = ai.transform_images(images, result.poses, backend=backend)
    expected = np.zeros_like(result.class_averages)
    for reference_index in range(2):
        selected = result.reference_assignments == reference_index
        weights = result.inlier_weights[selected].astype(np.float64)
        expected[reference_index] = (
            np.tensordot(
                weights,
                aligned[selected].astype(np.float64),
                axes=(0, 0),
            )
            / weights.sum()
        )
    expected *= soft_circular_mask(32, None, 4.0)[None]
    maximum_error = float(np.max(np.abs(expected - result.class_averages)))
    if maximum_error > 2e-5:
        raise AssertionError(
            f"final raw class average differs from direct reconstruction: {maximum_error}"
        )
    if result.metadata["class_average_estimator"] != (
        "final_map_pose_inlier_weighted_raw"
    ):
        raise AssertionError("final raw class-average estimator was not recorded")
    gpu_metrics = None
    if backend != "cpu":
        performance = result.metadata["performance"]
        counters = performance["stages"]["workflow/final_raw_average"]["counters"]
        expected_avoided = int(images.nbytes)
        expected_output = int(result.class_averages.nbytes)
        if counters.get("final_raw_aligned_d2h_bytes_avoided") != expected_avoided:
            raise AssertionError("final raw aligned-stack D2H accounting is inconsistent")
        if counters.get("final_raw_output_d2h_bytes") != expected_output:
            raise AssertionError("final raw output D2H accounting is inconsistent")
        if result.metadata.get("class_average_gpu_accumulation") != (
            "cupy_fp64_device"
        ):
            raise AssertionError("final raw average was not accumulated on the GPU")
        gpu_metrics = {
            "accumulation": result.metadata["class_average_gpu_accumulation"],
            "memory_plan": result.metadata["class_average_gpu_memory_plan"],
            "memory_events": result.metadata["class_average_gpu_memory_events"],
            "aligned_d2h_bytes_avoided": expected_avoided,
            "output_d2h_bytes": expected_output,
            "performance": performance["stages"]["workflow/final_raw_average"],
        }
    return {
        "parameters": {"backend": backend, "config": asdict(config)},
        "maximum_direct_reconstruction_error": maximum_error,
        "soft_reference_correlations": reference_correlations(
            result.class_averages, result.references
        ),
        "class_average_estimator": result.metadata["class_average_estimator"],
        "class_average_batch_size": result.metadata["class_average_batch_size"],
        "gpu_accumulation": gpu_metrics,
    }


def adaptive_refine_case(backend: str, batch_size: int) -> dict[str, Any]:
    """Exercise the prior-centered coarse/fine path and CPU/GPU parity."""
    references = make_references(1, 32, 79)
    source_angles = np.asarray(
        [-30.0, -24.0, -12.0, -6.0, 6.0, 12.0, 24.0, 30.0],
        dtype=np.float32,
    )
    source_poses = ai.PoseSet(
        source_angles,
        np.zeros(len(source_angles), dtype=np.float32),
        np.zeros(len(source_angles), dtype=np.float32),
        np.zeros(len(source_angles), dtype=np.bool_),
    )
    images = ai.transform_images(
        np.repeat(references, len(source_angles), axis=0), source_poses
    )
    truth_angles = -source_angles
    initial = ai.PoseSet(
        truth_angles + 6.0,
        np.zeros(len(source_angles), dtype=np.float32),
        np.zeros(len(source_angles), dtype=np.float32),
        np.zeros(len(source_angles), dtype=np.bool_),
    )
    config = ai.AlignmentConfig(
        search_strategy="adaptive_posterior",
        max_iterations=1,
        top_l=8,
        coarse_angle_step=6.0,
        coarse_shift_step=1.0,
        local_angle_range=12.0,
        local_shift_range=1.0,
        adaptive_fraction=0.99,
        oversampling_order=1,
        max_adaptive_cells=8,
        temperature_start=0.04,
        temperature_end=0.04,
        pose_angle_sigma=15.0,
        halfset_diagnostics=False,
        center_references=False,
        batch_size=batch_size,
    )

    def run(selected_backend: str) -> tuple[ai.AlignmentResult, float]:
        started = time.perf_counter()
        result = ai.refine_alignment(
            images,
            references,
            initial,
            config=config,
            backend=selected_backend,
        )
        return result, time.perf_counter() - started

    def angle_error(result: ai.AlignmentResult) -> np.ndarray:
        return np.abs((result.poses.angle_deg - truth_angles + 180.0) % 360.0 - 180.0)

    cpu, cpu_seconds = run("cpu")
    initial_error = np.abs((initial.angle_deg - truth_angles + 180.0) % 360.0 - 180.0)
    cpu_error = angle_error(cpu)
    if float(np.median(cpu_error)) >= float(np.median(initial_error)):
        raise AssertionError("adaptive CPU refinement did not improve the initial pose")
    if cpu.diagnostics[0]["mean_retained_fine_mass"] >= 0.99:
        raise AssertionError(
            "adaptive validation did not exercise full-posterior marginalization"
        )
    output: dict[str, Any] = {
        "parameters": {"backend": backend, "config": asdict(config)},
        "cpu": {
            "seconds": cpu_seconds,
            "initial_median_angle_error_deg": float(np.median(initial_error)),
            "final_median_angle_error_deg": float(np.median(cpu_error)),
            "mean_pose_entropy": float(np.mean(cpu.pose_entropy)),
            "mean_map_posterior": float(np.mean(cpu.map_posterior)),
            "diagnostics": cpu.diagnostics,
        },
    }
    if backend != "cpu":
        gpu, gpu_seconds = run(backend)
        gpu_error = angle_error(gpu)
        angle_delta = (
            cpu.poses.angle_deg - gpu.poses.angle_deg + 180.0
        ) % 360.0 - 180.0
        posterior_mae = float(
            np.mean(np.abs(cpu.candidates.posterior - gpu.candidates.posterior))
        )
        correlations = reference_correlations(cpu.references, gpu.references)
        max_angle_delta = float(np.max(np.abs(angle_delta)))
        max_shift_y_delta = float(
            np.max(np.abs(cpu.poses.shift_y_px - gpu.poses.shift_y_px))
        )
        max_shift_x_delta = float(
            np.max(np.abs(cpu.poses.shift_x_px - gpu.poses.shift_x_px))
        )
        if float(np.median(gpu_error)) >= float(np.median(initial_error)):
            raise AssertionError(
                "adaptive GPU refinement did not improve the initial pose"
            )
        if (
            max_angle_delta > 1e-3
            or max_shift_y_delta > 1e-3
            or max_shift_x_delta > 1e-3
            or posterior_mae > 5e-4
            or float(np.min(correlations)) < 0.999
        ):
            raise AssertionError(
                "adaptive CPU/GPU full-posterior parity exceeded tolerance"
            )
        output["gpu"] = {
            "seconds": gpu_seconds,
            "final_median_angle_error_deg": float(np.median(gpu_error)),
            "mean_pose_entropy": float(np.mean(gpu.pose_entropy)),
            "mean_map_posterior": float(np.mean(gpu.map_posterior)),
            "gpu_memory_plans": gpu.metadata.get("gpu_memory_plans"),
        }
        output["parity"] = {
            "max_absolute_angle_delta_deg": max_angle_delta,
            "max_absolute_shift_y_delta_px": max_shift_y_delta,
            "max_absolute_shift_x_delta_px": max_shift_x_delta,
            "candidate_posterior_mae": posterior_mae,
            "reference_correlations": correlations,
        }
    return output


def quadratic_refine_case(backend: str, batch_size: int) -> dict[str, Any]:
    """Check continuous pose precision against the frozen adaptive strategy."""
    size = 48
    y, x = np.indices((size, size), dtype=np.float32)
    reference = np.exp(-((y - 15) ** 2 + (x - 31) ** 2) / 8.0)
    reference += 0.7 * np.exp(-((y - 34) ** 2 + (x - 17) ** 2) / 5.6)
    reference = reference.astype(np.float32)
    source_angle = np.asarray(
        [-5.4, -3.2, -1.7, 0.9, 2.3, 4.6, 6.1, -6.7], dtype=np.float32
    )
    source_y = np.asarray(
        [0.8, -1.2, 0.4, 1.1, -0.6, 0.3, -1.0, 0.7], dtype=np.float32
    )
    source_x = np.asarray(
        [-0.7, 0.5, 1.2, -0.3, 0.8, -1.1, 0.2, -0.9], dtype=np.float32
    )
    source = ai.PoseSet(
        source_angle,
        source_y,
        source_x,
        np.zeros(len(source_angle), dtype=np.bool_),
    )
    images = ai.transform_images(
        np.repeat(reference[None], len(source_angle), axis=0), source
    )
    inverse_radians = np.deg2rad(-source_angle.astype(np.float64))
    rotated_x = np.cos(inverse_radians) * source_x - np.sin(
        inverse_radians
    ) * source_y
    rotated_y = np.sin(inverse_radians) * source_x + np.cos(
        inverse_radians
    ) * source_y
    truth = ai.PoseSet(
        -source_angle,
        -rotated_y,
        -rotated_x,
        np.zeros(len(source_angle), dtype=np.bool_),
    )
    initial = ai.PoseSet(
        truth.angle_deg
        + np.asarray([1.2, -0.9, 1.5, -1.3, 0.8, -1.1, 1.4, -0.7]),
        truth.shift_y_px + 0.25,
        truth.shift_x_px - 0.2,
        truth.mirror,
    )
    common = dict(
        max_iterations=1,
        top_l=8,
        candidate_scoring="fourier",
        reference_update="fourier",
        temperature_start=0.02,
        temperature_end=0.02,
        max_frequency=0.25,
        halfset_diagnostics=False,
        center_references=False,
        robust_weighting=False,
        batch_size=batch_size,
    )
    adaptive_config = ai.AlignmentConfig(
        search_strategy="adaptive_posterior",
        local_angle_range=15.0,
        coarse_angle_step=6.0,
        local_shift_range=3.0,
        coarse_shift_step=1.0,
        adaptive_fraction=0.999,
        oversampling_order=1,
        **common,
    )
    quadratic_config = ai.AlignmentConfig(
        search_strategy="quadratic_refine",
        local_angle_range=7.0,
        coarse_angle_step=1.0,
        local_shift_range=3.0,
        **common,
    )

    def run(config: ai.AlignmentConfig, selected_backend: str):
        started = time.perf_counter()
        result = ai.refine_alignment(
            images,
            reference,
            initial,
            config=config,
            backend=selected_backend,
        )
        return result, time.perf_counter() - started

    def errors(result: ai.AlignmentResult) -> tuple[np.ndarray, np.ndarray]:
        angle = np.abs(
            (result.poses.angle_deg - truth.angle_deg + 180.0) % 360.0 - 180.0
        )
        shift = np.hypot(
            result.poses.shift_y_px - truth.shift_y_px,
            result.poses.shift_x_px - truth.shift_x_px,
        )
        return angle, shift

    adaptive, adaptive_seconds = run(adaptive_config, backend)
    quadratic, quadratic_seconds = run(quadratic_config, backend)
    quadratic_angle, quadratic_shift = errors(quadratic)
    adaptive_angle, adaptive_shift = errors(adaptive)
    medians = {
        "adaptive_angle_error_deg": float(np.median(adaptive_angle)),
        "quadratic_angle_error_deg": float(np.median(quadratic_angle)),
        "adaptive_shift_error_px": float(np.median(adaptive_shift)),
        "quadratic_shift_error_px": float(np.median(quadratic_shift)),
    }
    if medians["quadratic_angle_error_deg"] > 0.5:
        raise AssertionError("quadratic noiseless median angle error exceeds 0.5 deg")
    if medians["quadratic_shift_error_px"] > 0.2:
        raise AssertionError("quadratic noiseless median shift error exceeds 0.2 px")
    if medians["quadratic_angle_error_deg"] > 0.75 * medians[
        "adaptive_angle_error_deg"
    ]:
        raise AssertionError("quadratic angle error did not improve by at least 25%")
    if medians["quadratic_shift_error_px"] > 0.75 * medians[
        "adaptive_shift_error_px"
    ]:
        raise AssertionError("quadratic shift error did not improve by at least 25%")
    if quadratic.metadata["backend"] != backend:
        raise AssertionError("quadratic backend fallback is not allowed")
    if (
        backend == "cuda"
        and quadratic.metadata.get("quadratic_peak_backend") != "native_cuda"
    ):
        raise AssertionError(
            "quadratic CUDA validation did not use the native peak kernel"
        )
    if (
        backend == "cupy"
        and quadratic.metadata.get("quadratic_peak_backend") != "cupy"
    ):
        raise AssertionError("quadratic CuPy validation did not use the CuPy peak solver")

    noise_scenarios: dict[str, Any] = {}
    for snr, seed in ((0.5, 901), (0.2, 902)):
        rng = np.random.default_rng(seed)
        noisy = images.copy()
        signal_variance = np.var(noisy, axis=(1, 2), keepdims=True)
        noisy += rng.normal(size=noisy.shape).astype(np.float32) * np.sqrt(
            signal_variance / snr
        )

        def run_noisy(config: ai.AlignmentConfig):
            return ai.refine_alignment(
                noisy,
                reference,
                initial,
                config=config,
                backend=backend,
            )

        adaptive_noisy = run_noisy(adaptive_config)
        quadratic_noisy = run_noisy(quadratic_config)
        adaptive_noisy_angle, adaptive_noisy_shift = errors(adaptive_noisy)
        quadratic_noisy_angle, quadratic_noisy_shift = errors(quadratic_noisy)
        scenario = {
            "adaptive_median_angle_error_deg": float(
                np.median(adaptive_noisy_angle)
            ),
            "quadratic_median_angle_error_deg": float(
                np.median(quadratic_noisy_angle)
            ),
            "adaptive_median_shift_error_px": float(
                np.median(adaptive_noisy_shift)
            ),
            "quadratic_median_shift_error_px": float(
                np.median(quadratic_noisy_shift)
            ),
            "quadratic_p95_angle_error_deg": float(
                np.quantile(quadratic_noisy_angle, 0.95)
            ),
            "quadratic_p95_shift_error_px": float(
                np.quantile(quadratic_noisy_shift, 0.95)
            ),
        }
        if scenario["quadratic_median_angle_error_deg"] >= scenario[
            "adaptive_median_angle_error_deg"
        ]:
            raise AssertionError(f"quadratic angle error did not improve at SNR {snr}")
        if scenario["quadratic_median_shift_error_px"] >= scenario[
            "adaptive_median_shift_error_px"
        ]:
            raise AssertionError(f"quadratic shift error did not improve at SNR {snr}")
        noise_scenarios[f"snr_{str(snr).replace('.', '_')}"] = scenario

    output: dict[str, Any] = {
        "parameters": {
            "backend": backend,
            "adaptive_config": asdict(adaptive_config),
            "quadratic_config": asdict(quadratic_config),
        },
        "adaptive_seconds": adaptive_seconds,
        "quadratic_seconds": quadratic_seconds,
        "median_errors": medians,
        "quadratic_p95_angle_error_deg": float(np.quantile(quadratic_angle, 0.95)),
        "quadratic_p95_shift_error_px": float(np.quantile(quadratic_shift, 0.95)),
        "quadratic_diagnostics": quadratic.diagnostics,
        "quadratic_metadata": quadratic.metadata,
        "noise_scenarios": noise_scenarios,
    }
    if backend != "cpu":
        cpu, cpu_seconds = run(quadratic_config, "cpu")
        angle_delta = np.abs(
            (quadratic.poses.angle_deg - cpu.poses.angle_deg + 180.0) % 360.0
            - 180.0
        )
        shift_delta = np.maximum(
            np.abs(quadratic.poses.shift_y_px - cpu.poses.shift_y_px),
            np.abs(quadratic.poses.shift_x_px - cpu.poses.shift_x_px),
        )
        maximum_angle_delta = float(np.max(angle_delta))
        maximum_shift_delta = float(np.max(shift_delta))
        if maximum_angle_delta > 1e-3 or maximum_shift_delta > 1e-3:
            raise AssertionError("quadratic backend pose diverged from CPU authority")
        output["cpu_backend"] = {
            "seconds": cpu_seconds,
            "maximum_angle_delta_deg": maximum_angle_delta,
            "maximum_shift_delta_px": maximum_shift_delta,
            "maximum_posterior_delta": float(
                np.max(
                    np.abs(
                        quadratic.candidates.posterior
                        - cpu.candidates.posterior
                    )
                )
            ),
            "reference_correlations": reference_correlations(
                quadratic.references, cpu.references
            ),
        }
    return output


def adaptive_rescue_case(backend: str, batch_size: int) -> dict[str, Any]:
    """Exercise uncertainty-triggered proposal rescue without a boundary hit."""
    references = make_references(1, 32, 80)
    source_angle = 60.0
    source_poses = ai.PoseSet(
        np.full(10, source_angle, dtype=np.float32),
        np.zeros(10, dtype=np.float32),
        np.zeros(10, dtype=np.float32),
        np.zeros(10, dtype=np.bool_),
    )
    images = ai.transform_images(np.repeat(references, 10, axis=0), source_poses)
    initial_angles = np.full(10, -source_angle, dtype=np.float32)
    initial_angles[0] = 0.0
    initial = ai.PoseSet(
        initial_angles,
        np.zeros(10, dtype=np.float32),
        np.zeros(10, dtype=np.float32),
        np.zeros(10, dtype=np.bool_),
    )
    config = ai.AlignmentConfig(
        search_strategy="adaptive_posterior",
        max_iterations=3,
        top_l=8,
        angle_samples=72,
        proposal_angles_per_reference=8,
        translation_range=2.0,
        coarse_angle_step=6.0,
        coarse_shift_step=1.0,
        local_angle_range=0.0,
        local_shift_range=0.0,
        adaptive_fraction=0.99,
        oversampling_order=1,
        rescue_uncertain_particles=True,
        rescue_normalized_entropy_threshold=0.0,
        rescue_max_fraction=0.1,
        rescue_min_score_improvement=0.0,
        temperature_start=0.04,
        temperature_end=0.04,
        halfset_diagnostics=False,
        center_references=False,
        batch_size=batch_size,
    )
    started = time.perf_counter()
    result = ai.refine_alignment(
        images, references, initial, config=config, backend=backend
    )
    seconds = time.perf_counter() - started
    final_error = float(
        abs((float(result.poses.angle_deg[0]) + source_angle + 180.0) % 360.0 - 180.0)
    )
    if result.diagnostics[0]["boundary_hit_count"] != 0:
        raise AssertionError(
            "uncertainty rescue smoke unexpectedly hit a local boundary"
        )
    if result.diagnostics[0]["rescue_scheduled_count"] != 1:
        raise AssertionError("uncertainty rescue did not respect its per-iteration cap")
    if result.diagnostics[1]["rescue_particle_count"] != 1:
        raise AssertionError("scheduled particle did not use proposal fallback")
    if result.diagnostics[1]["rescue_accepted_count"] != 1:
        raise AssertionError("better proposal fallback was not accepted")
    if result.diagnostics[2]["rescue_particle_count"] != 0:
        raise AssertionError("final iteration was not reserved for local confirmation")
    if final_error > 5.0:
        raise AssertionError("uncertainty rescue did not recover the global pose basin")
    return {
        "parameters": {"backend": backend, "config": asdict(config)},
        "seconds": seconds,
        "initial_bad_particle_angle_error_deg": source_angle,
        "final_bad_particle_angle_error_deg": final_error,
        "diagnostics": result.diagnostics,
        "gpu_memory_plans": result.metadata.get("gpu_memory_plans"),
    }


def warmup_case(backend: str, batch_size: int) -> dict[str, Any]:
    if backend == "cpu":
        return {
            "skipped": True,
            "reason": "CUDA warm-up is not needed for CPU-only validation.",
        }
    result = global_case(
        backend=backend,
        particle_count=4,
        size=32,
        reference_count=1,
        snr=1.0,
        iterations=1,
        angle_samples=16,
        top_l=2,
        batch_size=batch_size,
        seed=1,
    )
    return result


def fourier_native_conformance_case(backend: str) -> dict[str, Any]:
    """Compare the experimental Fourier transform with the frozen raster path."""
    size = 128
    center = size / 2.0
    y, x = np.indices((size, size), dtype=np.float32)
    image = np.exp(-((y - (center - 10)) ** 2 + (x - (center + 8)) ** 2) / 28.0)
    image += 0.65 * np.exp(-((y - (center + 12)) ** 2 + (x - (center - 9)) ** 2) / 16.0)
    image += 0.25 * np.exp(-((y - (center + 2)) ** 2 + (x - (center + 15)) ** 2) / 8.0)
    image = image.astype(np.float32)
    poses = ai.PoseSet(
        angle_deg=np.asarray([0.0, 0.0, 0.0, 23.5, -41.25, 90.0]),
        shift_y_px=np.asarray([0.0, 1.5, 0.0, 1.25, -2.0, 3.0]),
        shift_x_px=np.asarray([0.0, -2.0, 0.0, -0.75, 1.5, -2.0]),
        mirror=np.asarray([False, False, True, False, True, True]),
    )
    input_fourier = np.repeat(np.fft.fft2(image)[None], len(poses), axis=0).astype(
        np.complex64
    )
    cpu_started = time.perf_counter()
    cpu_fourier = np.stack(
        [
            transform_fourier_cpu(
                input_fourier[index],
                angle_deg=poses.angle_deg[index],
                shift_y_px=poses.shift_y_px[index],
                shift_x_px=poses.shift_x_px[index],
                mirror=poses.mirror[index],
            )
            for index in range(len(poses))
        ]
    )
    cpu_seconds = time.perf_counter() - cpu_started
    raster_fourier = np.fft.fft2(
        ai.transform_images(np.repeat(image[None], len(poses), axis=0), poses),
        axes=(-2, -1),
    )
    frequency = np.fft.fftfreq(size)
    fy, fx = np.meshgrid(frequency, frequency, indexing="ij")
    frequency_mask = ((np.hypot(fy, fx) > 0.0) & (np.hypot(fy, fx) <= 0.35)).astype(
        np.float32
    )
    raster_ncc = np.asarray(
        [
            _fourier_ncc(cpu_fourier[index], raster_fourier[index], frequency_mask)
            for index in range(len(poses))
        ]
    )
    if float(np.min(raster_ncc)) < 0.995:
        raise AssertionError(
            "Fourier-native transform does not conform to raster scoring"
        )

    selected_backend = backend
    backend_seconds = cpu_seconds
    backend_fourier = cpu_fourier
    candidate_cells = np.asarray(
        [
            (0, 0.0, 0.0, 0.0, False),
            (0, 23.5, 1.25, -0.75, False),
            (1, -41.25, -2.0, 1.5, True),
            (1, 90.0, 3.0, -2.0, True),
        ],
        dtype=CELL_DTYPE,
    )
    candidate_references = np.stack(
        (input_fourier[0], np.roll(input_fourier[0], 5, axis=0))
    )
    cpu_scores = score_fourier_candidates_cpu(
        input_fourier[0],
        candidate_references,
        candidate_cells,
        frequency_mask,
    )
    backend_scores = cpu_scores
    if backend != "cpu":
        from alignimg_gpu.backend import (
            score_fourier_candidates_cuda,
            score_fourier_candidates_cupy,
            transform_fourier_cuda,
            transform_fourier_cupy,
        )

        if backend in {"auto", "gpu"}:
            selected_backend = (
                "cuda"
                if ai.available_alignment_backends()["cuda"]["available"]
                else "cupy"
            )
        transform = (
            transform_fourier_cuda
            if selected_backend == "cuda"
            else transform_fourier_cupy
        )
        backend_started = time.perf_counter()
        backend_fourier = transform(input_fourier, poses)
        backend_seconds = time.perf_counter() - backend_started
        scorer = (
            score_fourier_candidates_cuda
            if selected_backend == "cuda"
            else score_fourier_candidates_cupy
        )
        backend_scores = scorer(
            input_fourier[0],
            candidate_references,
            candidate_cells,
            frequency_mask,
            ai.AlignmentConfig(batch_size=2),
        )

    delta = np.asarray(backend_fourier - cpu_fourier)
    relative_l2_error = float(
        np.linalg.norm(delta) / max(float(np.linalg.norm(cpu_fourier)), 1e-12)
    )
    maximum_error_relative_to_peak = float(
        np.max(np.abs(delta)) / max(float(np.max(np.abs(cpu_fourier))), 1e-12)
    )
    if relative_l2_error > 3e-5 or maximum_error_relative_to_peak > 3e-5:
        raise AssertionError("GPU Fourier-native transform does not match CPU")
    score_maximum_absolute_error = float(np.max(np.abs(backend_scores - cpu_scores)))
    if score_maximum_absolute_error > 2e-5:
        raise AssertionError("GPU Fourier-native candidate scores do not match CPU")
    return {
        "parameters": {
            "requested_backend": backend,
            "selected_backend": selected_backend,
            "size": size,
            "pose_count": len(poses),
            "maximum_frequency": 0.35,
        },
        "raster_band_ncc": raster_ncc,
        "minimum_raster_band_ncc": float(np.min(raster_ncc)),
        "cpu_seconds": cpu_seconds,
        "backend_seconds": backend_seconds,
        "backend_relative_l2_error": relative_l2_error,
        "backend_maximum_error_relative_to_peak": maximum_error_relative_to_peak,
        "cpu_candidate_scores": cpu_scores,
        "backend_candidate_scores": backend_scores,
        "candidate_score_maximum_absolute_error": score_maximum_absolute_error,
        "gpu_memory": gpu_memory_snapshot() if backend != "cpu" else None,
    }


def native_indexed_fourier_case(backend: str) -> dict[str, Any]:
    """Compare direct cache indexing with the previous gathered native input."""
    if backend != "cuda":
        return {
            "skipped": True,
            "reason": "native indexed Fourier transform is CUDA-specific",
        }
    import cupy as cp
    from alignimg_gpu.backend import (
        _transform_fourier_batch_cuda,
        _transform_fourier_indexed_batch_cuda,
    )

    rng = np.random.default_rng(20260909)
    size = 32
    source = (
        rng.normal(size=(4, size, size))
        + 1j * rng.normal(size=(4, size, size))
    ).astype(np.complex64)
    indices = np.asarray([3, 1, 3, 0, 2, 1], dtype=np.int32)
    angles = np.asarray([0.0, 23.5, -41.25, 90.0, 12.0, -73.0], dtype=np.float32)
    shifts_y = np.asarray([0.0, 1.5, -2.0, 3.0, -0.75, 2.25], dtype=np.float32)
    shifts_x = np.asarray([0.0, -2.0, 1.5, -2.0, 1.25, -0.5], dtype=np.float32)
    mirrors = np.asarray([False, False, True, True, False, True])
    device_source = cp.asarray(source)
    device_indices = cp.asarray(indices)
    gathered = _transform_fourier_batch_cuda(
        device_source[device_indices], angles, shifts_y, shifts_x, mirrors
    )
    indexed = _transform_fourier_indexed_batch_cuda(
        device_source, device_indices, angles, shifts_y, shifts_x, mirrors
    )
    delta = cp.asnumpy(indexed - gathered)
    maximum_absolute_error = float(np.max(np.abs(delta), initial=0.0))
    relative_l2_error = float(
        np.linalg.norm(delta)
        / max(float(np.linalg.norm(cp.asnumpy(gathered))), 1e-12)
    )
    if maximum_absolute_error != 0.0:
        raise AssertionError("indexed native transform changed Fourier values")
    return {
        "size": size,
        "source_count": len(source),
        "candidate_count": len(indices),
        "gather_bytes_avoided": int(len(indices) * size * size * 8),
        "maximum_absolute_error": maximum_absolute_error,
        "relative_l2_error": relative_l2_error,
        "gpu_memory": gpu_memory_snapshot(),
    }


def native_fused_fourier_accumulation_case(backend: str) -> dict[str, Any]:
    """Compare fused native M-step accumulation with the dev7 CuPy scatter."""
    if backend != "cuda":
        return {
            "skipped": True,
            "reason": "native fused Fourier accumulation is CUDA-specific",
        }
    import cupy as cp
    from alignimg_gpu.backend import (
        _accumulate_fourier_indexed_batch_cuda,
        _transform_fourier_indexed_batch_cuda,
    )

    rng = np.random.default_rng(20260911)
    size = 32
    source = (
        rng.normal(size=(4, size, size))
        + 1j * rng.normal(size=(4, size, size))
    ).astype(np.complex64)
    indices = np.asarray([3, 1, 3, 0, 2, 1, 0, 2], dtype=np.int32)
    angles = np.asarray(
        [0.0, 23.5, -41.25, 90.0, 12.0, -73.0, 7.5, 38.0],
        dtype=np.float32,
    )
    shifts_y = np.asarray(
        [0.0, 1.5, -2.0, 3.0, -0.75, 2.25, -1.5, 0.5], dtype=np.float32
    )
    shifts_x = np.asarray(
        [0.0, -2.0, 1.5, -2.0, 1.25, -0.5, 2.0, -1.25], dtype=np.float32
    )
    mirrors = np.asarray([False, False, True, True, False, True, False, True])
    accumulator_ids = np.asarray([0, 1, 2, 0, 1, 2, 0, 2], dtype=np.int32)
    weights = np.asarray(
        [1.0, 0.75, 0.5, 0.25, 0.125, 0.6, 0.4, 0.2], dtype=np.float32
    )
    accumulator_count = 3
    device_source = cp.asarray(source)
    device_indices = cp.asarray(indices)
    transformed = _transform_fourier_indexed_batch_cuda(
        device_source, device_indices, angles, shifts_y, shifts_x, mirrors
    )
    expected_real = cp.zeros((accumulator_count, size, size), dtype=cp.float64)
    expected_imag = cp.zeros_like(expected_real)
    expected_weights = cp.zeros(accumulator_count, dtype=cp.float64)
    device_accumulators = cp.asarray(accumulator_ids)
    device_weights = cp.asarray(weights, dtype=cp.float64)
    cp.add.at(
        expected_real,
        device_accumulators,
        transformed.real.astype(cp.float64) * device_weights[:, None, None],
    )
    cp.add.at(
        expected_imag,
        device_accumulators,
        transformed.imag.astype(cp.float64) * device_weights[:, None, None],
    )
    cp.add.at(expected_weights, device_accumulators, device_weights)

    actual_real = cp.zeros_like(expected_real)
    actual_imag = cp.zeros_like(expected_imag)
    actual_weights = cp.zeros_like(expected_weights)
    _accumulate_fourier_indexed_batch_cuda(
        device_source,
        device_indices,
        angles,
        shifts_y,
        shifts_x,
        mirrors,
        accumulator_ids,
        weights,
        actual_real,
        actual_imag,
        actual_weights,
    )
    cp.cuda.get_current_stream().synchronize()
    real_error = float(cp.max(cp.abs(actual_real - expected_real)).get())
    imag_error = float(cp.max(cp.abs(actual_imag - expected_imag)).get())
    weight_error = float(cp.max(cp.abs(actual_weights - expected_weights)).get())
    if max(real_error, imag_error, weight_error) > 1e-10:
        raise AssertionError("fused native accumulation changed FP64 sums")
    return {
        "size": size,
        "source_count": len(source),
        "candidate_count": len(indices),
        "accumulator_count": accumulator_count,
        "maximum_real_sum_error": real_error,
        "maximum_imaginary_sum_error": imag_error,
        "maximum_weight_error": weight_error,
        "transformed_batch_bytes_avoided": int(len(indices) * size * size * 8),
        "gpu_memory": gpu_memory_snapshot(),
    }


def _top_l_posterior_comparison(
    raster: ai.CandidateSet,
    fourier: ai.CandidateSet,
    *,
    angle_tolerance_deg: float = 3.0,
    shift_tolerance_px: float = 1.0,
) -> dict[str, Any]:
    """Match top-L pose cells before comparing their posterior mass."""
    if raster.posterior.shape != fourier.posterior.shape:
        raise ValueError("raster and Fourier candidate shapes must match")

    matched_fractions = []
    raster_matched_mass = []
    fourier_matched_mass = []
    posterior_total_variation = []
    matched_score_errors = []
    for particle_index in range(raster.posterior.shape[0]):
        raster_active = np.flatnonzero(raster.posterior[particle_index] > 0.0)
        fourier_active = np.flatnonzero(fourier.posterior[particle_index] > 0.0)
        if len(raster_active) == 0 or len(fourier_active) == 0:
            raise ValueError("each particle must retain at least one candidate")

        raster_angle = raster.angle_deg[particle_index, raster_active]
        fourier_angle = fourier.angle_deg[particle_index, fourier_active]
        raster_shift_y = raster.shift_y_px[particle_index, raster_active]
        fourier_shift_y = fourier.shift_y_px[particle_index, fourier_active]
        raster_shift_x = raster.shift_x_px[particle_index, raster_active]
        fourier_shift_x = fourier.shift_x_px[particle_index, fourier_active]
        raster_reference = raster.reference_index[particle_index, raster_active]
        fourier_reference = fourier.reference_index[particle_index, fourier_active]
        raster_mirror = raster.mirror[particle_index, raster_active]
        fourier_mirror = fourier.mirror[particle_index, fourier_active]
        angle_delta = np.abs(
            (raster_angle[:, None] - fourier_angle[None, :] + 180.0) % 360.0
            - 180.0
        )
        shift_y_delta = np.abs(
            raster_shift_y[:, None] - fourier_shift_y[None, :]
        )
        shift_x_delta = np.abs(
            raster_shift_x[:, None] - fourier_shift_x[None, :]
        )
        compatible = (
            (raster_reference[:, None] == fourier_reference[None, :])
            & (raster_mirror[:, None] == fourier_mirror[None, :])
            & (angle_delta <= angle_tolerance_deg)
            & (shift_y_delta <= shift_tolerance_px)
            & (shift_x_delta <= shift_tolerance_px)
        )
        cost = (
            angle_delta / angle_tolerance_deg
            + shift_y_delta / shift_tolerance_px
            + shift_x_delta / shift_tolerance_px
            + (~compatible) * 1.0e6
        )
        raster_rows, fourier_columns = linear_sum_assignment(cost)
        valid = compatible[raster_rows, fourier_columns]
        raster_rows = raster_rows[valid]
        fourier_columns = fourier_columns[valid]

        raster_indices = raster_active[raster_rows]
        fourier_indices = fourier_active[fourier_columns]
        raster_mass = float(
            np.sum(raster.posterior[particle_index, raster_indices])
        )
        fourier_mass = float(
            np.sum(fourier.posterior[particle_index, fourier_indices])
        )
        matched_difference = float(
            np.sum(
                np.abs(
                    raster.posterior[particle_index, raster_indices]
                    - fourier.posterior[particle_index, fourier_indices]
                )
            )
        )
        matched_fractions.append(
            len(raster_indices) / max(len(raster_active), len(fourier_active))
        )
        raster_matched_mass.append(raster_mass)
        fourier_matched_mass.append(fourier_mass)
        posterior_total_variation.append(
            0.5
            * (
                matched_difference
                + (1.0 - raster_mass)
                + (1.0 - fourier_mass)
            )
        )
        matched_score_errors.extend(
            np.abs(
                raster.score[particle_index, raster_indices]
                - fourier.score[particle_index, fourier_indices]
            ).tolist()
        )

    return {
        "pose_match_tolerance": {
            "angle_deg": angle_tolerance_deg,
            "shift_per_axis_px": shift_tolerance_px,
        },
        "minimum_matched_candidate_fraction": float(np.min(matched_fractions)),
        "mean_matched_candidate_fraction": float(np.mean(matched_fractions)),
        "minimum_raster_matched_posterior_mass": float(
            np.min(raster_matched_mass)
        ),
        "minimum_fourier_matched_posterior_mass": float(
            np.min(fourier_matched_mass)
        ),
        "mean_posterior_total_variation": float(
            np.mean(posterior_total_variation)
        ),
        "maximum_posterior_total_variation": float(
            np.max(posterior_total_variation)
        ),
        "matched_score_mae": float(np.mean(matched_score_errors)),
        "matched_score_maximum_absolute_error": float(
            np.max(matched_score_errors)
        ),
    }


def fourier_workflow_ab_case(backend: str, batch_size: int) -> dict[str, Any]:
    """Compare raster and Fourier-native scoring through production workflows."""
    particles, references, classes, _ = make_dataset(
        particle_count=16,
        size=64,
        reference_count=2,
        snr=None,
        seed=180,
        angle_samples=72,
    )
    class_priors = ai.make_class_priors(
        assignments=classes,
        n_components=len(references),
        trust=1.0,
    )

    def run_global(candidate_scoring: str):
        config = ai.AlignmentConfig(
            candidate_scoring=candidate_scoring,
            max_iterations=1,
            top_l=4,
            angle_samples=72,
            proposal_angles_per_reference=4,
            translation_range=4.0,
            halfset_diagnostics=False,
            center_references=False,
            batch_size=batch_size,
        )
        started = time.perf_counter()
        result = ai.align_to_references(
            particles,
            references,
            class_priors=class_priors,
            config=config,
            backend=backend,
        )
        return result, time.perf_counter() - started

    raster_global, raster_global_seconds = run_global("raster")
    fourier_global, fourier_global_seconds = run_global("fourier")

    def run_refine(candidate_scoring: str):
        config = ai.AlignmentConfig(
            candidate_scoring=candidate_scoring,
            search_strategy="adaptive_posterior",
            max_iterations=1,
            top_l=4,
            angle_samples=72,
            coarse_angle_step=3.0,
            coarse_shift_step=1.0,
            local_angle_range=6.0,
            local_shift_range=2.0,
            adaptive_fraction=0.95,
            oversampling_order=1,
            max_adaptive_cells=16,
            halfset_diagnostics=False,
            center_references=False,
            batch_size=batch_size,
        )
        started = time.perf_counter()
        result = ai.refine_alignment(
            particles,
            references,
            raster_global.poses,
            class_priors=class_priors,
            config=config,
            backend=backend,
        )
        return result, time.perf_counter() - started

    raster_refine, raster_refine_seconds = run_refine("raster")
    fourier_refine, fourier_refine_seconds = run_refine("fourier")

    def comparison(
        raster: ai.AlignmentResult,
        fourier: ai.AlignmentResult,
        raster_seconds: float,
        fourier_seconds: float,
    ) -> dict[str, Any]:
        angle_delta = np.abs(
            (fourier.poses.angle_deg - raster.poses.angle_deg + 180.0) % 360.0 - 180.0
        )
        shift_y_delta = np.abs(fourier.poses.shift_y_px - raster.poses.shift_y_px)
        shift_x_delta = np.abs(fourier.poses.shift_x_px - raster.poses.shift_x_px)
        shift_delta = np.hypot(shift_y_delta, shift_x_delta)
        correlations = stack_correlations(fourier.references, raster.references)
        top_l = _top_l_posterior_comparison(raster.candidates, fourier.candidates)
        metrics = {
            "raster_seconds": raster_seconds,
            "fourier_seconds": fourier_seconds,
            "raster_to_fourier_speed_ratio": raster_seconds
            / max(fourier_seconds, 1e-12),
            "assignment_disagreement_fraction": float(
                np.mean(fourier.reference_assignments != raster.reference_assignments)
            ),
            "pose_angle_delta_deg": {
                "maximum": float(np.max(angle_delta)),
                "median": float(np.median(angle_delta)),
            },
            "pose_shift_delta_px": {
                "maximum": float(np.max(shift_delta)),
                "median": float(np.median(shift_delta)),
                "maximum_y": float(np.max(shift_y_delta)),
                "maximum_x": float(np.max(shift_x_delta)),
            },
            "reference_correlation": correlations,
            "minimum_reference_correlation": float(np.min(correlations)),
            "responsibility_mae": float(
                np.mean(np.abs(fourier.responsibilities - raster.responsibilities))
            ),
            "top_l_posterior": top_l,
            "fourier_metadata": fourier.metadata,
        }
        if metrics["assignment_disagreement_fraction"] != 0.0:
            raise AssertionError("fixed-class assignments changed between scorers")
        if metrics["pose_angle_delta_deg"]["maximum"] > 3.0:
            raise AssertionError("Fourier-native scoring changed the selected angle")
        if (
            metrics["pose_shift_delta_px"]["maximum_y"] > 1.0
            or metrics["pose_shift_delta_px"]["maximum_x"] > 1.0
        ):
            raise AssertionError("Fourier-native scoring changed the selected shift")
        if metrics["minimum_reference_correlation"] < 0.99:
            raise AssertionError("Fourier-native scoring changed the updated reference")
        if top_l["minimum_matched_candidate_fraction"] < 1.0:
            raise AssertionError("Fourier-native scoring changed the top-L pose support")
        if top_l["mean_posterior_total_variation"] > 0.15:
            raise AssertionError("Fourier-native scoring changed the top-L posterior")
        if top_l["maximum_posterior_total_variation"] > 0.40:
            raise AssertionError(
                "Fourier-native scoring strongly changed one top-L posterior"
            )
        if top_l["matched_score_maximum_absolute_error"] > 0.05:
            raise AssertionError("Fourier-native candidate scores diverged from raster")
        return metrics

    return {
        "parameters": {
            "backend": backend,
            "particle_count": len(particles),
            "size": particles.shape[1],
            "reference_count": len(references),
            "batch_size": batch_size,
        },
        "global": comparison(
            raster_global,
            fourier_global,
            raster_global_seconds,
            fourier_global_seconds,
        ),
        "adaptive_refine": comparison(
            raster_refine,
            fourier_refine,
            raster_refine_seconds,
            fourier_refine_seconds,
        ),
        "gpu_memory": gpu_memory_snapshot() if backend != "cpu" else None,
    }


def fourier_mstep_workflow_ab_case(backend: str, batch_size: int) -> dict[str, Any]:
    """Compare spatial and Fourier M-steps through production workflows."""
    particles, references, classes, _ = make_dataset(
        particle_count=16,
        size=64,
        reference_count=2,
        snr=None,
        seed=190,
        angle_samples=72,
    )
    class_priors = ai.make_class_priors(
        assignments=classes,
        n_components=len(references),
        trust=1.0,
    )

    def run(reference_update: str, initial_poses=None, *, selected_backend=backend):
        adaptive = initial_poses is not None
        config = ai.AlignmentConfig(
            candidate_scoring="fourier",
            reference_update=reference_update,
            search_strategy="adaptive_posterior" if adaptive else "proposal",
            max_iterations=1,
            top_l=4,
            angle_samples=72,
            proposal_angles_per_reference=4,
            translation_range=4.0,
            coarse_angle_step=3.0,
            coarse_shift_step=1.0,
            local_angle_range=6.0,
            local_shift_range=2.0,
            adaptive_fraction=0.95,
            oversampling_order=1,
            max_adaptive_cells=16,
            robust_weighting=True,
            halfset_diagnostics=True,
            center_references=False,
            batch_size=batch_size,
        )
        started = time.perf_counter()
        if adaptive:
            result = ai.refine_alignment(
                particles,
                references,
                initial_poses,
                class_priors=class_priors,
                config=config,
                backend=selected_backend,
            )
        else:
            result = ai.align_to_references(
                particles,
                references,
                class_priors=class_priors,
                config=config,
                backend=selected_backend,
            )
        return result, time.perf_counter() - started

    spatial_global, spatial_global_seconds = run("spatial")
    fourier_global, fourier_global_seconds = run("fourier")
    spatial_refine, spatial_refine_seconds = run("spatial", spatial_global.poses)
    fourier_refine, fourier_refine_seconds = run("fourier", spatial_global.poses)
    cpu_global = fourier_global
    cpu_refine = fourier_refine
    if backend != "cpu":
        cpu_global, _ = run("fourier", selected_backend="cpu")
        cpu_refine, _ = run(
            "fourier", spatial_global.poses, selected_backend="cpu"
        )

    def comparison(
        spatial: ai.AlignmentResult,
        fourier: ai.AlignmentResult,
        cpu: ai.AlignmentResult,
        spatial_seconds: float,
        fourier_seconds: float,
    ) -> dict[str, Any]:
        angle_delta = np.abs(
            (fourier.poses.angle_deg - spatial.poses.angle_deg + 180.0) % 360.0
            - 180.0
        )
        shift_y_delta = np.abs(fourier.poses.shift_y_px - spatial.poses.shift_y_px)
        shift_x_delta = np.abs(fourier.poses.shift_x_px - spatial.poses.shift_x_px)
        correlations = stack_correlations(fourier.references, spatial.references)
        cpu_correlations = stack_correlations(fourier.references, cpu.references)
        top_l = _top_l_posterior_comparison(spatial.candidates, fourier.candidates)
        halfset_weight_error = float(
            np.max(
                np.abs(
                    fourier.diagnostics[-1]["halfset_effective_weight"]
                    - spatial.diagnostics[-1]["halfset_effective_weight"]
                )
            )
        )
        metrics = {
            "spatial_seconds": spatial_seconds,
            "fourier_seconds": fourier_seconds,
            "spatial_to_fourier_speed_ratio": spatial_seconds
            / max(fourier_seconds, 1e-12),
            "assignment_disagreement_fraction": float(
                np.mean(fourier.reference_assignments != spatial.reference_assignments)
            ),
            "pose_angle_delta_deg": {
                "maximum": float(np.max(angle_delta)),
                "median": float(np.median(angle_delta)),
            },
            "pose_shift_delta_px": {
                "maximum_y": float(np.max(shift_y_delta)),
                "maximum_x": float(np.max(shift_x_delta)),
            },
            "reference_correlation": correlations,
            "minimum_reference_correlation": float(np.min(correlations)),
            "cpu_backend_reference_correlation": cpu_correlations,
            "minimum_cpu_backend_reference_correlation": float(
                np.min(cpu_correlations)
            ),
            "responsibility_mae": float(
                np.mean(np.abs(fourier.responsibilities - spatial.responsibilities))
            ),
            "halfset_effective_weight_maximum_error": halfset_weight_error,
            "frc_is_finite": bool(
                np.all(np.isfinite(fourier.diagnostics[-1]["frc"]))
            ),
            "top_l_posterior": top_l,
            "fourier_metadata": fourier.metadata,
        }
        if metrics["assignment_disagreement_fraction"] != 0.0:
            raise AssertionError("Fourier M-step changed fixed-class assignments")
        if metrics["pose_angle_delta_deg"]["maximum"] > 1e-6:
            raise AssertionError("Fourier M-step changed the selected angle")
        if (
            metrics["pose_shift_delta_px"]["maximum_y"] > 1.0
            or metrics["pose_shift_delta_px"]["maximum_x"] > 1.0
        ):
            raise AssertionError("Fourier M-step changed the centered shift")
        if metrics["responsibility_mae"] > 1e-7:
            raise AssertionError("Fourier M-step changed responsibilities")
        if top_l["maximum_posterior_total_variation"] > 1e-6:
            raise AssertionError("Fourier M-step changed the top-L posterior")
        if metrics["minimum_reference_correlation"] < 0.99:
            raise AssertionError("Fourier M-step diverged from the spatial reference")
        if metrics["minimum_cpu_backend_reference_correlation"] < 0.9999:
            raise AssertionError("GPU Fourier M-step diverged from CPU authority")
        if halfset_weight_error > 1e-5 or not metrics["frc_is_finite"]:
            raise AssertionError("Fourier M-step broke half-set diagnostics")
        return metrics

    return {
        "parameters": {
            "backend": backend,
            "particle_count": len(particles),
            "size": particles.shape[1],
            "reference_count": len(references),
            "batch_size": batch_size,
        },
        "global": comparison(
            spatial_global,
            fourier_global,
            cpu_global,
            spatial_global_seconds,
            fourier_global_seconds,
        ),
        "adaptive_refine": comparison(
            spatial_refine,
            fourier_refine,
            cpu_refine,
            spatial_refine_seconds,
            fourier_refine_seconds,
        ),
        "gpu_memory": gpu_memory_snapshot() if backend != "cpu" else None,
    }


def whitened_scoring_ab_case(backend: str, batch_size: int) -> dict[str, Any]:
    """Check empirical whitening on deterministic radial colored noise."""
    rng = np.random.default_rng(5)
    size = 48
    particle_count = 24
    y, x = np.indices((size, size), dtype=np.float32)
    reference = np.exp(-((y - 13) ** 2 + (x - 31) ** 2) / 2.0)
    reference += 0.8 * np.exp(-((y - 32) ** 2 + (x - 17) ** 2) / 3.0)
    reference -= 0.5 * np.exp(-((y - 25) ** 2 + (x - 34) ** 2) / 1.5)
    reference = reference.astype(np.float32)
    source_angles = rng.choice(np.arange(-150, 181, 30), size=particle_count).astype(
        np.float32
    )
    particles = np.stack(
        [
            ai.transform_images(
                reference[None],
                ai.PoseSet(
                    np.asarray([angle], dtype=np.float32),
                    np.zeros(1, dtype=np.float32),
                    np.zeros(1, dtype=np.float32),
                    np.zeros(1, dtype=np.bool_),
                ),
            )[0]
            for angle in source_angles
        ]
    )
    colored_noise = np.stack(
        [gaussian_filter(rng.normal(size=(size, size)), 2.5) for _ in source_angles]
    ).astype(np.float32)
    colored_noise /= colored_noise.std(axis=(1, 2), keepdims=True)
    particles += colored_noise * particles.std() * 6.0
    common = dict(
        candidate_scoring="fourier",
        reference_update="fourier",
        max_iterations=1,
        top_l=4,
        angle_samples=72,
        proposal_angles_per_reference=72,
        translation_range=0.0,
        halfset_diagnostics=False,
        center_references=False,
        batch_size=batch_size,
    )

    def run(score_model: str, selected_backend: str):
        started = time.perf_counter()
        result = ai.align_to_references(
            particles,
            reference,
            config=ai.AlignmentConfig(score_model=score_model, **common),
            backend=selected_backend,
        )
        return result, time.perf_counter() - started

    uniform, uniform_seconds = run("fourier_ncc", backend)
    whitened, whitened_seconds = run("whitened_fourier_ncc", backend)
    cpu_whitened = whitened
    if backend != "cpu":
        cpu_whitened, _ = run("whitened_fourier_ncc", "cpu")
    expected_angles = (-source_angles + 180.0) % 360.0 - 180.0

    def errors(result: ai.AlignmentResult) -> np.ndarray:
        return np.abs(
            (result.poses.angle_deg - expected_angles + 180.0) % 360.0 - 180.0
        )

    uniform_error = errors(uniform)
    whitened_error = errors(whitened)
    backend_angle_delta = np.abs(
        (whitened.poses.angle_deg - cpu_whitened.poses.angle_deg + 180.0) % 360.0
        - 180.0
    )
    prepared = prepare_stack(
        particles, ai.AlignmentConfig(score_model="whitened_fourier_ncc")
    )
    active = prepared.frequency_mask > 0.0
    expanded_weights = np.stack(
        [score_weights_for_particle(prepared, index) for index in range(particle_count)]
    )
    metrics = {
        "parameters": {
            "backend": backend,
            "particle_count": particle_count,
            "size": size,
            "colored_noise_gaussian_sigma_px": 2.5,
            "colored_noise_to_signal_std": 6.0,
            "batch_size": batch_size,
        },
        "uniform": {
            "correct_within_5_deg": int(np.sum(uniform_error <= 5.0)),
            "mean_angle_error_deg": float(np.mean(uniform_error)),
            "seconds": uniform_seconds,
        },
        "whitened": {
            "correct_within_5_deg": int(np.sum(whitened_error <= 5.0)),
            "mean_angle_error_deg": float(np.mean(whitened_error)),
            "seconds": whitened_seconds,
            "metadata": whitened.metadata,
        },
        "score_weights": {
            "profile_count": int(len(prepared.score_weight_profiles)),
            "active_mean": float(np.mean(expanded_weights[:, active])),
            "active_minimum": float(np.min(expanded_weights[:, active])),
            "active_maximum": float(np.max(expanded_weights[:, active])),
            "finite": bool(np.all(np.isfinite(expanded_weights))),
        },
        "cpu_backend": {
            "maximum_angle_delta_deg": float(np.max(backend_angle_delta)),
            "maximum_posterior_error": float(
                np.max(
                    np.abs(
                        whitened.candidates.posterior
                        - cpu_whitened.candidates.posterior
                    )
                )
            ),
        },
        "gpu_memory": gpu_memory_snapshot() if backend != "cpu" else None,
    }
    if metrics["uniform"]["correct_within_5_deg"] > 20:
        raise AssertionError("colored-noise control is not discriminating")
    if metrics["whitened"]["correct_within_5_deg"] != particle_count:
        raise AssertionError("empirical whitening did not recover the known poses")
    if not metrics["score_weights"]["finite"]:
        raise AssertionError("empirical whitening produced non-finite weights")
    if abs(metrics["score_weights"]["active_mean"] - 1.0) > 1e-5:
        raise AssertionError("empirical whitening weights are not normalized")
    if metrics["cpu_backend"]["maximum_angle_delta_deg"] > 1e-3:
        raise AssertionError("backend whitening pose diverged from CPU authority")
    if metrics["cpu_backend"]["maximum_posterior_error"] > 2e-4:
        raise AssertionError("backend whitening posterior diverged from CPU authority")
    return metrics


def execution_profiling_case(backend: str, batch_size: int) -> dict[str, Any]:
    try:
        from tools.performance_validation import profiling_case
    except ModuleNotFoundError:
        from performance_validation import profiling_case
    # Existing validation supports auto/gpu aliases; compare the resolved backend.
    if backend in {"auto", "gpu"}:
        status = ai.available_alignment_backends()
        backend = "cuda" if status["cuda"]["available"] else (
            "cupy" if status["cupy"]["available"] else "cpu"
        )
    return profiling_case(backend, batch_size)


def suite_cases(
    suite: str, backend: str, batch_size: int
) -> list[tuple[str, Callable[[], dict[str, Any]]]]:
    cases: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("cuda_warmup", lambda: warmup_case(backend, batch_size)),
        ("execution_profiling", lambda: execution_profiling_case(backend, batch_size)),
        ("integer_center_contract", lambda: integer_center_contract_case(backend)),
        (
            "fourier_native_conformance",
            lambda: fourier_native_conformance_case(backend),
        ),
        (
            "native_indexed_fourier",
            lambda: native_indexed_fourier_case(backend),
        ),
        (
            "native_fused_fourier_accumulation",
            lambda: native_fused_fourier_accumulation_case(backend),
        ),
        (
            "fourier_workflow_ab",
            lambda: fourier_workflow_ab_case(backend, batch_size),
        ),
        (
            "fourier_mstep_workflow_ab",
            lambda: fourier_mstep_workflow_ab_case(backend, batch_size),
        ),
        (
            "whitened_scoring_ab",
            lambda: whitened_scoring_ab_case(backend, batch_size),
        ),
        ("cpu_gpu_parity", lambda: parity_case(backend, batch_size)),
        (
            "final_raw_average",
            lambda: final_raw_average_case(backend, batch_size),
        ),
        ("polar_proposal_recall", proposal_recall_case),
        ("quadratic_refine", lambda: quadratic_refine_case(backend, batch_size)),
    ]
    if suite == "quick":
        cases.extend(
            [
                (
                    "global_snr_0_5",
                    lambda: global_case(
                        backend=backend,
                        particle_count=64,
                        size=64,
                        reference_count=4,
                        snr=0.5,
                        iterations=1,
                        angle_samples=64,
                        top_l=4,
                        batch_size=batch_size,
                        seed=20,
                    ),
                ),
                (
                    "global_snr_0_2",
                    lambda: global_case(
                        backend=backend,
                        particle_count=64,
                        size=64,
                        reference_count=4,
                        snr=0.2,
                        iterations=1,
                        angle_samples=64,
                        top_l=4,
                        batch_size=batch_size,
                        seed=21,
                    ),
                ),
                (
                    "reference_free",
                    lambda: reference_free_case(
                        backend, 96, 64, 4, 0.5, 2, batch_size, 22
                    ),
                ),
                (
                    "robust_refine",
                    lambda: refine_case(backend, 64, 64, 4, 0.5, batch_size, 23),
                ),
                ("adaptive_refine", lambda: adaptive_refine_case(backend, batch_size)),
                ("adaptive_rescue", lambda: adaptive_rescue_case(backend, batch_size)),
                ("feedback_refine", lambda: feedback_refine_case(backend, batch_size)),
                (
                    "scale_k20",
                    lambda: global_case(
                        backend=backend,
                        particle_count=256,
                        size=128,
                        reference_count=20,
                        snr=0.5,
                        iterations=1,
                        angle_samples=128,
                        top_l=8,
                        batch_size=batch_size,
                        seed=24,
                    ),
                ),
            ]
        )
    elif suite == "standard":
        cases.extend(
            [
                (
                    "global_snr_0_5",
                    lambda: global_case(
                        backend=backend,
                        particle_count=256,
                        size=128,
                        reference_count=8,
                        snr=0.5,
                        iterations=2,
                        angle_samples=128,
                        top_l=8,
                        batch_size=batch_size,
                        seed=30,
                    ),
                ),
                (
                    "global_snr_0_2",
                    lambda: global_case(
                        backend=backend,
                        particle_count=256,
                        size=128,
                        reference_count=8,
                        snr=0.2,
                        iterations=2,
                        angle_samples=128,
                        top_l=8,
                        batch_size=batch_size,
                        seed=31,
                    ),
                ),
                (
                    "reference_free",
                    lambda: reference_free_case(
                        backend, 256, 96, 8, 0.5, 3, batch_size, 32
                    ),
                ),
                (
                    "robust_refine",
                    lambda: refine_case(backend, 256, 96, 8, 0.5, batch_size, 33),
                ),
                ("adaptive_refine", lambda: adaptive_refine_case(backend, batch_size)),
                ("adaptive_rescue", lambda: adaptive_rescue_case(backend, batch_size)),
                ("feedback_refine", lambda: feedback_refine_case(backend, batch_size)),
                (
                    "scale_k20",
                    lambda: global_case(
                        backend=backend,
                        particle_count=1024,
                        size=128,
                        reference_count=20,
                        snr=0.5,
                        iterations=1,
                        angle_samples=128,
                        top_l=8,
                        batch_size=batch_size,
                        seed=34,
                    ),
                ),
                (
                    "scale_k50",
                    lambda: global_case(
                        backend=backend,
                        particle_count=1024,
                        size=128,
                        reference_count=50,
                        snr=0.5,
                        iterations=1,
                        angle_samples=128,
                        top_l=8,
                        batch_size=batch_size,
                        seed=35,
                    ),
                ),
            ]
        )
    elif suite == "tuning":
        cases.extend(
            [
                (
                    "low_snr_ablation",
                    lambda: low_snr_ablation_case(backend, batch_size),
                ),
                (
                    "reference_free_ablation",
                    lambda: rf_ablation_case(backend, batch_size),
                ),
                ("batch_sweep_k50", lambda: batch_sweep_case(backend)),
            ]
        )
    else:
        cases.extend(
            [
                (
                    "global_snr_0_5",
                    lambda: global_case(
                        backend=backend,
                        particle_count=1024,
                        size=128,
                        reference_count=20,
                        snr=0.5,
                        iterations=3,
                        angle_samples=128,
                        top_l=8,
                        batch_size=batch_size,
                        seed=40,
                    ),
                ),
                (
                    "global_snr_0_2",
                    lambda: global_case(
                        backend=backend,
                        particle_count=1024,
                        size=128,
                        reference_count=20,
                        snr=0.2,
                        iterations=3,
                        angle_samples=128,
                        top_l=8,
                        batch_size=batch_size,
                        seed=41,
                    ),
                ),
                (
                    "reference_free",
                    lambda: reference_free_case(
                        backend, 1024, 128, 20, 0.5, 3, batch_size, 42
                    ),
                ),
                (
                    "robust_refine",
                    lambda: refine_case(backend, 1024, 128, 20, 0.5, batch_size, 43),
                ),
                ("adaptive_refine", lambda: adaptive_refine_case(backend, batch_size)),
                ("adaptive_rescue", lambda: adaptive_rescue_case(backend, batch_size)),
                ("feedback_refine", lambda: feedback_refine_case(backend, batch_size)),
                (
                    "scale_10000_k20",
                    lambda: global_case(
                        backend=backend,
                        particle_count=10000,
                        size=128,
                        reference_count=20,
                        snr=0.5,
                        iterations=1,
                        angle_samples=128,
                        top_l=8,
                        batch_size=batch_size,
                        seed=44,
                    ),
                ),
                (
                    "scale_10000_k50",
                    lambda: global_case(
                        backend=backend,
                        particle_count=10000,
                        size=128,
                        reference_count=50,
                        snr=0.5,
                        iterations=1,
                        angle_samples=128,
                        top_l=8,
                        batch_size=batch_size,
                        seed=45,
                    ),
                ),
            ]
        )
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("quick", "standard", "tuning", "full"),
        default="standard",
    )
    parser.add_argument(
        "--backend",
        choices=("cpu", "cuda", "cupy", "gpu", "auto"),
        default="auto",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--only",
        help="Comma-separated case names to run; other cases are omitted.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load an existing output file and skip cases that already passed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = args.output or Path(f"alignimg-validation-{args.suite}-{timestamp}.json")
    if args.resume and output.exists():
        report = json.loads(output.read_text(encoding="utf-8"))
        report["resumed_at_utc"] = utc_now()
    else:
        report = {
            "schema_version": 1,
            "started_at_utc": utc_now(),
            "suite": args.suite,
            "backend": args.backend,
            "device": args.device,
            "batch_size": args.batch_size,
            "environment": environment_info(args.backend, args.device),
            "cases": {},
        }
    write_report(output, report)
    requested = None if not args.only else set(args.only.split(","))
    cases = suite_cases(args.suite, args.backend, args.batch_size)
    known = {name for name, _ in cases}
    if requested is not None and not requested <= known:
        raise SystemExit(f"unknown --only cases: {sorted(requested - known)}")

    failed = 0
    print(f"AlignImg validation suite={args.suite} backend={args.backend}")
    print(f"Incremental report: {output.resolve()}")
    for name, operation in cases:
        if requested is not None and name not in requested:
            continue
        previous = report["cases"].get(name)
        if args.resume and previous and previous.get("status") == "passed":
            print(f"SKIP {name}: already passed")
            continue
        print(f"RUN  {name}", flush=True)
        started = time.perf_counter()
        case_started_at = utc_now()
        try:
            payload = operation()
            case = {
                "status": "passed",
                "started_at_utc": case_started_at,
                "wall_seconds": time.perf_counter() - started,
                **payload,
            }
            print(f"PASS {name}: {case['wall_seconds']:.3f} s", flush=True)
        except Exception as error:
            failed += 1
            case = {
                "status": "failed",
                "started_at_utc": case_started_at,
                "wall_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "gpu_memory": gpu_memory_snapshot(),
            }
            print(f"FAIL {name}: {type(error).__name__}: {error}", flush=True)
        report["cases"][name] = case
        report["updated_at_utc"] = utc_now()
        write_report(output, report)

    statuses = [case.get("status") for case in report["cases"].values()]
    report["completed_at_utc"] = utc_now()
    report["summary"] = {
        "passed": statuses.count("passed"),
        "failed": statuses.count("failed"),
        "total_recorded": len(statuses),
    }
    write_report(output, report)
    print(json.dumps(report["summary"], sort_keys=True))
    print(f"Report saved to: {output.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
