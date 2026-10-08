#!/usr/bin/env python3
"""T2 mirror conformance; real GPU execution is required, never skipped."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.ndimage import gaussian_filter

import alignimg as ai
from alignimg import _engine
from alignimg._fourier import prepare_stack
from alignimg._geometry import mirror_x_integer_origin
from alignimg._polar_hard import infer_polar_hard_candidates_cpu, sample_spatial_polar

if __package__:
    from tools.fast_hard_validation import (
        _array_hash,
        polar_result_arrays,
        runtime_identity,
    )
    from tools.performance_fixtures import ROOT, sha256, source_manifest
    from tools.performance_validation import compare_arrays, synchronize
    from tools.server_validation import environment_info, utc_now, write_report
else:
    from fast_hard_validation import _array_hash, polar_result_arrays, runtime_identity
    from performance_fixtures import ROOT, sha256, source_manifest
    from performance_validation import compare_arrays, synchronize
    from server_validation import environment_info, utc_now, write_report


def mirror_fixture():
    rng = np.random.default_rng(12)
    references = np.stack(
        [
            gaussian_filter(rng.normal(size=(32, 32)).astype(np.float32), sigma=1.0)
            for _ in range(2)
        ]
    ).astype(np.float32)
    labels = np.array([0, 1, 0, 1], dtype=np.int32)
    poses = ai.PoseSet(
        np.array([0.0, 30.0, -45.0, 95.0], dtype=np.float32),
        np.array([0.0, 0.21, -0.34, 0.13], dtype=np.float32),
        np.array([0.0, -0.17, 0.27, -0.31], dtype=np.float32),
        np.array([True, False, True, False]),
    )
    images = ai.transform_images(references[labels], poses, backend="cpu")
    centers = np.array(
        [
            [1 / 64, -1 / 64],
            [-3 / 64, 3 / 64],
            [0.23, -0.37],
            [-0.11, 0.29],
        ],
        dtype=np.float64,
    )
    return images, references, centers


def mirror_config(mirror=True):
    return replace(
        ai.AlignmentConfig.preset("fast3"),
        angle_samples=64,
        mask_radius=8.0,
        translation_range=1 / 64,
        translation_step=1 / 32,
        mirror_search=mirror,
        center_references=False,
        halfset_diagnostics=False,
        batch_size=512,
    ).normalized(workflow="global")


def sampler_centers():
    values = [-3 / 64, -1 / 64, 1 / 64, 3 / 64, -0.37, 0.23]
    # Adjacent values are taken at the actual source coordinate, not before
    # adding the integer origin (which could round the distinction away).
    for midpoint in (16 - 1 / 64, 16 + 1 / 64, 16 + 3 / 64):
        values.extend(
            [
                np.nextafter(midpoint, -np.inf) - 16,
                midpoint - 16,
                np.nextafter(midpoint, np.inf) - 16,
            ]
        )
    centers = [(value, value) for value in values]
    centers.extend([(0.0, 0.0), (0.0, -13.0), (0.0, 12.0), (-13.0, 0.0), (12.0, 0.0)])
    return np.asarray(centers, dtype=np.float64)


def sampler_expected(image, centers, mirrors, radius):
    return np.stack(
        [
            sample_spatial_polar(
                mirror_x_integer_origin(image) if mirror else image,
                center_y=float(y),
                center_x=float(x),
                radius=radius,
                angle_samples=64,
            )
            for (y, x), mirror in zip(centers, mirrors, strict=True)
        ]
    )


def compare_candidates(expected, actual, temperature=0.08):
    errors = {}
    for name in (
        "reference_index",
        "mirror",
        "_polar_discrete_angle_bin",
        "_polar_quadratic_fit_accepted",
        "_polar_evaluated_center_count",
    ):
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    for name in (
        "angle_deg",
        "shift_y_px",
        "shift_x_px",
        "_polar_center_y_px",
        "_polar_center_x_px",
        "_polar_raw_shift_y_px",
        "_polar_raw_shift_x_px",
        "score",
        "_polar_objective_margin",
    ):
        delta = actual[name].astype(np.float64) - expected[name].astype(np.float64)
        if name == "angle_deg":
            delta = (delta + 180) % 360 - 180
        tolerance = (
            1e-3
            if name in {"angle_deg", "_polar_objective_margin"}
            else 2e-5
            if name == "score"
            else 1e-5
        )
        np.testing.assert_allclose(delta, 0, atol=tolerance, rtol=0, err_msg=name)
        errors[name] = float(np.max(np.abs(delta)))
    # Reference priors cancel because reference selection is required to match.
    np.testing.assert_allclose(
        (actual["score"] - expected["score"]) / temperature,
        0,
        atol=1e-3,
        rtol=0,
        err_msg="candidate objective",
    )
    return errors


def compare_workflows(expected, actual):
    before, after = polar_result_arrays(expected), polar_result_arrays(actual)
    errors = compare_arrays(before, after)
    for name in (
        "assignments",
        "mirror",
        "candidate_reference_index",
        "candidate_mirror",
    ):
        np.testing.assert_array_equal(after[name], before[name], err_msg=name)
    angle_delta = (
        after["candidate_angle_deg"].astype(np.float64)
        - before["candidate_angle_deg"].astype(np.float64)
        + 180
    ) % 360 - 180
    np.testing.assert_allclose(angle_delta, 0, atol=1e-3, rtol=0)
    for name in ("polar_raw_shift_y_px", "polar_raw_shift_x_px"):
        np.testing.assert_allclose(
            after[name], before[name], atol=1e-5, rtol=0, err_msg=name
        )
    np.testing.assert_allclose(
        after["candidate_score"] - before["candidate_score"],
        0,
        atol=2e-5,
        rtol=0,
    )
    np.testing.assert_allclose(
        (after["candidate_score"] - before["candidate_score"])
        / expected.metadata["config"]["temperature_end"],
        0,
        atol=1e-3,
        rtol=0,
    )
    return errors


def validate_sampler(engine):
    from alignimg_gpu import backend

    cp = backend._cupy()
    image = mirror_fixture()[1][0]
    centers = sampler_centers()
    centers = np.repeat(centers, 2, axis=0)
    mirrors = np.tile([False, True], len(centers) // 2)
    sampler = (
        backend._polar_sample_cuda if engine == "cuda" else backend._polar_sample_cupy
    )
    errors = []
    for radius in (4.0, 8.0):
        selected = np.abs(centers).max(axis=1) <= (13 if radius == 4 else 1)
        current, reflected = centers[selected], mirrors[selected]
        offsets = backend._polar_offsets(radius, 64, int(radius))
        actual = sampler(
            cp.asarray(image[None]),
            cp.zeros(len(current), dtype=cp.int32),
            cp.asarray(current[:, 0]),
            cp.asarray(current[:, 1]),
            cp.asarray(reflected, dtype=cp.uint8),
            cp.asarray(offsets[0]),
            cp.asarray(offsets[1]),
        )
        expected = sampler_expected(image, current, reflected, radius)
        host = cp.asnumpy(actual)
        np.testing.assert_array_equal(host, expected)
        errors.append(
            {"radius": radius, "cases": len(current), "maximum_absolute_error": 0.0}
        )
    return errors


def validate_candidate(engine, mirror=True):
    from alignimg_gpu import backend

    images, references, centers = mirror_fixture()
    config = mirror_config(mirror)
    particles, refs = prepare_stack(images, config), prepare_stack(references, config)
    priors = np.full((len(images), len(references)), 0.5, dtype=np.float32)
    expected = infer_polar_hard_candidates_cpu(
        particles,
        refs,
        config,
        priors,
        0.08,
        None,
        translation_centers=centers,
    )
    actual = backend._gpu_candidate_inference(
        particles,
        refs,
        config,
        priors,
        0.08,
        None,
        translation_centers=centers,
        engine=engine,
    )
    return compare_candidates(expected, actual)


def validate_fast3(engine, mirror=True):
    images, references, _ = mirror_fixture()
    config = mirror_config(mirror)
    expected = ai.align_to_references(images, references, config=config, backend="cpu")
    actual = ai.align_to_references(images, references, config=config, backend=engine)
    kernel = "native_cuda" if engine == "cuda" else "cupy"
    assert actual.metadata["backend"] == engine, "GPU fallback is not allowed"
    assert actual.metadata["polar_sampler_backend"] == kernel
    assert actual.metadata["polar_peak_backend"] == kernel
    assert actual.metadata["polar_full_correlation_map_d2h"] is False
    errors = compare_workflows(expected, actual)
    repeated = ai.align_to_references(images, references, config=config, backend=engine)
    for name, values in polar_result_arrays(actual).items():
        np.testing.assert_array_equal(
            values, polar_result_arrays(repeated)[name], err_msg=name
        )
    return {"errors": errors, "deterministic_repeat": True, "metadata": actual.metadata}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("cuda", "cupy"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = {
        "schema": "alignimg.polar-231-mirror.v1",
        "task": "T2",
        "started_utc": utc_now(),
        "backend": args.backend,
        "source": source_manifest(),
        "status": "running",
        "checks": {},
    }
    started = time.perf_counter()
    try:
        report["runtime_identity"] = runtime_identity(args.backend)
        from alignimg_gpu import backend, memory

        # Editable core + installed GPU must both originate from this checkout.
        report["loaded_sources"] = {}
        for module, relative in (
            (ai, "src/alignimg/__init__.py"),
            (_engine, "src/alignimg/_engine.py"),
            (backend, "packages/alignimg-gpu/src/alignimg_gpu/backend.py"),
            (memory, "packages/alignimg-gpu/src/alignimg_gpu/memory.py"),
        ):
            loaded_hash = sha256(Path(module.__file__))
            report["loaded_sources"][relative] = {
                "module_path": module.__file__, "sha256": loaded_hash,
            }
            if loaded_hash != sha256(ROOT / relative):
                raise RuntimeError(f"installed source mismatch: {module.__file__}")
        report["environment"] = environment_info(args.backend, 0)
        images, references, centers = mirror_fixture()
        report["inputs"] = {
            "images": _array_hash(images),
            "references": _array_hash(references),
            "centers": _array_hash(centers),
            "sampler_centers": _array_hash(sampler_centers()),
            "seed": 12,
        }
        report["configs"] = {
            str(mirror): asdict(mirror_config(mirror)) for mirror in (False, True)
        }
        report["checks"]["sampler"] = validate_sampler(args.backend)
        for mirror in (False, True):
            print(f"RUN candidate + Fast 3 mirror={mirror}", flush=True)
            report["checks"][f"candidate_mirror_{mirror}"] = validate_candidate(
                args.backend, mirror
            )
            report["checks"][f"fast3_mirror_{mirror}"] = validate_fast3(
                args.backend, mirror
            )
        synchronize(args.backend)
        report["status"] = "passed"
    except Exception as error:
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    report["wall_seconds"] = time.perf_counter() - started
    write_report(args.output, report)
    print(f"{report['status'].upper()}: {args.output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
