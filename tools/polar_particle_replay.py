#!/usr/bin/env python3
"""Bounded, diagnostic-only replay of dev12 particle 1890; no reference updates."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import traceback
from unittest.mock import patch

import numpy as np

from alignimg import _engine, _polar_hard as cpu
from alignimg._fourier import prepare_stack

if __package__:
    from tools import polar_center_ab_validation as ab
else:
    import polar_center_ab_validation as ab

integration, mirror = ab.integration, ab.mirror
INDEX = 1890
ARCHIVE_SHA256 = "5b63c72691a740823ce2c493460ee3921ea346e4093ff2fcd524de821941c339"


def centers(values):
    return np.column_stack(
        (values["_polar_center_y_px"][:, 0], values["_polar_center_x_px"][:, 0])
    )


def candidate_summary(values):
    return {name: value.tolist() for name, value in values.items()}


def grid_summary(center, size, config):
    radius = cpu.polar_radius(size, config)
    grid, rejected = cpu.translation_center_grid(
        *center[0],
        size=size,
        radius=radius,
        translation_range=config.translation_range,
        translation_step=config.translation_step,
    )
    offsets = np.arange(
        -config.translation_range,
        config.translation_range + 0.5 * config.translation_step,
        config.translation_step,
    )
    legal = set(grid)
    excluded = [
        (float(center[0, 0] + y), float(center[0, 1] + x))
        for y in offsets
        for x in offsets
        if (center[0, 0] + y, center[0, 1] + x) not in legal
    ]
    return {
        "incoming_center": center.tolist(),
        "incoming_center_fp32_bits": np.asarray(center, np.float32)
        .view(np.uint32)
        .tolist(),
        "radius": radius,
        "legal_count": len(grid),
        "boundary_rejected_count": rejected,
        "legal_centers": grid,
        "excluded_centers": excluded,
    }


def cpu_candidates(particles, references, config, temperature, center):
    curves = []
    original = cpu.angular_correlation

    def capture(*args, **kwargs):
        curve = original(*args, **kwargs)
        curves.append(curve.copy())
        return curve

    with patch.object(cpu, "angular_correlation", capture):
        values = cpu.infer_polar_hard_candidates_cpu(
            particles,
            references,
            config,
            np.ones((1, 1), np.float32),
            temperature,
            None,
            translation_centers=center,
        )
    return values, np.stack(curves)


def cpu_peaks(curves):
    peak = np.argmax(curves, axis=1)
    offsets, scores, accepted = [], [], []
    for curve, index in zip(curves, peak):
        previous, middle, following = (
            curve[(index - 1) % len(curve)],
            curve[index],
            curve[(index + 1) % len(curve)],
        )
        offset, ok, _ = cpu.quadratic_peak_offset(previous, middle, following)
        offsets.append(offset)
        score = float(middle)
        if ok:
            score -= 0.25 * float(previous - following) * offset
        scores.append(score)
        accepted.append(ok)
    return peak, np.asarray(offsets), np.asarray(scores), np.asarray(accepted)


def peak_summary(peaks, grid, temperature, angle_samples):
    bins, offsets, scores, accepted = peaks
    flat_ids = np.arange(len(grid)) * angle_samples + bins
    order = np.lexsort((flat_ids, -scores))
    rows = []
    for index in order[:2]:
        rows.append(
            {
                "center_index": int(index),
                "center": list(grid[index]),
                "flat_id": int(flat_ids[index]),
                "angle_bin": int(bins[index]),
                "angle_deg": float(
                    ((bins[index] + offsets[index]) * 360 / angle_samples + 180) % 360
                    - 180
                ),
                "quadratic_accepted": bool(accepted[index]),
                "score": float(scores[index]),
                "objective": float(scores[index] / temperature),
            }
        )
    return {
        "top_two": rows,
        "objective_margin": rows[0]["objective"] - rows[1]["objective"],
        "all_center_scores": scores.tolist(),
    }


def comparison(expected, actual):
    try:
        return {"passed": True, "errors": mirror.compare_candidates(expected, actual)}
    except AssertionError as error:
        return {"passed": False, "error": str(error)}


def gpu_candidates(
    particles, references, config, temperature, center, cpu_curves, engine, record
):
    """Observe the real GPU sampler/normalizer/peak selector; never replace outputs."""
    from alignimg_gpu import backend

    cp = backend._cupy()
    sampler_name, peak_name = (
        ("_polar_sample_cuda", "_polar_peak_cuda")
        if engine == "cuda"
        else ("_polar_sample_cupy", "_polar_peak_cupy")
    )
    sampler, normalize, peak = (
        getattr(backend, sampler_name),
        backend._normalize_polar_rings_gpu,
        getattr(backend, peak_name),
    )
    record.update(samplers=[], normalizations=[], memory_records=[])
    oracle_normalized = []
    captured_peaks = []

    def sampled(images, indices, y, x, reflected, dy, dx):
        host_images = (
            references.spatial if not record["samplers"] else particles.spatial
        )
        assert len(record["samplers"]) < 2, "unexpected sampler call or retry"
        assert bool(cp.all(images == cp.asarray(host_images)).item())
        host_indices, host_y, host_x, host_mirror = [
            cp.asnumpy(v) for v in (indices, y, x, reflected)
        ]
        assert not np.any(host_mirror), "frozen case is mirror-off"
        radius = cpu.polar_radius(images.shape[-1], config)
        expected, normalized = [], []
        for index, cy, cx in zip(host_indices, host_y, host_x):
            kwargs = dict(
                center_y=float(cy),
                center_x=float(cx),
                radius=radius,
                angle_samples=config.angle_samples,
            )
            expected.append(cpu.sample_spatial_polar(host_images[index], **kwargs))
            normalized.append(cpu.spatial_polar_rings(host_images[index], **kwargs))
        actual = sampler(images, indices, y, x, reflected, dy, dx)
        difference = actual - cp.asarray(np.stack(expected))
        record["samplers"].append(
            {
                "role": "reference" if not record["samplers"] else "particle",
                "sample_count": int(actual.size),
                "maximum_abs_error": float(cp.max(cp.abs(difference)).item()),
                "different_values": int(cp.count_nonzero(difference).item()),
            }
        )
        oracle_normalized.append(np.stack(normalized))
        return actual

    def normalized(rings):
        actual = normalize(rings)
        expected = oracle_normalized[len(record["normalizations"])]
        record["normalizations"].append(
            {
                "maximum_abs_error": float(
                    cp.max(cp.abs(actual - cp.asarray(expected))).item()
                )
            }
        )
        return actual

    def selected(curves, *args):
        assert not captured_peaks, "unexpected peak call or retry"
        assert curves.shape == cpu_curves.shape
        record["correlation_curve_max_abs_error"] = float(
            cp.max(cp.abs(curves - cp.asarray(cpu_curves))).item()
        )
        output = peak(curves, *args)
        captured_peaks.extend(cp.asnumpy(value) for value in output)
        return output

    with ExitStack() as scoped:
        scoped.enter_context(patch.object(backend, sampler_name, sampled))
        scoped.enter_context(
            patch.object(backend, "_normalize_polar_rings_gpu", normalized)
        )
        scoped.enter_context(patch.object(backend, peak_name, selected))
        result = backend._gpu_candidate_inference(
            particles,
            references,
            config,
            np.ones((1, 1), np.float32),
            temperature,
            None,
            translation_centers=center,
            engine=engine,
            memory_records=record["memory_records"],
        )
    assert len(record["samplers"]) == len(record["normalizations"]) == 2
    assert len(captured_peaks) == 4
    completed = [p for p in record["memory_records"] if p.get("event") == "complete"]
    assert len(completed) == 1 and completed[0]["engine"] == engine
    assert completed[0]["full_correlation_map_d2h_bytes"] == 0
    record["full_correlation_map_d2h_bytes"] = 0
    return result, tuple(captured_peaks)


def endpoint(saved, values):
    mapping = {
        name: "candidate_" + name
        for name in (
            "angle_deg",
            "shift_y_px",
            "shift_x_px",
            "score",
            "reference_index",
            "mirror",
        )
    }
    mapping.update(
        {
            "_polar_raw_shift_y_px": "polar_raw_shift_y_px",
            "_polar_raw_shift_x_px": "polar_raw_shift_x_px",
        }
    )
    errors, passed = {}, True
    for name, stored in mapping.items():
        expected, actual = (
            np.asarray(saved[stored][INDEX]).reshape(-1),
            values[name].reshape(-1),
        )
        delta = actual.astype(float) - expected.astype(float)
        if name == "angle_deg":
            delta = (delta + 180) % 360 - 180
        tolerance = (
            1e-3
            if name == "angle_deg"
            else 2e-5
            if name == "score"
            else 0
            if name in {"reference_index", "mirror"}
            else 1e-5
        )
        errors[name] = float(np.max(np.abs(delta)))
        passed &= errors[name] <= tolerance
    return {
        "anchored_within_existing_tolerance": bool(passed),
        "maximum_abs_errors": errors,
    }


def load_inputs(path):
    import mrcfile
    import tarfile

    assert mirror.sha256(path) == ARCHIVE_SHA256, "dev12 archive hash mismatch"
    with tarfile.open(path) as archive:
        old = json.load(archive.extractfile("t6-dev12-cuda-center-ab.json"))
        saved = {}
        for variant, entry in old["variants"].items():
            payload = archive.extractfile(Path(entry["result"]).name).read()
            assert hashlib.sha256(payload).hexdigest() == entry["result_sha256"]
            with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
                saved[variant] = {name: arrays[name].copy() for name in arrays.files}
    baseline, _, hashes = integration.preflight(mirror.ROOT)
    assert old["status"] == "completed" and old["controller_restored"]
    assert old["inputs"] == hashes and old["baseline"] == baseline
    config = mirror.ai.AlignmentConfig(**baseline["config"]).normalized(
        workflow="global"
    )
    assert config.max_iterations == 3 and config.top_l == 1 and not config.mirror_search
    assert ab.legacy_helper_sha256() == ab.LEGACY_HELPER_SHA256
    with mrcfile.mmap(mirror.ROOT / "data/local/test_align.mrcs", mode="r") as stack:
        assert stack.data.shape == (3050, 100, 100)
        image = np.asarray(stack.data[INDEX : INDEX + 1], np.float32).copy()
    for variant, arrays in saved.items():
        assert arrays["reference_history"].shape == (4, 1, 100, 100)
        assert old["variants"][variant]["config"] == baseline["config"]
        assert all(
            t["reference_center_shifts"] == [[0.0, 0.0]]
            for t in old["variants"][variant]["center_trace"]
        )
    return image, config, saved, hashes


def run(engine, input_path, output, report):
    report["runtime"] = integration.runtime_info(engine)
    image, config, saved, report["inputs"] = load_inputs(input_path)
    report["config"] = asdict(config)
    particles = prepare_stack(image, config)
    report["variants"] = {}
    all_parity, all_anchored = True, True
    for variant, controller in (
        ("current", _engine._apply_reference_center_shifts),
        ("legacy", ab.legacy_center_update),
    ):
        entry = {"iterations": []}
        report["variants"][variant] = entry
        gpu_center, cpu_center = np.zeros((1, 2)), np.zeros((1, 2))
        for iteration in range(3):
            print(
                f"RUN {variant} particle={INDEX} iteration={iteration + 1}", flush=True
            )
            references = prepare_stack(
                saved[variant]["reference_history"][iteration], config
            )
            temperature = _engine._temperature(config, iteration)
            state = {
                "iteration": iteration + 1,
                "temperature": temperature,
                "gpu_driven_grid": grid_summary(gpu_center, 100, config),
                "independent_cpu_grid": grid_summary(cpu_center, 100, config),
            }
            entry["iterations"].append(state)
            cpu_same, curves = cpu_candidates(
                particles, references, config, temperature, gpu_center
            )
            independent, _ = cpu_candidates(
                particles, references, config, temperature, cpu_center
            )
            state["gpu_observation"] = {}
            gpu, peaks = gpu_candidates(
                particles,
                references,
                config,
                temperature,
                gpu_center,
                curves,
                engine,
                state["gpu_observation"],
            )
            state.update(
                cpu_same_center=candidate_summary(cpu_same),
                gpu_same_center=candidate_summary(gpu),
                independent_cpu=candidate_summary(independent),
            )
            state["parity"] = comparison(cpu_same, gpu)
            cpu_peak_values = cpu_peaks(curves)
            grid = state["gpu_driven_grid"]["legal_centers"]
            state["cpu_peaks"] = peak_summary(
                cpu_peak_values, grid, temperature, config.angle_samples
            )
            state["gpu_peaks"] = peak_summary(
                peaks, grid, temperature, config.angle_samples
            )
            score_error = float(np.max(np.abs(cpu_peak_values[2] - peaks[2])))
            state["all_center_score_max_abs_error"] = score_error
            state["all_center_objective_max_abs_error"] = score_error / temperature
            state["raw_sampling_exact"] = all(
                s["different_values"] == 0 for s in state["gpu_observation"]["samplers"]
            )
            all_parity &= (
                state["parity"]["passed"]
                and state["raw_sampling_exact"]
                and score_error <= 2e-5
                and score_error / temperature <= 1e-3
            )
            controller(gpu, [(0.0, 0.0)])
            controller(independent, [(0.0, 0.0)])
            gpu_center, cpu_center = centers(gpu), centers(independent)
            state.update(
                gpu_next_center=gpu_center.tolist(),
                independent_cpu_next_center=cpu_center.tolist(),
            )
            mirror.write_report(output, report)
        entry["gpu_endpoint"] = endpoint(saved[variant], gpu)
        entry["independent_cpu_endpoint"] = endpoint(saved[variant], independent)
        all_anchored &= entry["gpu_endpoint"]["anchored_within_existing_tolerance"]
    report.update(
        same_input_conformance_passed=bool(all_parity),
        gpu_endpoints_anchored=bool(all_anchored),
    )
    report["status"] = (
        "failed" if not all_parity else "completed" if all_anchored else "inconclusive"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "cupy"), default="cuda")
    parser.add_argument(
        "--input",
        type=Path,
        default=mirror.ROOT
        / "validation-results/maintenance-2.3.1/final/t6-dev12-return.tar.gz",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = {
        "schema": "alignimg.polar-particle-replay.v1",
        "status": "running",
        "diagnostic_only": True,
        "source": mirror.source_manifest(),
        "backend": args.backend,
        "particle_index_zero_based": INDEX,
        "input_archive_sha256": ARCHIVE_SHA256,
        "started_utc": mirror.utc_now(),
        "limits": [
            "One particle, two controller variants, three frozen references each; no M-step or full-stack alignment.",
            "GPU drives the shared-center CPU/GPU comparison; independent CPU trajectory is separate, not a backend parity oracle at different centers.",
            "N=1 changes FFT batch shape. Endpoint tolerance is necessary, not proof that every intermediate state matches the original B=64 run.",
            "Diagnostic wrappers download only sampling parameters, scalar error reductions and peak tables, never full GPU correlation maps.",
            "No production changes, threshold waivers, baseline replacement or T6 acceptance. Unanchored replay stops here.",
        ],
    }
    mirror.write_report(args.output, report)
    try:
        run(args.backend, args.input, args.output, report)
    except Exception as error:
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    mirror.write_report(args.output, report)
    print(f"DIAGNOSTIC {report['status'].upper()}: {args.output}", flush=True)
    return (
        0
        if report["status"] == "completed"
        else 2
        if report["status"] == "inconclusive"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
