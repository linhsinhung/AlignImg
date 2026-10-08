#!/usr/bin/env python3
"""T3 resident batch boundaries, compared with the immutable accepted T2 solver."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import time
import traceback
from types import ModuleType
from unittest.mock import patch

import numpy as np
from scipy.ndimage import gaussian_filter

import alignimg as ai
from alignimg._fourier import prepare_stack

if __package__:
    from tools import polar_231_mirror_validation as mirror
else:
    import polar_231_mirror_validation as mirror


FROZEN_BACKEND = (
    mirror.ROOT / "validation-results/maintenance-2.3.1/t3-batch/frozen-t2-backend.py"
)
FROZEN_SHA256 = "cd0524a4f7d702ba0fc97ff2cc8270519789781da4ad6137a4d8d1d657ff412a"
BATCHES = (1, 7, 256, 512)


def load_frozen_backend(path=FROZEN_BACKEND):
    path = Path(path)
    if mirror.sha256(path) != FROZEN_SHA256:
        raise RuntimeError(f"frozen T2 backend source mismatch: {path}")
    module = ModuleType("alignimg_gpu._polar_231_frozen_t2")
    module.__package__ = "alignimg_gpu"
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


def batch_fixture(count=515):
    _, first, _ = mirror.mirror_fixture()
    rng = np.random.default_rng(2313)
    third = gaussian_filter(rng.normal(size=(32, 32)).astype(np.float32), 1.0)
    references = np.concatenate((first, third[None])).astype(np.float32)
    rows = np.arange(count)
    labels = (rows % 3).astype(np.int32)
    poses = ai.PoseSet(
        ((rows * 13.7 + 17) % 360 - 180).astype(np.float32),
        ((rows % 5 - 2) * 0.13).astype(np.float32),
        ((rows % 7 - 3) * 0.11).astype(np.float32),
        rows % 2 == 0,
    )
    images = ai.transform_images(references[labels], poses, backend="cpu")
    centers = np.array(
        [[1 / 64, -3 / 64], [0.23, -0.37], [7, 7], [-8, -8], [0, 7]],
        dtype=np.float64,
    )[rows % 5]
    config = replace(
        mirror.mirror_config(), translation_range=1.0, translation_step=1.0
    )
    return images, references, centers, labels, config


def compare_batch_candidates(expected, actual, temperature=0.08):
    # Canonical pose shifts depend on the fitted angle. The 1e-5 polar gate
    # applies to sampling centers/raw grid offsets, not these rotated shifts.
    pose_names = ("angle_deg", "shift_y_px", "shift_x_px")
    pose_errors = mirror.compare_arrays(
        {name: expected[name] for name in pose_names},
        {name: actual[name] for name in pose_names},
    )
    errors = {
        name: value["maximum_absolute_error"] for name, value in pose_errors.items()
    }
    for name in (
        "reference_index",
        "mirror",
        "posterior",
        "_polar_discrete_angle_bin",
        "_polar_quadratic_fit_accepted",
        "_polar_evaluated_center_count",
        "_polar_boundary_rejected_count",
    ):
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    for name, tolerance in (
        ("_polar_center_y_px", 1e-5),
        ("_polar_center_x_px", 1e-5),
        ("_polar_raw_shift_y_px", 1e-5),
        ("_polar_raw_shift_x_px", 1e-5),
        ("score", 2e-5),
        ("_polar_objective_margin", 1e-3),
    ):
        np.testing.assert_allclose(
            actual[name],
            expected[name],
            atol=tolerance,
            rtol=0,
            equal_nan=False,
            err_msg=name,
        )
        finite = np.isfinite(expected[name])
        delta = actual[name][finite].astype(np.float64) - expected[name][finite].astype(
            np.float64
        )
        errors[name] = float(np.max(np.abs(delta), initial=0))
    # Reference priors cancel because reference selection must match exactly.
    np.testing.assert_allclose(
        (actual["score"] - expected["score"]) / temperature,
        0,
        atol=1e-3,
        rtol=0,
        err_msg="candidate objective",
    )
    return errors


def validate_batches(engine, frozen, count=515, *, report=None):
    from alignimg_gpu import backend

    images, references, centers, labels, config = batch_fixture(count)
    particles, refs = prepare_stack(images, config), prepare_stack(references, config)
    if report is None:
        report = {}
    report.update(
        {
            "config": asdict(config),
            "comparison_contract": {
                "canonical_pose": "performance_validation.compare_arrays",
                "canonical_angle_atol_deg": 1e-3,
                "canonical_shift_atol_px": 1e-3,
                "sampling_center_and_raw_shift_atol_px": 1e-5,
                "score_atol": 2e-5,
                "objective_atol": 1e-3,
                "t2_same_batch_and_repeat": "exact",
            },
            "inputs": {
                name: mirror._array_hash(value)
                for name, value in (
                    ("images", images),
                    ("references", references),
                    ("centers", centers),
                    ("labels", labels),
                )
            },
            "particle_count": count,
            "cases": {},
        }
    )
    for fixed in (False, True):
        priors = (
            np.eye(3, dtype=np.float32)[labels]
            if fixed
            else np.full((count, 3), 1 / 3, dtype=np.float32)
        )
        cross_batch = None
        for batch in BATCHES:
            report["active_case"] = {
                "fixed_priors": fixed,
                "particle_batch": batch,
                "check": "t2_same_batch",
            }
            print(f"RUN resident fixed={fixed} batch={batch} N={count}", flush=True)
            arguments = (particles, refs, config, priors, 0.08, None)
            kwargs = dict(
                translation_centers=centers, engine=engine, particle_limit=batch
            )
            expected = frozen._gpu_polar_hard_candidate_inference_once(
                *arguments, **kwargs
            )
            records = []
            actual = backend._gpu_polar_hard_candidate_inference_once(
                *arguments, **kwargs, memory_records=records
            )
            errors = compare_batch_candidates(expected, actual)
            # Same backend/batch must retain exactly the T2 arithmetic.
            for name in expected:
                np.testing.assert_array_equal(
                    actual[name], expected[name], err_msg=name
                )
            report["active_case"]["check"] = "deterministic_repeat"
            repeated = backend._gpu_polar_hard_candidate_inference_once(
                *arguments, **kwargs
            )
            for name in actual:
                np.testing.assert_array_equal(
                    repeated[name], actual[name], err_msg=name
                )
            report["active_case"]["check"] = "cross_batch"
            cross_errors = (
                compare_batch_candidates(cross_batch, actual)
                if cross_batch is not None
                else None
            )
            if cross_batch is None:
                cross_batch = actual
            report["active_case"]["check"] = "assignment_and_memory_records"
            if fixed:
                np.testing.assert_array_equal(actual["reference_index"][:, 0], labels)
            complete = records[-1]
            assert complete["particle_storage_policy"] == "resident"
            assert complete["maximum_winner_batch_size"] == min(count, batch)
            assert complete["result_download_batches"] == (count + batch - 1) // batch
            assert complete["final_result_d2h_bytes"] == count * 11 * 8
            assert complete["full_correlation_map_d2h_bytes"] == 0
            report["cases"][f"fixed_{fixed}_batch_{batch}"] = {
                "t2_same_batch_exact": True,
                "deterministic_repeat_exact": True,
                "t2_errors": errors,
                "cross_batch_errors": cross_errors,
                "result_hashes": {
                    name: mirror._array_hash(value) for name, value in actual.items()
                },
                "memory_records": records,
            }
    report.pop("active_case")
    return report


def validate_fixed_workflow(engine, frozen):
    from alignimg_gpu import backend

    images, references, _, labels, config = batch_fixture(21)
    config = replace(
        config, translation_range=0.0, batch_size=7, profile_execution=True
    )
    priors = ai.make_class_priors(assignments=labels, n_components=3)
    arguments = dict(config=config, class_priors=priors, backend=engine)

    def frozen_resident(*args, particle_storage_policy="resident", workspace=None, **kwargs):
        if particle_storage_policy != "resident":
            raise AssertionError("frozen T2 workflow comparison requires resident particles")
        return frozen._gpu_polar_hard_candidate_inference_once(*args, **kwargs)

    with (
        patch.object(backend.WorkflowGpuWorkspace, "spatial_cache_allowed", lambda self: False),
        patch.object(
            backend,
            "_gpu_polar_hard_candidate_inference_once",
            frozen_resident,
        ),
    ):
        expected = ai.align_to_references(images, references, **arguments)
    joint = ai.align_to_references(images, references, **arguments)
    assert joint.metadata["backend"] == engine, "GPU fallback is not allowed"
    assert joint.metadata["gpu_workspace"]["closed"]
    errors = mirror.compare_workflows(expected, joint)
    np.testing.assert_array_equal(joint.reference_assignments, labels)
    per_class = []
    for index in range(3):
        selected = labels == index
        single = ai.align_to_references(
            images[selected], references[index], config=config, backend=engine
        )
        assert single.metadata["backend"] == engine
        angle_delta = (
            joint.poses.angle_deg[selected] - single.poses.angle_deg + 180
        ) % 360 - 180
        shift_delta = np.hypot(
            joint.poses.shift_y_px[selected] - single.poses.shift_y_px,
            joint.poses.shift_x_px[selected] - single.poses.shift_x_px,
        )
        np.testing.assert_allclose(angle_delta, 0, atol=1e-3, rtol=0)
        np.testing.assert_allclose(shift_delta, 0, atol=1e-3, rtol=0)
        np.testing.assert_array_equal(joint.poses.mirror[selected], single.poses.mirror)
        np.testing.assert_array_equal(single.reference_assignments, 0)
        np.testing.assert_allclose(
            joint.responsibilities[selected, index],
            single.responsibilities[:, 0],
            atol=2e-4,
            rtol=3e-5,
        )
        correlation = float(
            np.corrcoef(joint.references[index].ravel(), single.references[0].ravel())[
                0, 1
            ]
        )
        assert correlation >= 0.9999
        per_class.append(
            {
                "reference_index": index,
                "angle_max_deg": float(np.max(np.abs(angle_delta))),
                "shift_max_px": float(np.max(shift_delta)),
                "reference_correlation": correlation,
            }
        )
    return {"t2_errors": errors, "per_class": per_class, "metadata": joint.metadata}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("cuda", "cupy"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = {
        "schema": "alignimg.polar-231-batch.v1",
        "task": "T3",
        "started_utc": mirror.utc_now(),
        "backend": args.backend,
        "source": mirror.source_manifest(),
        "status": "running",
        "checks": {},
    }
    started = time.perf_counter()
    try:
        frozen = load_frozen_backend()
        report["frozen_t2_backend_sha256"] = FROZEN_SHA256
        report["runtime_identity"] = mirror.runtime_identity(args.backend)
        from alignimg_gpu import backend, memory

        report["loaded_sources"] = {}
        for module, relative in (
            (ai, "src/alignimg/__init__.py"),
            (mirror._engine, "src/alignimg/_engine.py"),
            (backend, "packages/alignimg-gpu/src/alignimg_gpu/backend.py"),
            (memory, "packages/alignimg-gpu/src/alignimg_gpu/memory.py"),
        ):
            loaded_hash = mirror.sha256(Path(module.__file__))
            if loaded_hash != mirror.sha256(mirror.ROOT / relative):
                raise RuntimeError(f"installed source mismatch: {module.__file__}")
            report["loaded_sources"][relative] = {
                "path": module.__file__,
                "sha256": loaded_hash,
            }
        report["environment"] = mirror.environment_info(args.backend, 0)
        batch_report = {}
        report["checks"]["resident_batches"] = batch_report
        validate_batches(args.backend, frozen, report=batch_report)
        report["checks"]["fixed_k3_vs_k1"] = validate_fixed_workflow(
            args.backend, frozen
        )
        mirror.synchronize(args.backend)
        report["status"] = "passed"
    except Exception as error:
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    report["wall_seconds"] = time.perf_counter() - started
    mirror.write_report(args.output, report)
    print(f"{report['status'].upper()}: {args.output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
