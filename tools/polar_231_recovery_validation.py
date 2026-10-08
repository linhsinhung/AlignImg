#!/usr/bin/env python3
"""T5 fault-injected GPU recovery; not a physical VRAM-exhaustion benchmark."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
import time
import traceback
from unittest.mock import patch
import weakref

import numpy as np

from alignimg._fourier import prepare_stack
from alignimg._profiling import ExecutionProfile, _ACTIVE

if __package__:
    from tools import polar_231_batch_validation as batch
else:
    import polar_231_batch_validation as batch

mirror = batch.mirror


@contextmanager
def injected_oom(mode, particle_shape):
    """Exercise real GPU kernels, injecting only the allocator failure signal."""
    from alignimg_gpu import backend

    upload, solve, download = (
        backend._asdevice,
        backend._gpu_polar_batch_solver,
        backend._ashost,
    )
    once, planner = (
        backend._gpu_polar_hard_candidate_inference_once,
        backend._polar_memory_plan,
    )
    state = {
        "mode": mode,
        "attempts": [],
        "faults": 0,
        "solver_calls": 0,
        "released_before_replan": True,
        "candidate_active": False,
        "compact_result_d2h_bytes": 0,
        "full_correlation_map_d2h_bytes": 0,
    }
    failed_arrays, retained_errors = [], []

    def fail(device):
        failed_arrays.append(weakref.ref(device))
        error = RuntimeError("T5 injected out of memory")
        retained_errors.append(error)
        state["faults"] += 1
        raise error

    def attempt(*args, **kwargs):
        state["attempts"].append(
            {
                "policy": kwargs["particle_storage_policy"],
                "batch": kwargs["particle_limit"],
            }
        )
        state["candidate_active"] = True
        try:
            return once(*args, **kwargs)
        finally:
            state["candidate_active"] = False

    def allocate(values, **kwargs):
        device = upload(values, **kwargs)
        if (
            mode == "resident_upload"
            and state["faults"] == 0
            and tuple(values.shape) == tuple(particle_shape)
        ):
            fail(device)
        return device

    def solver(*args, **kwargs):
        state["solver_calls"] += 1
        current = state["attempts"][-1]
        should_fail = (
            mode == "batch1"
            or mode == "mid_batch"
            and state["solver_calls"] == 2
            or mode == "cache_then_halving"
            and (
                current["policy"] in {"resident", "cached"}
                or current["batch"] > max(1, state["attempts"][0]["batch"] // 2)
            )
        )
        packed = solve(*args, **kwargs)
        if should_fail:
            fail(packed)
        return packed

    def plan(*args, **kwargs):
        released = all(ref() is None for ref in failed_arrays)
        state["released_before_replan"] &= released
        assert released, "failed GPU array retained during replan"
        return planner(*args, **kwargs)

    def ashost(values):
        if state["candidate_active"]:
            assert values.ndim == 2 and values.shape[1] == 11, "non-compact polar D2H"
            state["compact_result_d2h_bytes"] += int(values.nbytes)
        return download(values)

    with (
        patch.object(backend, "_asdevice", allocate),
        patch.object(backend, "_gpu_polar_batch_solver", solver),
        patch.object(backend, "_gpu_polar_hard_candidate_inference_once", attempt),
        patch.object(backend, "_polar_memory_plan", plan),
        patch.object(backend, "_ashost", ashost),
    ):
        try:
            yield state
        finally:
            state["failed_array_count"] = len(failed_arrays)
            state["failed_arrays_released"] = all(
                ref() is None for ref in failed_arrays
            )
            state["injected_tracebacks_detached"] = all(
                e.__traceback__ is None for e in retained_errors
            )


def run_candidate(engine, particles, refs, config, priors, centers, mode, *, report):
    from alignimg_gpu import backend
    from alignimg_gpu._workspace import WorkflowGpuWorkspace

    records = []
    report["memory_records"] = records
    workspace = WorkflowGpuWorkspace(backend._cupy(), config, records)
    try:
        if mode == "cache_then_halving":
            for role in ("update", "scoring"):
                workspace.acquire_fourier(
                    role,
                    particles.fourier,
                    fixed_bytes=0,
                    bytes_per_item=1,
                    upload=backend._asdevice,
                )
            assert (
                workspace.allocation_budget()["live_cache_bytes"]
                == 2 * particles.fourier.nbytes
            )
        with injected_oom(mode, particles.spatial.shape) as faults:
            report["faults"] = faults
            result = backend._gpu_candidate_inference(
                particles,
                refs,
                config,
                priors,
                0.08,
                None,
                translation_centers=centers,
                engine=engine,
                workspace=workspace,
                memory_records=records,
            )
        assert faults["faults"] > 0
        assert (
            faults["failed_arrays_released"] and faults["injected_tracebacks_detached"]
        )
        plans = [r for r in records if r.get("item_unit") and "event" not in r]
        assert all(
            p["fits_minimum"] and p["estimated_peak_bytes"] <= p["budget_bytes"]
            for p in plans
        )
        assert all(p["requested_particle_batch"] == config.batch_size for p in plans)
        assert records[-1]["event"] == "complete"
        assert records[-1]["full_correlation_map_d2h_bytes"] == 0
        assert records[-1]["oom_retry_count"] == faults["faults"]
        assert records[-1]["particle_storage_policy"] == "streaming"
        if mode == "cache_then_halving":
            actions = [r["action"] for r in records if r.get("event") == "oom_retry"]
            assert actions[:3] == ["streaming", "evict_caches", "halve_batch"]
            expected_evictions = (
                ["polar_spatial"]
                if faults["attempts"][0]["policy"] == "cached" else []
            ) + ["update", "scoring"]
            assert [
                r["workspace_role"] for r in records if r.get("event") == "evict"
            ] == expected_evictions
        return result
    finally:
        workspace.close()
        report["workspace"] = workspace.summary()


def validate_recovery(engine, *, batch_size=512, report=None):
    from alignimg_gpu import backend

    if report is None:
        report = {}
    images, references, centers, labels, config = batch.batch_fixture()
    config = replace(config, batch_size=batch_size)
    particles, refs = prepare_stack(images, config), prepare_stack(references, config)
    report.update(
        config=asdict(config),
        particle_count=len(images),
        cases={},
        inputs={
            name: mirror._array_hash(value)
            for name, value in (
                ("images", images),
                ("references", references),
                ("centers", centers),
                ("labels", labels),
            )
        },
    )
    for fixed in (False, True):
        priors = (
            np.eye(3, dtype=np.float32)[labels]
            if fixed
            else np.full((len(images), 3), 1 / 3, np.float32)
        )
        expected = backend._gpu_candidate_inference(
            particles,
            refs,
            config,
            priors,
            0.08,
            None,
            translation_centers=centers,
            engine=engine,
        )
        for mode in ("resident_upload", "mid_batch", "cache_then_halving"):
            name = f"fixed_{fixed}_{mode}"
            print(f"RUN {name}: injected warm-up + 3 repeats + profile", flush=True)
            entry = {}
            report["cases"][name] = entry
            report["active_case"] = name
            warm = run_candidate(
                engine, particles, refs, config, priors, centers, mode, report={}
            )
            errors = batch.compare_batch_candidates(expected, warm)
            timings = []
            for _ in range(3):
                mirror.synchronize(engine)
                started = time.perf_counter()
                actual = run_candidate(
                    engine, particles, refs, config, priors, centers, mode, report={}
                )
                mirror.synchronize(engine)
                timings.append(time.perf_counter() - started)
                for key in warm:
                    np.testing.assert_array_equal(actual[key], warm[key], err_msg=key)
            profile = ExecutionProfile()
            token = _ACTIVE.set(profile)
            try:
                profile.attach_cuda(backend._cupy())
                actual = run_candidate(
                    engine, particles, refs, config, priors, centers, mode, report=entry
                )
                mirror.synchronize(engine)
            finally:
                try:
                    profile.close()
                    entry["profile"] = profile.asdict()
                finally:
                    _ACTIVE.reset(token)
            for key in warm:
                np.testing.assert_array_equal(actual[key], warm[key], err_msg=key)
            entry.update(
                errors=errors,
                timings_seconds=timings,
                median_seconds=float(np.median(timings)),
                deterministic_repeat_exact=True,
                result_hashes={
                    key: mirror._array_hash(value) for key, value in actual.items()
                },
            )
            if fixed:
                np.testing.assert_array_equal(actual["reference_index"][:, 0], labels)
            plans = [
                r
                for r in entry["memory_records"]
                if r.get("item_unit") and "event" not in r
            ]
            assert profile.memory["tracked_pool_live_bytes_peak"] <= max(
                p["live_cache_bytes"] + p["estimated_peak_bytes"] for p in plans
            )
            device_increase = max(
                0,
                profile.memory["sampled_device_used_bytes_peak"]
                - profile.memory["entry"]["device_used_bytes"],
            )
            entry["observed_device_increase_bytes"] = device_increase
            assert device_increase <= entry["workspace"]["budget_bytes"]
            assert (
                profile.counters["d2h_bytes"]
                == entry["faults"]["compact_result_d2h_bytes"]
            )
    report.pop("active_case")
    return report


def validate_workflow(engine, *, report):
    from alignimg_gpu import backend

    images, references, _, labels, config = batch.batch_fixture(21)
    config = replace(
        config, translation_range=0.0, batch_size=7, profile_execution=True
    )
    priors = mirror.ai.make_class_priors(assignments=labels, n_components=3)
    arguments = dict(config=config, class_priors=priors, backend=engine)
    expected = mirror.ai.align_to_references(images, references, **arguments)
    update = backend._gpu_reference_updater
    created, steps = [], []
    workspace_type = backend.WorkflowGpuWorkspace

    def workspace(*args):
        value = workspace_type(*args)
        created.append(value)
        return value

    def mstep(*args, **kwargs):
        steps.append(1)
        return update(*args, **kwargs)

    with (
        patch.object(backend, "WorkflowGpuWorkspace", workspace),
        patch.object(backend, "_gpu_reference_updater", mstep),
    ):
        with injected_oom("mid_batch", images.shape) as faults:
            actual = mirror.ai.align_to_references(images, references, **arguments)
        report["recovered"] = {
            "faults": faults,
            "errors": mirror.compare_workflows(expected, actual),
            "metadata": actual.metadata,
            "mstep_calls": len(steps),
        }
        assert len(steps) == config.max_iterations
        assert (
            actual.metadata["backend"] == engine
            and actual.metadata["gpu_workspace"]["closed"]
        )
        np.testing.assert_array_equal(actual.reference_assignments, labels)
        steps.clear()
        with injected_oom("batch1", images.shape) as faults:
            try:
                mirror.ai.align_to_references(images, references, **arguments)
            except MemoryError as error:
                report["terminal"] = {
                    "expected_error": str(error),
                    "memory_records": error.polar_memory_records,
                    "faults": faults,
                    "mstep_calls": len(steps),
                }
            else:
                raise AssertionError("batch-1 failure was swallowed")
        assert report["terminal"]["memory_records"][-2]["event"] == "oom_failure"
        assert report["terminal"]["faults"]["failed_arrays_released"]
        assert report["terminal"]["faults"]["injected_tracebacks_detached"]
        assert not steps and all(w.summary()["closed"] for w in created)
        report["terminal"]["workspace"] = created[-1].summary()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("cuda", "cupy"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if not 2 <= args.batch_size <= 512:
        parser.error("batch-size must be 2..512 for the 515-particle mid-batch fixture")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = {
        "schema": "alignimg.polar-231-recovery.v1",
        "task": "T5",
        "status": "running",
        "started_utc": mirror.utc_now(),
        "backend": args.backend,
        "source": mirror.source_manifest(),
        "checks": {},
        "note": "Fault-injected real-GPU recovery. Not real VRAM exhaustion or throughput of a physical 8 GiB card.",
    }
    started = time.perf_counter()
    try:
        batch.load_frozen_backend()
        report["frozen_t2_backend_sha256"] = batch.FROZEN_SHA256
        report["runtime_identity"] = mirror.runtime_identity(args.backend)
        from alignimg_gpu import backend, _polar_memory, _workspace

        report["loaded_sources"] = {}
        for module in (mirror.ai, mirror._engine, backend, _polar_memory, _workspace):
            relative = (
                "src/alignimg/"
                if module.__name__.startswith("alignimg.") or module is mirror.ai
                else "packages/alignimg-gpu/src/alignimg_gpu/"
            ) + Path(module.__file__).name
            loaded = mirror.sha256(Path(module.__file__))
            if loaded != mirror.sha256(mirror.ROOT / relative):
                raise RuntimeError(f"installed source mismatch: {module.__file__}")
            report["loaded_sources"][relative] = {
                "path": module.__file__,
                "sha256": loaded,
            }
        report["environment"] = mirror.environment_info(args.backend, 0)
        report["checks"]["candidate_recovery"] = {}
        validate_recovery(
            args.backend,
            batch_size=args.batch_size,
            report=report["checks"]["candidate_recovery"],
        )
        report["checks"]["workflow"] = {}
        validate_workflow(args.backend, report=report["checks"]["workflow"])
        report["status"] = "passed"
    except Exception as error:
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        if hasattr(error, "polar_memory_records"):
            report["failure_memory_records"] = error.polar_memory_records
    report["wall_seconds"] = time.perf_counter() - started
    mirror.write_report(args.output, report)
    print(f"{report['status'].upper()}: {args.output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
