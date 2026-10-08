#!/usr/bin/env python3
"""T4 resident/streaming parity and measured resource checks; requires a GPU."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import os
from pathlib import Path
import subprocess
import time
import traceback
from unittest.mock import patch

import numpy as np

from alignimg._fourier import prepare_stack
from alignimg._profiling import ExecutionProfile, _ACTIVE

if __package__:
    from tools import polar_231_batch_validation as batch
else:
    import polar_231_batch_validation as batch

mirror = batch.mirror


class MemoryTraceProfile(ExecutionProfile):
    """Runner-only trace of existing sample points; no extra CUDA queries."""

    def __init__(self, *, timing_events=True):
        super().__init__()
        self.timing_events = timing_events
        self.trace = []
        self.sampling_scope = "profile_attach"

    def sample_memory(self):
        super().sample_memory()
        if self.cp is not None:
            self.trace.append(
                {
                    "sample_index": self.samples - 1,
                    "monotonic_ns": time.perf_counter_ns(),
                    "wall_time_ns": time.time_ns(),
                    "sampling_scope": self.sampling_scope,
                    "active_parent_stack": [frame["name"] for frame in self.stack],
                    "pending_event_pairs": len(self.events),
                    "tracked_pool_live_bytes": self.hook.live_bytes,
                    **self.memory["last"],
                }
            )

    @contextmanager
    def stage(self, name, *, cuda=False):
        previous = self.sampling_scope
        self.sampling_scope = "/".join([f["name"] for f in self.stack] + [name])
        try:
            with super().stage(name, cuda=cuda and self.timing_events):
                yield
        finally:
            self.sampling_scope = previous

    def close(self):
        self.sampling_scope = "profile_close"
        super().close()


def device_context_snapshot():
    """Boundary-only process context, outside measured alignment execution."""
    snapshot = {
        "wall_time_ns": time.time_ns(),
        "monotonic_ns": time.perf_counter_ns(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q", "-x"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        snapshot.update(status="unavailable", error=str(error))
    else:
        snapshot.update(
            status="captured" if result.returncode == 0 else "unavailable",
            returncode=result.returncode,
            xml=result.stdout,
            stderr=result.stderr,
        )
    return snapshot


@contextmanager
def limited_polar_budget(cap_bytes):
    """Runner-only cap on NEW polar allocations; never change public config."""
    from alignimg_gpu import backend
    from alignimg_gpu._polar_memory import plan_polar_storage

    original = backend._polar_memory_plan

    def limited(*args, **kwargs):
        real = original(*args, **kwargs)
        return plan_polar_storage(
            *args,
            free_bytes=real.free_bytes,
            total_bytes=real.total_bytes,
            workflow_budget_bytes=min(real.workflow_remaining_bytes, cap_bytes)
            + real.live_cache_bytes,
            live_cache_bytes=real.live_cache_bytes,
            force_streaming=kwargs.get("force_streaming", False),
            particle_limit=kwargs.get("particle_limit"),
        )

    with patch.object(backend, "_polar_memory_plan", limited):
        yield


def budget_interval(config, count):
    from alignimg_gpu._polar_memory import polar_allocation_inventory

    fixed, item = polar_allocation_inventory(32, 3, 64, 8, 9, 2)
    common, per_particle = sum(fixed.values()), sum(item.values())
    streaming_minimum = common + per_particle + 32 * 32 * 4
    resident_minimum = common + per_particle + count * 32 * 32 * 4
    assert streaming_minimum < resident_minimum
    return streaming_minimum, resident_minimum


def measure_case(
    engine,
    particles,
    refs,
    config,
    priors,
    centers,
    *,
    workspace=None,
    report=None,
    trace_memory=False,
    timing_events=True,
):
    from alignimg_gpu import backend

    cp = backend._cupy()
    if report is None:
        report = {}
    records = []
    report["memory_records"] = records

    def run():
        return backend._gpu_candidate_inference(
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

    # One warm-up, three separate unprofiled repeats; profiling is not timing.
    expected = run()
    mirror.synchronize(engine)
    timings = []
    for _ in range(3):
        started = time.perf_counter()
        actual = run()
        mirror.synchronize(engine)
        timings.append(time.perf_counter() - started)
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)

    profile = (
        MemoryTraceProfile(timing_events=timing_events)
        if trace_memory
        else ExecutionProfile()
    )
    if trace_memory:
        report["memory_trace"] = profile.trace
        report["timing_events_enabled"] = timing_events
    shapes = {}
    original_upload = backend._asdevice

    def upload(value, **kwargs):
        shape = tuple(value.shape)
        key = "x".join(map(str, shape))
        shapes[key] = shapes.get(key, 0) + 1
        return original_upload(value, **kwargs)

    token = _ACTIVE.set(profile)
    try:
        profile.attach_cuda(cp)
        with patch.object(backend, "_asdevice", upload):
            actual = run()
        mirror.synchronize(engine)
    finally:
        try:
            profile.close()
        finally:
            _ACTIVE.reset(token)
            report["profile"] = profile.asdict()
    memory = profile.memory
    device_increase = max(
        0,
        memory["sampled_device_used_bytes_peak"] - memory["entry"]["device_used_bytes"],
    )
    report.update(
        timings_seconds=timings,
        median_seconds=float(np.median(timings)),
        deterministic_repeat_exact=True,
        result_hashes={
            name: mirror._array_hash(value) for name, value in actual.items()
        },
        profile=profile.asdict(),
        profiled_upload_shapes=shapes,
        observed_device_increase_bytes=device_increase,
        resource_note="Pool-hook peak and sampled device increase are measured separately; device samples are lower bounds, not a continuous hard-limit audit. The small budget tests capacity/parity, not an 8 GiB card's throughput.",
    )
    batch.compare_batch_candidates(expected, actual)
    plan, complete = records[-2:]
    assert plan["fits_minimum"]
    assert plan["estimated_peak_bytes"] <= plan["budget_bytes"]
    assert complete["engine"] == engine
    assert complete["full_correlation_map_d2h_bytes"] == 0
    assert profile.counters["d2h_bytes"] == len(particles.spatial) * 11 * 8
    if complete["particle_storage_policy"] == "streaming":
        assert "x".join(map(str, particles.spatial.shape)) not in shapes
        assert complete["maximum_winner_batch_size"] == plan["batch_size"]
    assert memory["tracked_pool_live_bytes_peak"] <= plan["estimated_peak_bytes"], (
        "unaccounted pooled allocation: "
        f"observed={memory['tracked_pool_live_bytes_peak']}, "
        f"estimated_peak={plan['estimated_peak_bytes']}, budget={plan['budget_bytes']}, "
        f"policy={complete['particle_storage_policy']}, batch={plan['batch_size']}"
    )
    assert device_increase <= plan["budget_bytes"], (
        "observed device usage exceeds polar budget: "
        f"observed={device_increase}, budget={plan['budget_bytes']}, "
        f"estimated_peak={plan['estimated_peak_bytes']}, "
        f"pool_peak={memory['tracked_pool_live_bytes_peak']}, "
        f"device_entry={memory['entry']['device_used_bytes']}, "
        f"device_peak={memory['sampled_device_used_bytes_peak']}, "
        f"policy={complete['particle_storage_policy']}, batch={plan['batch_size']}"
    )
    return actual, report


def validate_streaming(
    engine,
    *,
    batch_size=512,
    cap_bytes=None,
    report=None,
    trace_memory=False,
    timing_events=True,
):
    from alignimg_gpu import backend
    from alignimg_gpu._workspace import WorkflowGpuWorkspace

    if report is None:
        report = {}
    images, references, centers, labels, config = batch.batch_fixture()
    config = replace(config, batch_size=batch_size)
    low, high = budget_interval(config, len(images))
    cap = (low + high) // 2 if cap_bytes is None else cap_bytes
    if not low < cap < high:
        raise ValueError(f"polar budget must satisfy {low} < bytes < {high}")
    particles, refs = prepare_stack(images, config), prepare_stack(references, config)
    report.update(
        config=asdict(config),
        particle_count=len(images),
        polar_budget_bytes=cap,
        budget_interval_bytes=[low, high],
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
    frozen = batch.load_frozen_backend()
    for fixed in (False, True):
        report["active_case"] = f"fixed_{fixed}"
        print(f"RUN resident/streaming fixed={fixed}", flush=True)
        priors = (
            np.eye(3, dtype=np.float32)[labels]
            if fixed
            else np.full((len(images), 3), 1 / 3, dtype=np.float32)
        )
        entry = {"resident": {}, "streaming": {}}
        report["cases"][f"fixed_{fixed}"] = entry
        expected, resident = measure_case(
            engine,
            particles,
            refs,
            config,
            priors,
            centers,
            report=entry["resident"],
            trace_memory=trace_memory,
            timing_events=timing_events,
        )
        assert resident["memory_records"][-1]["particle_storage_policy"] == "resident"
        with limited_polar_budget(cap):
            actual, streamed = measure_case(
                engine,
                particles,
                refs,
                config,
                priors,
                centers,
                report=entry["streaming"],
                trace_memory=trace_memory,
                timing_events=timing_events,
            )
        assert streamed["memory_records"][-1]["particle_storage_policy"] == "streaming"
        errors = batch.compare_batch_candidates(expected, actual)
        limit = streamed["memory_records"][-1]["particle_batch_size"]
        t2 = frozen._gpu_polar_hard_candidate_inference_once(
            particles,
            refs,
            config,
            priors,
            0.08,
            None,
            translation_centers=centers,
            engine=engine,
            particle_limit=limit,
        )
        t2_errors = batch.compare_batch_candidates(t2, actual)
        if fixed:
            np.testing.assert_array_equal(actual["reference_index"][:, 0], labels)
        entry.update(cross_policy_errors=errors, t2_same_batch_errors=t2_errors)
        cache_records = []
        workspace = WorkflowGpuWorkspace(backend._cupy(), config, cache_records)
        try:
            workspace.acquire_fourier(
                "update",
                particles.fourier,
                fixed_bytes=0,
                bytes_per_item=1,
                upload=backend._asdevice,
            )
            assert workspace.allocation_budget()["live_cache_bytes"] > 0
            entry["cache_coexistence"] = {}
            with limited_polar_budget(cap):
                cached, cache_case = measure_case(
                    engine,
                    particles,
                    refs,
                    config,
                    priors,
                    centers,
                    workspace=workspace,
                    report=entry["cache_coexistence"],
                    trace_memory=trace_memory,
                    timing_events=timing_events,
                )
            entry["cache_coexistence_errors"] = batch.compare_batch_candidates(
                actual, cached
            )
            assert (
                cache_case["memory_records"][-2]["live_cache_bytes"]
                == particles.fourier.nbytes
            )
        finally:
            workspace.close()
            entry["workspace"] = workspace.summary()
            entry["cache_records"] = cache_records
    report.pop("active_case")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("cuda", "cupy"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--polar-budget-mib", type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = {
        "schema": "alignimg.polar-231-streaming.v1",
        "task": "T4",
        "status": "running",
        "started_utc": mirror.utc_now(),
        "backend": args.backend,
        "source": mirror.source_manifest(),
        "checks": {},
    }
    started = time.perf_counter()
    try:
        batch.load_frozen_backend()
        report["frozen_t2_backend_sha256"] = batch.FROZEN_SHA256
        report["runtime_identity"] = mirror.runtime_identity(args.backend)
        from alignimg_gpu import backend, memory, _polar_memory, _workspace

        report["loaded_sources"] = {}
        for module in (
            mirror.ai,
            mirror._engine,
            backend,
            memory,
            _polar_memory,
            _workspace,
        ):
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
        validate_streaming(
            args.backend,
            batch_size=args.batch_size,
            cap_bytes=None
            if args.polar_budget_mib is None
            else int(args.polar_budget_mib * 1024**2),
            report=report["checks"],
        )
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
