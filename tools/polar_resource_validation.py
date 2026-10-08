#!/usr/bin/env python3
"""T6 polar conformance, resources and frozen representative integration gates."""

from __future__ import annotations

import argparse
from contextlib import chdir, contextmanager
from dataclasses import replace
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import numpy as np

if __package__:
    from tools import fast_hard_validation as fast
    from tools import performance_validation as perf
    from tools import polar_231_baseline as baseline
    from tools import polar_231_batch_validation as batch
    from tools import polar_231_streaming_validation as streaming
else:
    import fast_hard_validation as fast
    import performance_validation as perf
    import polar_231_baseline as baseline
    import polar_231_batch_validation as batch
    import polar_231_streaming_validation as streaming

mirror = batch.mirror
BASELINE_RELATIVE = "validation-results/maintenance-2.3.1/baseline/fast3-n3050-2.3"
BASELINE_REPORT_SHA256 = (
    "6c315d3f123574d3002bbfcd125e2cd50c446f846bfc326d44c7bbe137374d1d"
)
BASELINE_RESULT_SHA256 = (
    "08e9fdd4e540ef91aa5fcb29dd6836a6625eb3f33d65a00f138e224f15000eb0"
)
CORRECTED_BASELINE = (
    "validation-results/maintenance-2.3.1/baseline/corrected-current-dev12"
)
CORRECTED_MANIFEST_SHA256 = (
    "76628a311294acf6c2925bf235c2991cb7bc50f085ef448dab8a8bcd6221385b"
)
CORRECTED_RESULT_SHA256 = (
    "1315b6e801ab4872597f677026e625087f090ece1a86ca52ff39f4df0e3e7138"
)


def corrected_baseline(root, config, hashes, fields):
    manifest_path = root / (CORRECTED_BASELINE + ".json")
    result_path = root / (CORRECTED_BASELINE + ".npz")
    for path, expected in (
        (manifest_path, CORRECTED_MANIFEST_SHA256),
        (result_path, CORRECTED_RESULT_SHA256),
    ):
        if mirror.sha256(path) != expected:
            raise RuntimeError(f"corrected baseline hash mismatch: {path}")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["config"] == config and manifest["inputs"] == hashes
    assert manifest["comparison_fields"] == list(fields)
    assert manifest["result_npz_sha256"] == CORRECTED_RESULT_SHA256
    approval_path = root / manifest["acceptance_path"]
    if mirror.sha256(approval_path) != manifest["acceptance_sha256"]:
        raise RuntimeError("baseline acceptance hash mismatch")
    approval = json.loads(approval_path.read_text())
    assert (
        approval["compatibility_exception"]["id"]
        == "zero-shift-noop-local3050-legacy-baseline"
    )
    with np.load(result_path, allow_pickle=False) as saved:
        values = {name: saved[name].copy() for name in fields}
    return values, manifest


def numeric_baseline_checks(old_values, current, corrected_values, report):
    """A narrow accepted legacy difference never excuses a new-baseline failure."""
    try:
        errors = perf.compare_arrays(old_values, current)
        report["legacy_baseline_comparison"] = {
            "passed": True,
            "accepted_exception": False,
        }
    except AssertionError as error:
        report["legacy_baseline_comparison"] = {
            "passed": False,
            "error": str(error),
            "accepted_exception": corrected_values is not None,
        }
        if corrected_values is None:
            raise
    if corrected_values is None:
        report["baseline_errors"] = errors
    else:
        report["corrected_baseline_errors"] = perf.compare_arrays(
            corrected_values, current
        )


def runtime_info(engine):
    frozen = batch.load_frozen_backend()
    identity = mirror.runtime_identity(engine)
    import alignimg_gpu

    manifest = mirror.source_manifest()
    return {
        "identity": identity,
        "frozen_t2_backend_sha256": mirror.sha256(Path(frozen.__file__)),
        "loaded_sources": {
            "core": perf.installed_source_info(
                mirror.ai, "src/alignimg/", manifest["files"]
            ),
            "gpu": perf.installed_source_info(
                alignimg_gpu,
                "packages/alignimg-gpu/src/alignimg_gpu/",
                manifest["files"],
            ),
        },
    }


def preflight(root):
    specs = {
        **baseline.FROZEN_INPUTS,
        "local_particles": (
            "data/local/test_align.mrcs",
            baseline.EXPECTED_PARTICLES_SHA256,
        ),
        "local_reference": (
            "data/local/mu_aligned_mean.mrc",
            baseline.EXPECTED_REFERENCE_SHA256,
        ),
        "baseline_report": (BASELINE_RELATIVE + ".json", BASELINE_REPORT_SHA256),
        "baseline_result": (BASELINE_RELATIVE + ".npz", BASELINE_RESULT_SHA256),
    }
    checked = {}
    for name, (relative, expected) in specs.items():
        actual = mirror.sha256(root / relative)
        if actual != expected:
            raise RuntimeError(f"frozen {name} hash mismatch: {actual} != {expected}")
        checked[name] = {"path": relative, "sha256": actual}
    old = json.loads((root / (BASELINE_RELATIVE + ".json")).read_text())
    assert old["status"] == "passed" and old["deterministic_repeats"]
    with np.load(root / (BASELINE_RELATIVE + ".npz"), allow_pickle=False) as saved:
        values = {name: saved[name].copy() for name in saved.files}
    return old, values, checked


def load_representative(root):
    old, values, hashes = preflight(root)
    # Existing loaders resolve the frozen manifest's repository-relative paths.
    with chdir(root), patch.object(fast, "ROOT", root):
        cases, _ = fast.homogeneous_cases()
        frozen_path = root / baseline.FROZEN_INPUTS["mra1000"][0]
        cases.update(fast.freeze_inputs("mra1000", frozen_path)["cases"])
    images, _ = baseline.load_stack(root / "data/local/test_align.mrcs")
    references, _ = baseline.load_stack(root / "data/local/mu_aligned_mean.mrc")
    assert images.shape == (3050, 100, 100) and references.shape == (1, 100, 100)
    cases["local3050"] = fast._case_values(images, references)
    assert cases["fixed_k3"]["images"].shape == (384, 128, 128)
    assert cases["open_k10"]["images"].shape == (1000, 128, 128)
    assert cases["open_k10"]["references"].shape == (10, 128, 128)
    return cases, old, values, hashes


def planning_checks():
    from alignimg_gpu._polar_memory import plan_polar_storage

    gib = 1024**3
    config = replace(
        mirror.ai.AlignmentConfig.preset("fast3"), batch_size=512, memory_fraction=0.8
    )
    plans = {
        str(n): plan_polar_storage(
            config, 128, n, 10, 256, 52, 81, 1, free_bytes=8 * gib, total_bytes=8 * gib
        ).asdict()
        for n in (1000, 105000, 140000)
    }
    for plan in plans.values():
        assert plan["fits_minimum"] and plan["particle_storage_policy"] == "streaming"
        assert plan["estimated_peak_bytes"] <= plan["budget_bytes"]
        assert "resident_particles_fp32" not in plan["fixed_components"]
        assert plan["fixed_components"] == plans["1000"]["fixed_components"]
        assert plan["item_components"] == plans["1000"]["item_components"]
        assert plan["batch_size"] == plans["1000"]["batch_size"]
    return plans


def validate_storage_parity(engine, *, batch_size, report):
    from alignimg_gpu import backend

    images, references, centers, labels, config = batch.batch_fixture(17)
    config = replace(config, batch_size=batch_size)
    particles = batch.prepare_stack(images, config)
    refs = batch.prepare_stack(references, config)
    low, high = streaming.budget_interval(config, len(images))
    for fixed in (False, True):
        entry = {"plans": []}
        report[f"fixed_{fixed}"] = entry
        priors = (
            np.eye(3, dtype=np.float32)[labels]
            if fixed
            else np.full((len(images), 3), 1 / 3, np.float32)
        )
        args = (particles, refs, config, priors, 0.08, None)
        kwargs = dict(
            translation_centers=centers, engine=engine, memory_records=entry["plans"]
        )
        resident = backend._gpu_candidate_inference(*args, **kwargs)
        assert entry["plans"][-1]["particle_storage_policy"] == "resident"
        with streaming.limited_polar_budget((low + high) // 2):
            streamed = backend._gpu_candidate_inference(*args, **kwargs)
        assert entry["plans"][-1]["particle_storage_policy"] == "streaming"
        assert entry["plans"][-1]["engine"] == engine
        entry["errors"] = batch.compare_batch_candidates(resident, streamed)
        for plan in entry["plans"]:
            if "event" not in plan:
                assert (
                    plan["fits_minimum"]
                    and plan["estimated_peak_bytes"] <= plan["budget_bytes"]
                )
            elif plan["event"] == "complete":
                assert plan["full_correlation_map_d2h_bytes"] == 0


def repeat_execution_diagnostics(records):
    """Describe comparable allocation schedules; this does not waive a gate."""
    fields = {
        "stage",
        "event",
        "workspace_role",
        "policy",
        "particle_storage_policy",
        "batch_size",
        "particle_batch_size",
        "actual_particle_batch",
        "maximum_combo_batch_size",
        "maximum_translation_centers",
        "maximum_winner_batch_size",
        "angle_samples",
        "radial_bins",
        "cache_bytes",
        "oom_retry_count",
        "action",
    }
    schedules = []
    for record in records:
        if record["phase"] != "unprofiled":
            continue
        schedule = [
            {key: value for key, value in plan.items() if key in fields}
            for plan in (record.get("gpu_memory_plans") or [])
        ]
        if record.get("class_average_batch_size") is not None:
            schedule.append(
                {
                    "stage": "final_raw_average",
                    "batch_size": record["class_average_batch_size"],
                    "backend": record.get("class_average_transform_backend"),
                    "plan": {
                        key: value
                        for key, value in (
                            record.get("class_average_gpu_memory_plan") or {}
                        ).items()
                        if key in fields
                    },
                    "events": [
                        {key: value for key, value in event.items() if key in fields}
                        for event in (
                            record.get("class_average_gpu_memory_events") or []
                        )
                    ],
                }
            )
        schedules.append(schedule)
    same = (
        all(schedule == schedules[0] for schedule in schedules[1:])
        if len(schedules) >= 2 and all(schedules)
        else None
    )
    return {
        "same_recorded_allocation_schedule": same,
        "unprofiled_allocation_schedules": schedules,
        "note": "Actual batch/cache schedules only; budgets and timings are excluded. Different or missing schedules do not establish same-schedule determinism.",
    }


def resource_checks(entry, engine):
    execution = entry.get("execution_records", [])
    repeated = repeat_execution_diagnostics(execution)
    entry.setdefault("repeat_execution_diagnostics", {}).update(repeated)
    metadata = entry["backend_metadata"]
    assert metadata["backend"] == engine, "GPU fallback is not allowed"
    kernel = "native_cuda" if engine == "cuda" else "cupy"
    assert metadata["polar_sampler_backend"] == metadata["polar_peak_backend"] == kernel
    assert metadata["polar_full_correlation_map_d2h"] is False
    controlled = entry.get("controlled_repeatability")
    if controlled is not None:
        assert controlled.get("passed") is True, (
            "controlled repeatability is unverified"
        )
    require_exact_repeats(controlled if controlled is not None else entry)
    workspace = entry["gpu_workspace"]
    assert workspace["closed"]
    profiled = next(record for record in execution if record["phase"] == "profiled")
    profile_workspace = profiled["gpu_workspace"]
    assert profile_workspace["closed"]
    all_execution = execution + (controlled["execution_records"] if controlled else [])
    for record in all_execution:
        assert record["gpu_workspace"]["closed"]
    plans = [
        p
        for p in entry["gpu_memory_plans"]
        + [
            plan
            for record in all_execution
            for plan in (
                record["gpu_memory_plans"]
                + (
                    [record["class_average_gpu_memory_plan"]]
                    if record.get("class_average_gpu_memory_plan")
                    else []
                )
            )
        ]
        if {"budget_bytes", "fixed_bytes", "bytes_per_item", "batch_size"}.issubset(p)
    ]
    assert plans, "no allocation plans recorded"
    for p in plans:
        assert p["fits_minimum"]
        peak = p.get(
            "estimated_peak_bytes",
            p["fixed_bytes"] + p["batch_size"] * p["bytes_per_item"],
        )
        assert peak <= p["budget_bytes"], "executed plan exceeds budget"
    profile = entry["performance"]
    assert (
        profile["counters"].get(
            ("native" if engine == "cuda" else "cupy") + "_polar_sampling_calls", 0
        )
        > 0
    )
    memory = profile["gpu_memory"]
    pool_peak = memory["tracked_pool_live_bytes_peak"]
    device_increase = max(
        0,
        memory["sampled_device_used_bytes_peak"] - memory["entry"]["device_used_bytes"],
    )
    budget = profile_workspace["budget_bytes"]
    assert pool_peak <= budget, (
        f"measured pool peak exceeds workflow budget: observed={pool_peak}, budget={budget}"
    )
    assert device_increase <= budget, (
        f"observed device usage exceeds workflow budget: observed={device_increase}, "
        f"budget={budget}, device_entry={memory['entry']['device_used_bytes']}, "
        f"device_peak={memory['sampled_device_used_bytes_peak']}"
    )
    return {
        "plan_count": len(plans),
        "budget_bytes": budget,
        "budget_execution_phase": "profiled",
        "budget_execution_index": profiled["index"],
        "tracked_pool_live_bytes_peak": pool_peak,
        "sampled_device_increase_bytes": device_increase,
        "note": "Pool-hook peak excludes native/cuFFT outside the pool. Device samples are lower bounds, not continuous peak proof.",
    }


def require_exact_repeats(entry):
    diagnostic = repeat_execution_diagnostics(entry["execution_records"])
    entry["repeat_execution_diagnostics"] = diagnostic
    assert entry["deterministic_exact_match"], "same-backend repeats are not exact"
    assert len(diagnostic["unprofiled_allocation_schedules"]) == 3, (
        "three repeats required"
    )
    assert diagnostic["same_recorded_allocation_schedule"] is True, (
        "same allocation schedule has not been established"
    )


@contextmanager
def controlled_polar_batch(particle_batch, records):
    """Validation-only cap: use the real planner and refuse any smaller batch."""
    from alignimg_gpu import backend

    original = backend._polar_memory_plan

    def planned(*args, **kwargs):
        kwargs["force_streaming"] = True
        limit = kwargs.get("particle_limit")
        kwargs["particle_limit"] = (
            min(particle_batch, limit) if limit is not None else particle_batch
        )
        plan = original(*args, **kwargs)
        records.append(plan.asdict())
        assert plan.fits_minimum and plan.batch_size == particle_batch, (
            f"controlled polar batch cannot fit: requested={particle_batch}, "
            f"actual={plan.batch_size}, budget={plan.budget_bytes}"
        )
        assert plan.particle_storage_policy == "streaming"
        return plan

    with patch.object(backend, "_polar_memory_plan", planned):
        yield


def ensure_repeatability(entry, name, case, config, engine, output):
    diagnostic = repeat_execution_diagnostics(entry["execution_records"])
    entry["repeat_execution_diagnostics"] = diagnostic
    same = diagnostic["same_recorded_allocation_schedule"]
    assert same is not None, "repeat allocation schedules were not recorded"
    if same:
        require_exact_repeats(entry)
        return

    repeats = [r for r in entry["execution_records"] if r["phase"] == "unprofiled"]
    grouped = {}
    diagnostic["same_schedule_hashes_exact"] = True
    for record, schedule in zip(repeats, diagnostic["unprofiled_allocation_schedules"]):
        key = json.dumps(schedule, sort_keys=True)
        previous = grouped.setdefault(key, record)
        if previous["result_hashes"]["combined"] != record["result_hashes"]["combined"]:
            diagnostic["same_schedule_hashes_exact"] = False
            diagnostic["mismatched_repeat_indices"] = [
                previous["index"],
                record["index"],
            ]
            raise AssertionError("same-schedule repeats are not exact")

    observed = [
        plan["batch_size"]
        for record in entry["execution_records"]
        for plan in record["gpu_memory_plans"]
        if plan.get("stage") == "polar_hard_candidate_inference" and "event" not in plan
    ]
    assert observed, "no observed feasible polar batch"
    particle_batch = min(observed)
    pending = {
        "target_particle_batch": particle_batch,
        "planning_records": [],
        "execution_records": [],
    }
    entry["controlled_repeatability"] = pending
    print(
        f"CHECK {name}: fixed actual polar batch={particle_batch}, streaming, 3 repeats",
        flush=True,
    )
    with controlled_polar_batch(particle_batch, pending["planning_records"]):
        controlled = fast.run_case(
            name,
            case,
            config=config,
            backend=engine,
            measured_repeats=3,
            profile_execution=False,
            output=output,
            variant="maintenance-2.3.1-same-batch",
            result_label="same-batch",
            execution_records=pending["execution_records"],
        )
    controlled.update(
        target_particle_batch=particle_batch,
        planning_records=pending["planning_records"],
    )
    entry["controlled_repeatability"] = controlled
    require_exact_repeats(controlled)
    for record in controlled["execution_records"]:
        plans = [
            p
            for p in record["gpu_memory_plans"]
            if p.get("stage") == "polar_hard_candidate_inference" and "event" not in p
        ]
        assert plans and all(
            p["batch_size"] == particle_batch
            and p["particle_storage_policy"] == "streaming"
            for p in plans
        )
    with np.load(entry["result"], allow_pickle=False) as saved:
        normal_values = {key: saved[key] for key in saved.files}
    with np.load(controlled["result"], allow_pickle=False) as saved:
        controlled_values = {key: saved[key] for key in saved.files}
    controlled["normal_result_parity"] = perf.compare_arrays(
        normal_values, controlled_values
    )
    controlled["normal_polar_parity"] = fast.compare_polar_backend_parity(
        Path(entry["result"]), Path(controlled["result"]), case, config
    )
    assert controlled["normal_polar_parity"]["within_tolerance"], (
        "controlled polar parity failed"
    )
    controlled["passed"] = True


def speed_gate(old, median, engine, environment):
    previous = old["median_seconds"]
    assert np.isfinite(median) and median > 0
    same = (
        environment["hostname"] == old["environment"]["hostname"]
        and environment["gpu"]["name"] == old["environment"]["gpu"]
    )
    return {
        "applicable": engine == "cuda",
        "same_server": same,
        "baseline_seconds": previous,
        "current_seconds": median,
        "limit_seconds": previous * 1.05,
        "current_over_baseline": median / previous,
        "passed": same and median <= previous * 1.05 if engine == "cuda" else None,
        "note": "Frozen baseline uses CUDA. CuPy timings are recorded, not treated as CUDA performance parity.",
    }


def validate_representative(
    engine, output, environment, *, report, numeric_baseline="original"
):
    cases, old, old_values, hashes = load_representative(mirror.ROOT)
    report.update(inputs=hashes, baseline=old, cases={})
    config = mirror.ai.AlignmentConfig(**old["config"])
    corrected_values = None
    if numeric_baseline == "corrected":
        corrected_values, report["corrected_transition_baseline"] = corrected_baseline(
            mirror.ROOT, old["config"], hashes, old_values
        )
    assert (
        config.batch_size == 512
        and config.memory_fraction == 0.8
        and not config.mirror_search
    )
    for name in (
        "k1_class_0",
        "k1_class_1",
        "k1_class_2",
        "fixed_k3",
        "open_k10",
        "local3050",
    ):
        report["active_case"] = name
        print(f"RUN {name}: warm-up + 3 unprofiled + 1 profiled", flush=True)
        execution_records = []
        report["cases"][name] = {"execution_records": execution_records}
        entry = fast.run_case(
            name,
            cases[name],
            config=config,
            backend=engine,
            measured_repeats=3,
            profile_execution=True,
            output=output,
            variant="maintenance-2.3.1-fast3",
            result_label="t6",
            execution_records=execution_records,
        )
        report["cases"][name] = entry
        ensure_repeatability(entry, name, cases[name], config, engine, output)
        entry["resource_checks"] = resource_checks(entry, engine)
    fixed = {
        name: {"fast_hard": report["cases"][name]}
        for name in cases
        if name.startswith("k1_class_") or name == "fixed_k3"
    }
    report["fixed_mra_vs_k1"] = fast.compare_homogeneous_fixed_k3_to_k1(fixed, cases)
    comparison = report["fixed_mra_vs_k1"]
    assert (
        comparison["pose_within_tolerance"] and comparison["assignment_contract_exact"]
    )
    assert all(c["reference_correlation"] >= 0.9999 for c in comparison["per_class"])
    final = report["cases"]["local3050"]
    with np.load(final["result"], allow_pickle=False) as saved:
        current = {name: saved[name].copy() for name in old_values}
    report["speed_gate"] = speed_gate(
        old, final["unprofiled_median_seconds"], engine, environment
    )
    numeric_baseline_checks(old_values, current, corrected_values, report)
    if engine == "cuda":
        assert report["speed_gate"]["passed"], (
            "3050 CUDA same-server performance gate failed; release must wait"
        )
    report.pop("active_case")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", required=True, choices=("conformance", "resources", "representative")
    )
    parser.add_argument("--backend", required=True, choices=("cuda", "cupy"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--polar-budget-mib", type=float)
    parser.add_argument("--trace-memory", action="store_true")
    parser.add_argument("--no-timing-events", action="store_true")
    parser.add_argument(
        "--numeric-baseline", choices=("original", "corrected"), default="original"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.numeric_baseline == "corrected" and args.suite != "representative":
        parser.error("corrected numeric baseline is only valid for representative")
    if args.trace_memory and args.suite != "resources":
        parser.error("trace-memory is only valid for resources")
    if args.no_timing_events and not args.trace_memory:
        parser.error("no-timing-events requires trace-memory")
    if not 1 <= args.batch_size <= 512:
        parser.error("batch-size must be 1..512")
    if args.suite == "representative" and args.batch_size != 512:
        parser.error(
            "representative uses frozen batch-size=512 for the 2.3.0 comparison"
        )
    if args.polar_budget_mib is not None:
        if (
            args.suite != "resources"
            or not np.isfinite(args.polar_budget_mib)
            or args.polar_budget_mib <= 0
        ):
            parser.error(
                "polar-budget-mib must be positive/finite and is only valid for resources"
            )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = {
        "schema": "alignimg.polar-resource-validation.v1",
        "task": "T6",
        "status": "running",
        "started_utc": mirror.utc_now(),
        "suite": args.suite,
        "backend": args.backend,
        "parameters": vars(args),
        "diagnostic_only": args.trace_memory,
        "source": mirror.source_manifest(),
        "checks": {},
    }
    mirror.write_report(args.output, report)
    started = time.perf_counter()
    try:
        if args.trace_memory:
            report["memory_diagnostics"] = {
                "timing_events_enabled": not args.no_timing_events,
                "before": streaming.device_context_snapshot(),
                "note": (
                    "Trace copies existing sample points without extra CUDA queries. "
                    "Pending event pairs are not event memory or all live handles. "
                    "Disabling timing events also removes event-flush synchronization; "
                    "this is diagnostic, not a replacement acceptance/performance run. "
                    "Device memory is global; boundary process snapshots cannot exclude "
                    "transient external usage. Match physical GPUs by XML UUID/PCI ID "
                    "and CUDA_VISIBLE_DEVICES, not the display index alone."
                ),
            }
        report["runtime"] = runtime_info(args.backend)
        report["environment"] = mirror.environment_info(args.backend, 0)
        checks = report["checks"]
        if args.suite == "conformance":
            checks["sampler"] = mirror.validate_sampler(args.backend)
            for enabled in (False, True):
                checks[f"candidate_mirror_{enabled}"] = mirror.validate_candidate(
                    args.backend, enabled
                )
                checks[f"fast3_mirror_{enabled}"] = mirror.validate_fast3(
                    args.backend, enabled
                )
            checks["batches"] = {}
            batch.validate_batches(
                args.backend, batch.load_frozen_backend(), report=checks["batches"]
            )
            checks["fixed_synthetic_mra_vs_k1"] = batch.validate_fixed_workflow(
                args.backend, batch.load_frozen_backend()
            )
            checks["storage_parity"] = {}
            validate_storage_parity(
                args.backend,
                batch_size=args.batch_size,
                report=checks["storage_parity"],
            )
        elif args.suite == "resources":
            checks["size_only_8gib_plans"] = planning_checks()
            checks["storage"] = {}
            cap = (
                None
                if args.polar_budget_mib is None
                else int(args.polar_budget_mib * 1024**2)
            )
            streaming.validate_streaming(
                args.backend,
                batch_size=args.batch_size,
                cap_bytes=cap,
                report=checks["storage"],
                trace_memory=args.trace_memory,
                timing_events=not args.no_timing_events,
            )
        else:
            validate_representative(
                args.backend,
                args.output,
                report["environment"],
                report=checks,
                numeric_baseline=args.numeric_baseline,
            )
        report["status"] = "passed"
    except Exception as error:
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        if hasattr(error, "polar_memory_records"):
            report["failure_memory_records"] = error.polar_memory_records
    finally:
        if args.trace_memory:
            report["memory_diagnostics"]["after"] = streaming.device_context_snapshot()
    report["wall_seconds"] = time.perf_counter() - started
    mirror.write_report(args.output, report)
    label = "DIAGNOSTIC " if args.trace_memory else ""
    print(f"{label}{report['status'].upper()}: {args.output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
