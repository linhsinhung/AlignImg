#!/usr/bin/env python3
"""Bounded A/B of frozen 2.3.1 and workflow-scoped polar spatial reuse."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time
import traceback
from types import ModuleType
from unittest.mock import patch

import numpy as np

if __package__:
    from tools import fast_hard_validation as fast
    from tools import performance_validation as perf
    from tools import polar_resource_validation as resources
else:
    import fast_hard_validation as fast
    import performance_validation as perf
    import polar_resource_validation as resources

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = (
    ROOT / "validation-results/maintenance-2.3.1/release-2.3.1/"
    "source-snapshot-00656c02.tar.gz"
)
SNAPSHOT_SHA256 = "08470891fb213c2a851cc17ee596fc3a5f3af66745bae0e17a9ed7998dd9064b"
SOURCE_SHA256 = "00656c02547d8174287c5d839472787e5fc7f3949e37d083f3002308105c6dd6"
GPU_PREFIX = "packages/alignimg-gpu/src/alignimg_gpu/"
FROZEN_MODULES = ("memory", "_polar_memory", "_workspace", "backend")
SMOKE_CASES = ("k1_mirror_off", "k1_mirror_on", "k3_noiseless")
REPRESENTATIVE_CASES = ("fixed_k3", "open_k10", "local3050")


def read_frozen_sources(path=SNAPSHOT):
    """Read pinned bytes without extracting or importing an unverified archive."""
    path = Path(path)
    if fast.sha256(path) != SNAPSHOT_SHA256:
        raise RuntimeError("frozen 2.3.1 snapshot hash mismatch")
    with tarfile.open(path, "r:gz") as archive:
        manifest = json.loads(archive.extractfile("manifest.json").read())
        digest = hashlib.sha256(
            json.dumps(manifest["files"], sort_keys=True).encode()
        ).hexdigest()
        if digest != SOURCE_SHA256 or manifest["source_sha256"] != digest:
            raise RuntimeError("frozen 2.3.1 source manifest mismatch")
        sources = {}
        for name in FROZEN_MODULES:
            relative = GPU_PREFIX + name + ".py"
            source = archive.extractfile(relative).read()
            if hashlib.sha256(source).hexdigest() != manifest["files"][relative]:
                raise RuntimeError(f"frozen module hash mismatch: {name}")
            sources[name] = source
    return sources, manifest


@contextmanager
def frozen_backend(sources, engine):
    """Keep frozen relative imports isolated; share only the installed native binary."""
    import alignimg_gpu

    name = "_alignimg_polar_spatial_frozen"
    names = [name, *(name + "." + part for part in FROZEN_MODULES), name + "._native"]
    if any(key in sys.modules for key in names):
        raise RuntimeError("frozen backend context is already active")
    package = ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
    try:
        native = alignimg_gpu.backend._native_module()
        if native is not None:
            package._native = native
            sys.modules[name + "._native"] = native
        for part in FROZEN_MODULES:
            module = ModuleType(name + "." + part)
            module.__package__ = name
            module.__file__ = f"{SNAPSHOT}!/{GPU_PREFIX}{part}.py"
            sys.modules[module.__name__] = module
            setattr(package, part, module)
            exec(compile(sources[part], module.__file__, "exec"), module.__dict__)
        # The existing repeatability helper imports this package attribute, so
        # its controlled planner patch reaches the frozen backend as well.
        exports = {
            "backend": package.backend,
            **{
                prefix + engine: getattr(package.backend, prefix + engine)
                for prefix in (
                    "run_soft_alignment_",
                    "_final_raw_class_averages_",
                    "transform_images_",
                )
            },
        }
        with patch.multiple(alignimg_gpu, **exports):
            yield package.backend
    finally:
        for key in names:
            sys.modules.pop(key, None)


def runtime_identity(engine, manifest):
    import alignimg_gpu

    source = perf.source_manifest()
    identity = fast.runtime_identity(engine)
    for module, prefix in (
        (fast.ai, "src/alignimg/"),
        (alignimg_gpu, GPU_PREFIX),
    ):
        perf.installed_source_info(module, prefix, source["files"])
    # Only GPU caching is under experiment; frozen math and native sources must
    # still be the release implementation for this to be a one-factor A/B.
    for relative, expected in manifest["files"].items():
        if relative.startswith(("src/alignimg/", GPU_PREFIX + "native/")):
            if source["files"].get(relative) != expected:
                raise RuntimeError(f"non-cache release source changed: {relative}")
    return identity, source


def cache_evidence(entry):
    counters = entry["performance"]["counters"]
    roles = entry["gpu_workspace"]["roles"]
    spatial = roles.get("polar_spatial", {})
    return {
        "spatial_cache": spatial,
        "cache_hits": int(spatial.get("cache_hits", 0)),
        "uploads": int(spatial.get("uploads", 0)),
        "transfers": {
            key: value
            for key, value in counters.items()
            if "h2d" in key or "d2h" in key or "polar_spatial" in key
        },
        "note": "Transfer counters are from the separate profiled run; timing ratios use unprofiled repeats only.",
    }


@contextmanager
def controlled_cached_batch(particle_batch, records):
    from alignimg_gpu import backend

    original = backend._polar_memory_plan

    def planned(*args, **kwargs):
        limit = kwargs.get("particle_limit")
        kwargs["particle_limit"] = (
            min(particle_batch, limit) if limit else particle_batch
        )
        plan = original(*args, **kwargs)
        records.append(plan.asdict())
        assert plan.fits_minimum and plan.batch_size == particle_batch
        assert plan.particle_storage_policy == "cached", (
            "controlled repeat cannot retain the cache under the available budget"
        )
        return plan

    with patch.object(backend, "_polar_memory_plan", planned):
        yield


def ensure_cached_repeatability(entry, name, case, config, engine, output):
    """Control batch variation without disabling the optimization being tested."""
    diagnostic = resources.repeat_execution_diagnostics(entry["execution_records"])
    entry["repeat_execution_diagnostics"] = diagnostic
    if diagnostic["same_recorded_allocation_schedule"] is True:
        resources.require_exact_repeats(entry)
        return
    assert diagnostic["same_recorded_allocation_schedule"] is not None
    repeats = [r for r in entry["execution_records"] if r["phase"] == "unprofiled"]
    seen = {}
    for record, schedule in zip(repeats, diagnostic["unprofiled_allocation_schedules"]):
        key = json.dumps(schedule, sort_keys=True)
        value = record["result_hashes"]["combined"]
        assert seen.setdefault(key, value) == value, (
            "same-schedule repeats are not exact"
        )
    plans = [
        p
        for r in repeats
        for p in r["gpu_memory_plans"]
        if p.get("stage") == "polar_hard_candidate_inference" and "event" not in p
    ]
    assert plans and all(p["particle_storage_policy"] == "cached" for p in plans), (
        "varied cached/uncached schedules cannot establish cached determinism"
    )
    particle_batch = min(p["batch_size"] for p in plans)
    pending = {"planning_records": [], "execution_records": []}
    entry["controlled_repeatability"] = pending
    print(f"CHECK {name}: cached actual batch={particle_batch}, 3 repeats", flush=True)
    with controlled_cached_batch(particle_batch, pending["planning_records"]):
        controlled = fast.run_case(
            name,
            case,
            config=config,
            backend=engine,
            measured_repeats=3,
            profile_execution=False,
            output=output,
            variant="cached-same-batch",
            result_label="same-batch",
            execution_records=pending["execution_records"],
        )
    controlled.update(
        planning_records=pending["planning_records"],
        target_particle_batch=particle_batch,
    )
    entry["controlled_repeatability"] = controlled
    resources.require_exact_repeats(controlled)
    with np.load(entry["result"], allow_pickle=False) as normal:
        with np.load(controlled["result"], allow_pickle=False) as bounded:
            controlled["normal_result_parity"] = perf.compare_arrays(
                dict(normal), dict(bounded)
            )
    controlled["normal_polar_parity"] = fast.compare_polar_backend_parity(
        Path(entry["result"]), Path(controlled["result"]), case, config
    )
    assert controlled["normal_polar_parity"]["within_tolerance"]
    controlled["passed"] = True


def compare_case(baseline, cached, case, config):
    with np.load(baseline["result"], allow_pickle=False) as old:
        with np.load(cached["result"], allow_pickle=False) as new:
            errors = perf.compare_arrays(dict(old), dict(new))
    polar = fast.compare_polar_backend_parity(
        Path(baseline["result"]), Path(cached["result"]), case, config
    )
    assert polar["within_tolerance"], "polar candidate parity failed"
    ratio = cached["unprofiled_median_seconds"] / baseline["unprofiled_median_seconds"]
    evidence = cache_evidence(cached)
    return {
        "array_errors": errors,
        "polar_parity": polar,
        "cache_evidence": evidence,
        "cached_over_frozen_wall_ratio": ratio,
        "within_5_percent_regression_limit": ratio <= 1.05,
        "material_benefit_at_least_5_percent": ratio <= 0.95,
        "informative_cache_hit": evidence["cache_hits"] > 0,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "representative"), required=True)
    parser.add_argument("--backend", choices=("cuda", "cupy"), required=True)
    parser.add_argument("--only", choices=SMOKE_CASES + REPRESENTATIVE_CASES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    allowed = SMOKE_CASES if args.suite == "smoke" else REPRESENTATIVE_CASES
    if args.only and args.only not in allowed:
        parser.error("--only case is not in the selected suite")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists() or any(
        args.output.parent.glob(args.output.stem + ".*.result.npz")
    ):
        raise FileExistsError(f"refusing to overwrite results for {args.output}")
    report = {
        "schema": "alignimg.polar-spatial-cache-ab.v1",
        "status": "running",
        "started_utc": fast.utc_now(),
        "parameters": vars(args),
        "cases": {},
    }
    fast.write_report(args.output, report)
    started = time.perf_counter()
    try:
        sources, manifest = read_frozen_sources()
        report["frozen_baseline"] = {
            "snapshot": str(SNAPSHOT),
            "snapshot_sha256": SNAPSHOT_SHA256,
            "source_sha256": SOURCE_SHA256,
            "module_sha256": {
                key: hashlib.sha256(value).hexdigest() for key, value in sources.items()
            },
        }
        report["runtime"], report["source"] = runtime_identity(args.backend, manifest)
        if args.suite == "representative":
            cases, old, _, report["input_hashes"] = resources.load_representative(ROOT)
            config = fast.ai.AlignmentConfig(**old["config"])
            names = REPRESENTATIVE_CASES
        else:
            cases = fast.synthetic_cases()
            config = replace(
                fast.ai.AlignmentConfig.preset("fast3"),
                angle_samples=64,
                translation_range=2.0,
                batch_size=512,
                memory_fraction=0.8,
            )
            names = SMOKE_CASES
        names = (args.only,) if args.only else names
        for name in names:
            entry = report["cases"][name] = {}
            case = cases[name]
            local = replace(config, mirror_search=bool(case["mirror_search"]))
            for variant in ("frozen", "cached"):
                records = []
                entry[variant] = {"execution_records": records}
                fast.write_report(args.output, report)
                print(
                    f"RUN {name} {variant}: warm-up + 3 unprofiled + 1 profiled",
                    flush=True,
                )
                context = (
                    frozen_backend(sources, args.backend)
                    if variant == "frozen"
                    else nullcontext()
                )
                with context:
                    value = fast.run_case(
                        name,
                        case,
                        config=local,
                        backend=args.backend,
                        measured_repeats=3,
                        profile_execution=True,
                        output=args.output,
                        variant="spatial-cache-" + variant,
                        result_label=variant,
                        execution_records=records,
                    )
                    entry[variant] = value
                    # Distinct controlled artifact stems avoid collisions when
                    # both variants need the existing same-batch repeat probe.
                    controlled_output = args.output.with_name(
                        args.output.stem + "." + variant + ".json"
                    )
                    repeat_check = (
                        resources.ensure_repeatability
                        if variant == "frozen"
                        else ensure_cached_repeatability
                    )
                    repeat_check(
                        value, name, case, local, args.backend, controlled_output
                    )
                    value["resource_checks"] = resources.resource_checks(
                        value, args.backend
                    )
                fast.write_report(args.output, report)
            entry["comparison"] = compare_case(
                entry["frozen"], entry["cached"], case, local
            )
            fast.write_report(args.output, report)
        comparisons = [value["comparison"] for value in report["cases"].values()]
        assert all(item["within_5_percent_regression_limit"] for item in comparisons), (
            "end-to-end regression exceeds 5%"
        )
        informative = any(item["informative_cache_hit"] for item in comparisons)
        report["status"] = "passed" if informative else "uninformative"
        report["material_benefit"] = args.suite == "representative" and any(
            item["informative_cache_hit"]
            and item["material_benefit_at_least_5_percent"]
            for item in comparisons
        )
        report["release_decision"] = (
            "experiment only; requires representative benefit and all regression gates"
        )
    except Exception as error:
        report.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
    report["wall_seconds"] = time.perf_counter() - started
    fast.write_report(args.output, report)
    print(f"{report['status'].upper()}: {args.output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
