"""Compare frozen dev5 M-step updates with shared full/half-set accumulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import traceback


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIGEST = "429acddfb5854ecf917a2b12e256f9df4a6ec877615cc3e0070c5e5be4a4f3e7"
BASELINE_VERSION = "2.1.0.dev5"
CURRENT_VERSION = "2.1.0.dev6"
STAGE_CASES = {
    "smoke": ("global", "adaptive"),
    "representative": ("pose_fixed_mra", "local_refine"),
}


def unpack_baseline(
    directory: Path, target: Path, *, verify_current_native: bool = True
) -> dict:
    from tools.performance_fixtures import sha256

    manifest = json.loads((directory / "manifest.json").read_text())
    digest = hashlib.sha256(
        json.dumps(manifest["files"], sort_keys=True).encode()
    ).hexdigest()
    if digest != BASELINE_DIGEST or manifest["source_sha256"] != BASELINE_DIGEST:
        raise ValueError("baseline must be the accepted dev5 source snapshot")
    archive = directory / "source.tar.gz"
    if sha256(archive) != manifest["archive_sha256"]:
        raise ValueError("baseline archive hash mismatch")
    with tarfile.open(archive) as bundle:
        for name, expected in manifest["files"].items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("invalid baseline path")
            member = bundle.getmember(name)
            if not member.isfile():
                raise ValueError("baseline must contain regular files")
            data = bundle.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"baseline file hash mismatch: {name}")
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)

    if not verify_current_native:
        return manifest

    native = Path("packages/alignimg-gpu/src/alignimg_gpu/native")
    expected_native = {
        name for name in manifest["files"] if name.startswith(str(native) + "/")
    }
    actual_native = {
        str(path.relative_to(ROOT))
        for path in (ROOT / native).rglob("*")
        if path.is_file() and path.suffix in {".cpp", ".cu", ".h", ".hpp"}
    }
    if actual_native != expected_native:
        raise ValueError("native source file set changed")
    for name in manifest["files"]:
        if str(native) in name or name == "packages/alignimg-gpu/CMakeLists.txt":
            old = (target / name).read_bytes().replace(
                BASELINE_VERSION.encode(), b"VERSION"
            )
            new = (ROOT / name).read_bytes().replace(
                CURRENT_VERSION.encode(), b"VERSION"
            )
            if old != new:
                raise ValueError("native code changed; shared-binary A/B is not valid")
    return manifest


def worker(args) -> None:
    args.capture_candidates = True
    sys.path[:0] = [
        str(args.worker_source / "src"),
        str(args.worker_source / "packages/alignimg-gpu/src"),
        str(ROOT),
    ]
    import alignimg as ai

    if ai.__version__ != BASELINE_VERSION or not Path(ai.__file__).is_relative_to(
        args.worker_source
    ):
        raise RuntimeError("frozen dev5 core was not loaded")
    import alignimg_gpu

    if not Path(alignimg_gpu.__file__).is_relative_to(args.worker_source):
        raise RuntimeError("frozen dev5 GPU backend was not loaded")
    alignimg_gpu.__path__.append(str(args.native_directory))
    from tools import performance_validation as perf

    report = {
        "version": ai.__version__,
        "source_sha256": BASELINE_DIGEST,
        "environment": perf.environment_info(args.backend, args.device),
        "native_policy": "shared installed binary; kernel source verified unchanged",
        "cases": {},
    }
    for name in args.only.split(","):
        print(f"BASELINE {name}", flush=True)
        try:
            report["cases"][name] = {"status": "passed", **perf.run_case(name, args)}
        except Exception as error:
            report["cases"][name] = {
                "status": "failed",
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            perf.write_report(args.output, report)
            raise
        perf.write_report(args.output, report)


def _mstep_transform_calls(case: dict) -> int:
    suffix = (
        "/fourier_transform"
        if case["config"]["reference_update"] == "fourier"
        else "/spatial_transform"
    )
    return sum(
        int(value["calls"])
        for name, value in case["performance"]["stages"].items()
        if name.endswith(suffix)
        and any(
            marker in name
            for marker in (
                "/full_reference_update/",
                "/half_a_update/",
                "/half_b_update/",
                "/shared_reference_update/",
            )
        )
    )


def validate_shared_mstep_contract(name: str, baseline: dict, current: dict) -> dict:
    if current.get("halfset_update_policy") != "shared_unnormalized_accumulation":
        raise AssertionError(f"{name}: shared half-set policy was not used")
    workspace = current.get("gpu_workspace")
    if not workspace or not workspace.get("closed"):
        raise AssertionError(f"{name}: workflow GPU workspace was not released")
    if workspace["peak_committed_cache_bytes"] > workspace["budget_bytes"]:
        raise AssertionError(f"{name}: retained caches exceed the shared VRAM budget")
    if any(record.get("event") == "oom_retry" for record in current["gpu_memory_plans"]):
        raise AssertionError(f"{name}: unexpected OOM retry in the standard workload")

    before = baseline["performance"]["counters"]
    after = current["performance"]["counters"]
    transform_before = _mstep_transform_calls(baseline)
    transform_after = _mstep_transform_calls(current)
    if transform_before < 1 or transform_after < 1 or transform_after >= transform_before:
        raise AssertionError(f"{name}: M-step transform calls did not decrease")
    if after.get("shared_mstep_candidate_transforms", 0) < 1:
        raise AssertionError(f"{name}: no shared M-step candidate transforms recorded")
    for counter in ("h2d_calls", "h2d_bytes", "d2h_calls", "d2h_bytes"):
        if after.get(counter, 0) > before.get(counter, 0):
            raise AssertionError(f"{name}: {counter} increased")
    old_requests = before.get("workspace_update_cache_requests", 0)
    new_requests = after.get("workspace_update_cache_requests", 0)
    if old_requests and new_requests >= old_requests:
        raise AssertionError(f"{name}: update-cache requests did not decrease")
    return {
        "mstep_transform_calls_before": transform_before,
        "mstep_transform_calls_after": transform_after,
        "mstep_transform_calls_saved": transform_before - transform_after,
        "workspace_update_cache_requests_saved": old_requests - new_requests,
        "h2d_calls_saved": before.get("h2d_calls", 0) - after.get("h2d_calls", 0),
        "h2d_bytes_saved": before.get("h2d_bytes", 0) - after.get("h2d_bytes", 0),
        "d2h_calls_saved": before.get("d2h_calls", 0) - after.get("d2h_calls", 0),
        "d2h_bytes_saved": before.get("d2h_bytes", 0) - after.get("d2h_bytes", 0),
    }


def compare_reports(baseline_path: Path, current_path: Path, backend: str) -> dict:
    import numpy as np
    from tools.performance_validation import compare_arrays, sha256

    old, new = (json.loads(path.read_text()) for path in (baseline_path, current_path))
    output = {}
    for name, baseline in old["cases"].items():
        current = new["cases"][name]
        if baseline["frozen_inputs_sha256"] != current["frozen_inputs_sha256"]:
            raise AssertionError(f"{name}: baseline and current inputs differ")
        arrays = []
        for report_path, case in ((baseline_path, baseline), (current_path, current)):
            path = report_path.with_name(f"{report_path.stem}.{name}.result.npz")
            if sha256(path) != case["result_sha256"]:
                raise AssertionError(f"{name}: result hash mismatch")
            with np.load(path, allow_pickle=False) as data:
                arrays.append(dict(data))
        ratio = (
            current["unprofiled_median_seconds"]
            / baseline["unprofiled_median_seconds"]
        )
        output[name] = {
            "numeric_errors": compare_arrays(*arrays),
            "current_over_baseline_seconds": ratio,
            "timing_review": (
                "repeat_required"
                if backend == "cuda" and ratio > 1.05
                else "fallback_observation"
                if backend == "cupy"
                else "within_gate"
            ),
            "baseline_counters": baseline["performance"]["counters"],
            "current_counters": current["performance"]["counters"],
            **validate_shared_mstep_contract(name, baseline, current),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "cupy"), default="cuda")
    parser.add_argument("--suite", choices=tuple(STAGE_CASES), default="smoke")
    parser.add_argument("--only")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--baseline-source",
        type=Path,
        default=ROOT / "validation-results/performance/stage-3/dev5-delivery",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--native-directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_source:
        worker(args)
        return 0

    selected = tuple(args.only.split(",")) if args.only else STAGE_CASES[args.suite]
    if not selected or not set(selected) <= set(STAGE_CASES[args.suite]):
        parser.error("unknown case")
    if args.repeats < 1 or args.batch_size < 2 or not 0 < args.memory_fraction <= 1:
        parser.error("invalid repeats, batch size, or memory fraction")
    args.only = ",".join(selected)
    outputs = [
        args.output,
        args.output.with_suffix(".baseline.json"),
        args.output.with_suffix(".current.json"),
    ]
    if any(path.exists() for path in outputs):
        parser.error("output exists; choose a new report name")

    sys.path.insert(0, str(ROOT))
    from tools import performance_validation as perf

    report = {
        "stage": 4,
        "status": "running",
        "backend": args.backend,
        "baseline": str(outputs[1]),
        "current": str(outputs[2]),
        "cases": {},
    }
    perf.write_report(args.output, report)
    common = [
        "--backend", args.backend,
        "--suite", args.suite,
        "--only", args.only,
        "--batch-size", str(args.batch_size),
        "--memory-fraction", str(args.memory_fraction),
        "--device", str(args.device),
        "--repeats", str(args.repeats),
    ]
    try:
        if perf.ai.__version__ != CURRENT_VERSION:
            raise RuntimeError(f"install AlignImg {CURRENT_VERSION} before running stage 4 A/B")
        with tempfile.TemporaryDirectory(prefix="alignimg-stage4-baseline-") as temporary:
            unpack_baseline(args.baseline_source, Path(temporary))
            import alignimg_gpu

            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                *common,
                "--worker-source", temporary,
                "--native-directory", str(Path(alignimg_gpu.__file__).parent),
                "--output", str(outputs[1].resolve()),
            ]
            subprocess.run(command, cwd=ROOT, check=True)
        if perf.main([*common, "--capture-candidates", "--output", str(outputs[2])]):
            raise RuntimeError("current engine validation failed; inspect current JSON")
        report["cases"] = compare_reports(outputs[1], outputs[2], args.backend)
        report["status"] = (
            "repeat_required"
            if any(
                case["timing_review"] == "repeat_required"
                for case in report["cases"].values()
            )
            else "passed"
        )
        report["scientific_review"] = "pending"
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
    perf.write_report(args.output, report)
    print(f"Stage 4: {report['status']}; report: {args.output}", flush=True)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
