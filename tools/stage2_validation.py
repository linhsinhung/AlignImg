"""Compare the frozen dev1 GPU engine with the dev2 workflow workspace."""

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
BASELINE_DIGEST = "5e4cc3b501fda6578da4169e547c0a38769264acd124a53df9baa0f6db6f776a"
BASELINE_VERSION = "2.1.0.dev1"
CURRENT_VERSION = "2.1.0.dev3"


def unpack_baseline(directory, target):
    from tools.performance_fixtures import sha256

    manifest = json.loads((directory / "manifest.json").read_text())
    digest = hashlib.sha256(
        json.dumps(manifest["files"], sort_keys=True).encode()
    ).hexdigest()
    if digest != BASELINE_DIGEST or manifest["source_sha256"] != BASELINE_DIGEST:
        raise ValueError("baseline must be the accepted dev1 source-clean snapshot")
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


def worker(args):
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
        raise RuntimeError("frozen dev1 core was not loaded")
    import alignimg_gpu

    if not Path(alignimg_gpu.__file__).is_relative_to(args.worker_source):
        raise RuntimeError("frozen dev1 GPU backend was not loaded")
    alignimg_gpu.__path__.append(str(args.native_directory))
    from tools import performance_validation as perf

    names = perf.SMOKE_CASES if args.suite == "smoke" else perf.REAL_CASES
    names = args.only.split(",") if args.only else names
    report = {
        "version": ai.__version__,
        "source_sha256": BASELINE_DIGEST,
        "environment": perf.environment_info(args.backend, args.device),
        "native_policy": "shared installed binary; kernel source verified unchanged",
        "cases": {},
    }
    for name in names:
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


def validate_workspace_contract(name, baseline, current):
    workspace = current.get("gpu_workspace")
    if not workspace or not workspace.get("closed"):
        raise AssertionError(f"{name}: workflow GPU workspace was not released")
    roles = workspace.get("roles", {})
    iterations = int(current["config"]["max_iterations"])
    expected_requests = {
        "scoring": iterations,
        "update": iterations
        * (3 if current["config"]["halfset_diagnostics"] else 1),
    }
    counters = current["performance"]["counters"]
    for role, requests in expected_requests.items():
        state = roles.get(role)
        if state is None or state.get("policy") != "device":
            raise AssertionError(f"{name}: {role} FFT was not retained on device")
        expected = {
            "requests": requests,
            "uploads": 1,
            "cache_hits": requests - 1,
            "host_streamed_requests": 0,
            "upload_oom_fallbacks": 0,
            "evictions": 0,
        }
        for field, value in expected.items():
            if state.get(field) != value:
                raise AssertionError(
                    f"{name}: {role} {field}={state.get(field)!r}, expected {value}"
                )
        for suffix, value in (
            ("cache_uploads", 1),
            ("cache_hits", requests - 1),
        ):
            if counters.get(f"workspace_{role}_{suffix}", 0) != value:
                raise AssertionError(f"{name}: invalid {role} profiling counter")
    if counters.get("workspace_release_calls") != 1:
        raise AssertionError(f"{name}: workspace release count is not one")
    if workspace["peak_committed_cache_bytes"] > workspace["budget_bytes"]:
        raise AssertionError(f"{name}: retained caches exceed the shared VRAM budget")
    if workspace["released_cache_bytes"] != sum(
        int(roles[role]["cache_bytes"]) for role in expected_requests
    ):
        raise AssertionError(f"{name}: retained cache bytes were not fully released")
    if any(record.get("event") == "oom_retry" for record in current["gpu_memory_plans"]):
        raise AssertionError(f"{name}: unexpected OOM retry in the standard workload")
    requested = current["config"]["batch_size"]
    cache_plans = [
        record
        for record in current["gpu_memory_plans"]
        if record.get("workspace_role") in expected_requests
    ]
    if not cache_plans or any(record.get("batch_size") != requested for record in cache_plans):
        raise AssertionError(f"{name}: standard workload did not retain requested batch size")

    before = baseline["performance"]["counters"]
    if counters.get("h2d_bytes", 0) >= before.get("h2d_bytes", 0):
        raise AssertionError(f"{name}: H2D payload did not decrease")
    if counters.get("h2d_calls", 0) >= before.get("h2d_calls", 0):
        raise AssertionError(f"{name}: H2D call count did not decrease")
    if counters.get("d2h_bytes", 0) != before.get("d2h_bytes", 0):
        raise AssertionError(f"{name}: D2H payload unexpectedly changed")
    return {
        "workspace": workspace,
        "h2d_calls_saved": before["h2d_calls"] - counters["h2d_calls"],
        "h2d_bytes_saved": before["h2d_bytes"] - counters["h2d_bytes"],
        "d2h_bytes_change": counters.get("d2h_bytes", 0)
        - before.get("d2h_bytes", 0),
    }


def compare_reports(baseline_path, current_path, backend):
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
            **validate_workspace_contract(name, baseline, current),
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "cupy"), default="cuda")
    parser.add_argument("--suite", choices=("smoke", "representative"), default="smoke")
    parser.add_argument("--only")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--baseline-source",
        type=Path,
        default=ROOT / "validation-results/performance/stage-1/dev1-delivery",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--native-directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_source:
        worker(args)
        return 0

    sys.path.insert(0, str(ROOT))
    from tools import performance_validation as perf

    cases = perf.SMOKE_CASES if args.suite == "smoke" else perf.REAL_CASES
    if args.only and not set(args.only.split(",")) <= set(cases):
        parser.error("unknown case")
    if args.repeats < 1 or args.batch_size < 1 or not 0 < args.memory_fraction <= 1:
        parser.error("invalid repeats, batch size, or memory fraction")
    outputs = [
        args.output,
        args.output.with_suffix(".baseline.json"),
        args.output.with_suffix(".current.json"),
    ]
    if any(path.exists() for path in outputs):
        parser.error("output exists; choose a new report name")
    report = {
        "stage": 2,
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
        "--batch-size", str(args.batch_size),
        "--memory-fraction", str(args.memory_fraction),
        "--device", str(args.device),
        "--repeats", str(args.repeats),
    ]
    if args.only:
        common += ["--only", args.only]
    try:
        if perf.ai.__version__ != CURRENT_VERSION:
            raise RuntimeError(f"install AlignImg {CURRENT_VERSION} before running stage 2 A/B")
        with tempfile.TemporaryDirectory(prefix="alignimg-stage2-baseline-") as temporary:
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
    print(f"Stage 2: {report['status']}; report: {args.output}", flush=True)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
