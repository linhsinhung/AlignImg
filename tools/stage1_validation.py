"""Run the frozen dev0 Python engine and dev1 sequentially on identical cases."""

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
BASELINE_DIGEST = "175176b13c4a19aaf39c72fe88e11eaf893983302f304cf5d71f639f3cdaad62"


def unpack_baseline(directory, target):
    from tools.performance_fixtures import sha256

    manifest = json.loads((directory / "manifest.json").read_text())
    digest = hashlib.sha256(
        json.dumps(manifest["files"], sort_keys=True).encode()
    ).hexdigest()
    if digest != BASELINE_DIGEST or manifest["source_sha256"] != BASELINE_DIGEST:
        raise ValueError("baseline must be the accepted dev0 source-clean snapshot")
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
    # Only the version stamp changed in native code. Both Python engines use
    # the same installed binary; reject this comparison if kernel code changes.
    native = Path("packages/alignimg-gpu/src/alignimg_gpu/native")
    expected_native = {name for name in manifest["files"] if name.startswith(str(native) + "/")}
    actual_native = {str(path.relative_to(ROOT)) for path in (ROOT / native).rglob("*")
                     if path.is_file() and path.suffix in {".cpp", ".cu", ".h", ".hpp"}}
    if actual_native != expected_native:
        raise ValueError("native source file set changed")
    for name in manifest["files"]:
        if str(native) in name or name == "packages/alignimg-gpu/CMakeLists.txt":
            old = (target / name).read_bytes().replace(b"2.1.0.dev0", b"VERSION")
            new = (ROOT / name).read_bytes().replace(b"2.1.0.dev1", b"VERSION")
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

    if ai.__version__ != "2.1.0.dev0" or not Path(ai.__file__).is_relative_to(
        args.worker_source
    ):
        raise RuntimeError("frozen core was not loaded")
    if args.backend != "cpu":
        import alignimg_gpu

        if not Path(alignimg_gpu.__file__).is_relative_to(args.worker_source):
            raise RuntimeError("frozen GPU Python backend was not loaded")
        # _native is lazily imported by the backend; no installation changes.
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


def compare_reports(baseline_path, current_path):
    import numpy as np
    from tools.performance_validation import compare_arrays, sha256

    old, new = (json.loads(path.read_text()) for path in (baseline_path, current_path))
    output = {}
    for name, a in old["cases"].items():
        b = new["cases"][name]
        if a["frozen_inputs_sha256"] != b["frozen_inputs_sha256"]:
            raise AssertionError(f"{name}: baseline and current inputs differ")
        arrays = []
        for report_path, case in ((baseline_path, a), (current_path, b)):
            path = report_path.with_name(f"{report_path.stem}.{name}.result.npz")
            if sha256(path) != case["result_sha256"]:
                raise AssertionError(f"{name}: result hash mismatch")
            with np.load(path, allow_pickle=False) as data:
                arrays.append(dict(data))
        ratio = b["unprofiled_median_seconds"] / a["unprofiled_median_seconds"]
        output[name] = {
            "numeric_errors": compare_arrays(*arrays),
            "current_over_baseline_seconds": ratio,
            "timing_review": "repeat_required" if ratio > 1.05 else "within_gate",
            "baseline_counters": a["performance"]["counters"],
            "current_counters": b["performance"]["counters"],
        }
        before, after = (
            output[name]["baseline_counters"],
            output[name]["current_counters"],
        )
        # Uniform adaptive norms: K once per iteration, formerly 2*N*K.
        if (
            b["config"]["search_strategy"] == "adaptive_posterior"
            and "reference_norm_evaluations" in before
        ):
            if (
                after["reference_norm_evaluations"]
                >= before["reference_norm_evaluations"]
            ):
                raise AssertionError(f"{name}: reference norm work did not decrease")
        if before.get("polar_angular_fft_calls", 0):
            if after["polar_angular_fft_calls"] >= before["polar_angular_fft_calls"]:
                raise AssertionError(f"{name}: polar FFT work did not decrease")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda", "cupy"), default="cuda")
    parser.add_argument("--suite", choices=("smoke", "representative"), default="smoke")
    parser.add_argument("--only")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--baseline-source",
        type=Path,
        default=ROOT
        / "validation-results/performance/stage-0/dev0-source-clean-delivery",
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
        "stage": 1,
        "status": "running",
        "baseline": str(outputs[1]),
        "current": str(outputs[2]),
        "cases": {},
    }
    perf.write_report(args.output, report)
    common = [
        "--backend",
        args.backend,
        "--suite",
        args.suite,
        "--batch-size",
        str(args.batch_size),
        "--memory-fraction",
        str(args.memory_fraction),
        "--device",
        str(args.device),
        "--repeats",
        str(args.repeats),
    ]
    if args.only:
        common += ["--only", args.only]
    try:
        if perf.ai.__version__ != "2.1.0.dev1":
            raise RuntimeError("install AlignImg 2.1.0.dev1 before running stage 1 A/B")
        with tempfile.TemporaryDirectory(
            prefix="alignimg-stage1-baseline-"
        ) as temporary:
            unpack_baseline(args.baseline_source, Path(temporary))
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                *common,
                "--worker-source",
                temporary,
                "--output",
                str(outputs[1].resolve()),
            ]
            if args.backend != "cpu":
                import alignimg_gpu

                command += [
                    "--native-directory",
                    str(Path(alignimg_gpu.__file__).parent),
                ]
            subprocess.run(command, cwd=ROOT, check=True)
        if perf.main([*common, "--capture-candidates", "--output", str(outputs[2])]):
            raise RuntimeError("current engine validation failed; inspect current JSON")
        report["cases"] = compare_reports(outputs[1], outputs[2])
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
        report.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
    perf.write_report(args.output, report)
    print(f"Stage 1: {report['status']}; report: {args.output}", flush=True)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
