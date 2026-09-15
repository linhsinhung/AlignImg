#!/usr/bin/env python3
"""Run the AlignImg 2.1 release-candidate preflight and formal RF gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import tarfile
import traceback
from typing import Any

import mrcfile
import numpy as np
from scipy.optimize import linear_sum_assignment

import alignimg as ai

if __package__:
    from tools import re2dc_70s_relion_benchmark as relion_benchmark
    from tools import re2dc_70s_rf_validation as rf_validation
    from tools.performance_fixtures import ROOT, source_manifest
    from tools.performance_validation import installed_source_info
    from tools.re2dc_70s_final_validation import validate_result_bundle
    from tools.server_validation import environment_info
else:
    import re2dc_70s_relion_benchmark as relion_benchmark
    import re2dc_70s_rf_validation as rf_validation
    from performance_fixtures import ROOT, source_manifest
    from performance_validation import installed_source_info
    from re2dc_70s_final_validation import validate_result_bundle
    from server_validation import environment_info


EXPECTED_VERSION = "2.1.0"
DEFAULT_STACK = ROOT / "data/re2dc_70s_testdata/prepared/re2dc_70s_n5000_s128.mrcs"
DEFAULT_PREPARED_STAR = (
    ROOT / "data/re2dc_70s_testdata/prepared/re2dc_70s_n5000_s128.star"
)
DEFAULT_RELION_STAR = (
    ROOT / "data/re2dc_70s_testdata/particles_Relion2Dclassification.star"
)
DEFAULT_BASELINE = ROOT / "validation-results/re2dc-70s-final-n5000-k10-i20.json"
DEFAULT_DEV10_MANIFEST = (
    ROOT / "validation-results/performance/stage-5/dev10-delivery/manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT / "validation-results/performance/stage-7/alignimg-2.1.0-final.json"
)

FORMAL_SPEC = {
    "component_count": 10,
    "iterations": 20,
    "anneal_iterations": 10,
    "seeds": [0, 1, 2],
    "angle_samples": 128,
    "translation_range": 4.0,
    "candidate_scoring": "fourier",
    "score_model": "fourier_ncc",
    "reference_update": "fourier",
}

FROZEN_COMPUTE_FILES = (
    "src/alignimg/_adaptive.py",
    "src/alignimg/_engine.py",
    "src/alignimg/_fourier.py",
    "src/alignimg/_fourier_native.py",
    "src/alignimg/_frequency_tables.py",
    "src/alignimg/_geometry.py",
    "src/alignimg/_profiling.py",
    "src/alignimg/_transform.py",
    "src/alignimg/compat.py",
    "src/alignimg/models.py",
    "src/alignimg/priors.py",
    "src/alignimg/workflows.py",
    "packages/alignimg-gpu/src/alignimg_gpu/_workspace.py",
    "packages/alignimg-gpu/src/alignimg_gpu/backend.py",
    "packages/alignimg-gpu/src/alignimg_gpu/memory.py",
    "packages/alignimg-gpu/src/alignimg_gpu/native/transform_cuda.cu",
    "packages/alignimg-gpu/src/alignimg_gpu/native/transform_cuda.hpp",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derived_path(output: Path, suffix: str) -> Path:
    return Path(f"{output.with_suffix('')}.{suffix}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_dev10_compute_freeze(
    manifest_path: Path = DEFAULT_DEV10_MANIFEST,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    expected = manifest["files"]
    archive = manifest_path.parent / "source.tar.gz"
    if not archive.exists():
        archive = DEFAULT_DEV10_MANIFEST.parent / "source.tar.gz"
    expected_archive_hash = manifest.get("archive_sha256")
    if expected_archive_hash is not None and sha256(archive) != expected_archive_hash:
        raise RuntimeError("frozen dev10 source archive hash changed")
    files: dict[str, dict[str, Any]] = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for name in FROZEN_COMPUTE_FILES:
            expected_hash = expected.get(name)
            if expected_hash is None:
                raise RuntimeError(f"dev10 manifest is missing frozen source: {name}")
            extracted = bundle.extractfile(name)
            if extracted is None:
                raise RuntimeError(f"dev10 source archive is missing: {name}")
            actual_hash = hashlib.sha256(extracted.read()).hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError(f"frozen dev10 compute source changed: {name}")
            files[name] = {"sha256": actual_hash, "matches_dev10": True}
    return {
        "baseline_version": manifest["alignimg_version"],
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "files": files,
    }


def _distribution_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as error:
        raise RuntimeError(f"required distribution is not installed: {name}") from error


def release_preflight(backend: str) -> dict[str, Any]:
    if ai.__version__ != EXPECTED_VERSION:
        raise RuntimeError(
            f"AlignImg module version is {ai.__version__}, expected {EXPECTED_VERSION}"
        )
    if _distribution_version("alignimg") != EXPECTED_VERSION:
        raise RuntimeError("installed AlignImg metadata does not match the release candidate")

    config = ai.AlignmentConfig()
    defaults = {
        "candidate_scoring": config.candidate_scoring,
        "score_model": config.score_model,
        "reference_update": config.reference_update,
        "memory_fraction": config.memory_fraction,
    }
    expected_defaults = {
        "candidate_scoring": "fourier",
        "score_model": "fourier_ncc",
        "reference_update": "fourier",
        "memory_fraction": 0.8,
    }
    if defaults != expected_defaults:
        raise RuntimeError(f"release defaults changed: {defaults}")
    if hasattr(config, "gpu_accumulation_precision"):
        raise RuntimeError("rejected Stage 6 precision option is still public")

    manifest = source_manifest(ROOT)
    installed = {
        "alignimg": installed_source_info(
            ai, "src/alignimg/", manifest["files"]
        )
    }
    gpu_versions = None
    native_version = None
    if backend != "cpu":
        import alignimg_gpu

        if alignimg_gpu.__version__ != EXPECTED_VERSION:
            raise RuntimeError("AlignImg GPU module does not match the release candidate")
        if _distribution_version("alignimg-gpu") != EXPECTED_VERSION:
            raise RuntimeError("installed AlignImg GPU metadata does not match the release candidate")
        installed["alignimg_gpu"] = installed_source_info(
            alignimg_gpu, "packages/alignimg-gpu/src/alignimg_gpu/", manifest["files"]
        )
        gpu_versions = {
            "module": alignimg_gpu.__version__,
            "distribution": _distribution_version("alignimg-gpu"),
        }
        if backend == "cuda":
            from alignimg_gpu.backend import _native_module

            native = _native_module()
            if native is None:
                raise RuntimeError("native CUDA extension is unavailable")
            native_version = str(native.__version__)
            if native_version != EXPECTED_VERSION:
                raise RuntimeError("native CUDA extension does not match the release candidate")

    available = ai.available_alignment_backends()
    if not available.get(backend, {}).get("available", False):
        raise RuntimeError(f"requested backend is unavailable: {backend}")
    return {
        "alignimg": {
            "module": ai.__version__,
            "distribution": _distribution_version("alignimg"),
        },
        "alignimg_gpu": gpu_versions,
        "native_cuda": native_version,
        "defaults": defaults,
        "rejected_precision_option_absent": True,
        "source": {
            "sha256": manifest["source_sha256"],
            "installed": installed,
            "dev10_compute_freeze": verify_dev10_compute_freeze(),
        },
    }


def validate_formal_report(
    report: dict[str, Any], *, backend: str, batch_size: int, memory_fraction: float
) -> dict[str, Any]:
    if report.get("status") != "completed":
        raise RuntimeError("formal RF report did not complete")
    parameters = report["parameters"]
    actual_spec = {
        "component_count": parameters["component_count"],
        "iterations": parameters["config"]["max_iterations"],
        "anneal_iterations": parameters["config"]["temperature_anneal_iterations"],
        "seeds": parameters["seeds"],
        "angle_samples": parameters["config"]["angle_samples"],
        "translation_range": parameters["config"]["translation_range"],
        "candidate_scoring": parameters["config"]["candidate_scoring"],
        "score_model": parameters["config"]["score_model"],
        "reference_update": parameters["config"]["reference_update"],
    }
    if actual_spec != FORMAL_SPEC:
        raise RuntimeError(f"formal RF specification changed: {actual_spec}")
    if parameters["backend"] != backend:
        raise RuntimeError("formal report backend changed")
    if parameters["config"]["batch_size"] != batch_size:
        raise RuntimeError("formal report batch size changed")
    if parameters["config"]["memory_fraction"] != memory_fraction:
        raise RuntimeError("formal report memory fraction changed")

    technical: dict[str, Any] = {}
    expected_engine = f"alignimg-soft-fourier-{backend}"
    for seed in FORMAL_SPEC["seeds"]:
        run = report["runs"].get(str(seed))
        if run is None:
            raise RuntimeError(f"formal report is missing seed {seed}")
        metadata = run["metadata"]
        if metadata["backend"] != backend or metadata["engine"] != expected_engine:
            raise RuntimeError(f"seed {seed} used an unexpected backend or engine")
        for name in ("candidate_scoring", "score_model", "reference_update"):
            if metadata[name] != FORMAL_SPEC[name]:
                raise RuntimeError(f"seed {seed} changed {name}")
        plans = metadata.get("gpu_memory_plans", []) if backend != "cpu" else []
        planned_batches = [plan["batch_size"] for plan in plans if "batch_size" in plan]
        if backend != "cpu" and (
            not planned_batches or any(value != batch_size for value in planned_batches)
        ):
            raise RuntimeError(f"seed {seed} did not retain batch {batch_size}")

        artifacts = run["artifacts"]
        result_path = Path(artifacts["result"])
        reference_path = Path(artifacts["references"])
        bundle = validate_result_bundle(
            result_path,
            reference_path,
            particle_count=5000,
            component_count=FORMAL_SPEC["component_count"],
        )
        with np.load(result_path, allow_pickle=False) as saved:
            saved_version = str(saved["alignimg_version"])
        if saved_version != EXPECTED_VERSION:
            raise RuntimeError(f"seed {seed} artifact version is {saved_version}")
        technical[str(seed)] = {
            **bundle,
            "backend": metadata["backend"],
            "engine": metadata["engine"],
            "memory_plan_count": len(plans),
            "planned_batch_sizes": sorted(set(planned_batches)),
            "artifacts_sha256": {
                name: sha256(Path(path)) for name, path in artifacts.items()
            },
        }
    return technical


def compare_to_baseline(
    current: dict[str, Any], baseline: dict[str, Any], baseline_path: Path
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    current_seconds = []
    baseline_seconds = []
    for seed in FORMAL_SPEC["seeds"]:
        current_run = current["runs"][str(seed)]
        baseline_run = baseline["runs"][str(seed)]
        current_seconds.append(float(current_run["seconds"]))
        baseline_seconds.append(float(baseline_run["seconds"]))

        current_result = Path(current_run["artifacts"]["result"])
        baseline_result = Path(baseline_run["artifacts"]["result"])
        with np.load(current_result, allow_pickle=False) as new, np.load(
            baseline_result, allow_pickle=False
        ) as old:
            assignment_ari = rf_validation.adjusted_rand_index(
                np.asarray(old["assignments"]), np.asarray(new["assignments"])
            )

        current_references = Path(current_run["artifacts"]["references"])
        baseline_references = Path(baseline_run["artifacts"]["references"])
        with mrcfile.mmap(current_references, permissive=True, mode="r") as new_mrc:
            new_profiles = rf_validation.radial_profiles(np.asarray(new_mrc.data))
        with mrcfile.mmap(baseline_references, permissive=True, mode="r") as old_mrc:
            old_profiles = rf_validation.radial_profiles(np.asarray(old_mrc.data))
        similarity = old_profiles @ new_profiles.T
        rows, columns = linear_sum_assignment(-similarity)
        comparisons[str(seed)] = {
            "runtime_seconds": {
                "baseline": baseline_seconds[-1],
                "release_candidate": current_seconds[-1],
                "candidate_over_baseline": current_seconds[-1]
                / baseline_seconds[-1],
            },
            "assignment_adjusted_rand_index": assignment_ari,
            "occupancy_entropy_delta": float(
                current_run["hard_occupancy_normalized_entropy"]
                - baseline_run["hard_occupancy_normalized_entropy"]
            ),
            "mean_max_responsibility_delta": float(
                current_run["mean_max_responsibility"]
                - baseline_run["mean_max_responsibility"]
            ),
            "radial_profile_reference_matching": {
                "baseline_indices": rows,
                "candidate_indices": columns,
                "matched_correlations": similarity[rows, columns],
                "mean_matched_correlation": float(np.mean(similarity[rows, columns])),
                "scope_note": "Radial profiles discard angular information; this is diagnostic only.",
            },
        }
    median_current = float(np.median(current_seconds))
    median_baseline = float(np.median(baseline_seconds))
    return {
        "baseline_report": str(baseline_path),
        "per_seed": comparisons,
        "median_runtime_seconds": {
            "baseline": median_baseline,
            "release_candidate": median_current,
            "candidate_over_baseline": median_current / median_baseline,
        },
        "scope_note": (
            "Performance and scientific comparisons require review; they are not "
            "automatic biological-classification pass/fail thresholds."
        ),
    }


def main(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {
        "schema": "alignimg.stage7-release-validation.v1",
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "release_candidate": EXPECTED_VERSION,
        "spec": {
            **FORMAL_SPEC,
            "backend": args.backend,
            "batch_size": args.batch_size,
            "memory_fraction": args.memory_fraction,
        },
    }
    rf_validation.write_json(args.output, report)
    try:
        report["environment"] = environment_info(args.backend, args.device)
        report["preflight"] = release_preflight(args.backend)
        if args.check_only:
            report["status"] = "completed"
            report["release_decision"] = "preflight_passed"
            report["completed_utc"] = datetime.now(timezone.utc).isoformat()
            rf_validation.write_json(args.output, report)
            print(f"PASS Stage 7 preflight: {args.output.resolve()}")
            return 0

        for path in (
            args.stack,
            args.prepared_star,
            args.relion_star,
            args.baseline_report,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        rf_output = derived_path(args.output, "rf.json")
        relion_output = derived_path(args.output, "relion.json")
        report["inputs"] = {
            "stack": str(args.stack),
            "stack_sha256": sha256(args.stack),
            "prepared_star": str(args.prepared_star),
            "prepared_star_sha256": sha256(args.prepared_star),
            "relion_star": str(args.relion_star),
            "relion_star_sha256": sha256(args.relion_star),
            "baseline_report": str(args.baseline_report),
            "baseline_report_sha256": sha256(args.baseline_report),
        }
        report["artifacts"] = {
            "rf_report": str(rf_output),
            "relion_report": str(relion_output),
        }
        report["phase"] = "reference_free"
        rf_validation.write_json(args.output, report)

        print("PHASE 1/2  formal 5000-particle RF, seeds 0/1/2", flush=True)
        rf_validation.main(
            argparse.Namespace(
                stack=args.stack,
                output=rf_output,
                backend=args.backend,
                components=FORMAL_SPEC["component_count"],
                iterations=FORMAL_SPEC["iterations"],
                anneal_iterations=FORMAL_SPEC["anneal_iterations"],
                seeds=FORMAL_SPEC["seeds"],
                angle_samples=FORMAL_SPEC["angle_samples"],
                candidate_scoring=FORMAL_SPEC["candidate_scoring"],
                score_model=FORMAL_SPEC["score_model"],
                reference_update=FORMAL_SPEC["reference_update"],
                top_l=None,
                proposal_angles_per_reference=None,
                temperature_start=None,
                temperature_end=None,
                translation_range=FORMAL_SPEC["translation_range"],
                batch_size=args.batch_size,
                memory_fraction=args.memory_fraction,
                minimum_frc_halfset_weight=10.0,
            )
        )
        current = read_json(rf_output)
        report["technical_checks"] = validate_formal_report(
            current,
            backend=args.backend,
            batch_size=args.batch_size,
            memory_fraction=args.memory_fraction,
        )
        report["baseline_comparison"] = compare_to_baseline(
            current, read_json(args.baseline_report), args.baseline_report
        )
        report["phase"] = "relion_comparison"
        rf_validation.write_json(args.output, report)

        print("PHASE 2/2  RELION partition comparison", flush=True)
        rf_results = [
            Path(f"{rf_output.with_suffix('')}.seed-{seed}.result.npz")
            for seed in FORMAL_SPEC["seeds"]
        ]
        relion_args = argparse.Namespace(
            relion_star=args.relion_star,
            prepared_star=args.prepared_star,
            rf_results=rf_results,
            output=relion_output,
            confidence_thresholds=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
        )
        relion_report = relion_benchmark.build_report(relion_args)
        rf_validation.write_json(relion_output, relion_report)
        report["relion_comparison"] = [
            {
                "result": run["result"],
                "adjusted_rand_index": run["overall"]["adjusted_rand_index"],
                "normalized_mutual_information": run["overall"][
                    "normalized_mutual_information"
                ],
                "optimal_label_agreement": run["overall"][
                    "optimal_label_agreement"
                ],
            }
            for run in relion_report["runs"]
        ]
        report["status"] = "completed"
        report["phase"] = "completed"
        report["technical_gate"] = "passed"
        report["release_decision"] = "pending_review"
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        rf_validation.write_json(args.output, report)
        print(f"PASS Stage 7 technical gate: {args.output.resolve()}")
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["technical_gate"] = "failed"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        rf_validation.write_json(args.output, report)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", type=Path, default=DEFAULT_STACK)
    parser.add_argument("--prepared-star", type=Path, default=DEFAULT_PREPARED_STAR)
    parser.add_argument("--relion-star", type=Path, default=DEFAULT_RELION_STAR)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backend", choices=("cpu", "cuda", "cupy"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate source, installed packages, native CUDA, and defaults only.",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 0.0 < args.memory_fraction <= 1.0:
        parser.error("--memory-fraction must be in (0, 1]")
    return args


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
