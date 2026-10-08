#!/usr/bin/env python3
"""Capture the 2.3.0 Fast 3 baseline before installing 2.3.1 changes."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import socket
import time

import mrcfile
import numpy as np

import alignimg as ai

try:
    from tools.performance_fixtures import ROOT, source_manifest
except ModuleNotFoundError:
    from performance_fixtures import ROOT, source_manifest


# The accepted server snapshot and final main differ only in docs/test files.
ACCEPTED_SOURCE_SNAPSHOTS = {
    "5c09e9553b7c7d51396e908c9c5333de46ced44436d0dd80b2c706adbb62f816": "server-cleanup-r2",
    "548fe80789778bb54b232d01914ef16bdd366f306a2fdc1426509c324fb3b68a": "main-4a51d41",
}
EXPECTED_PARTICLES_SHA256 = "228978d659e01b7a8d30fb05e6d84a158dc9c7ef0d6f35fcbae5b4f3e4cec253"
EXPECTED_REFERENCE_SHA256 = "7e70a74660bd5e2e6f67227b8a70bd90092d11bf2a70169d9aee505b0664dc72"
FROZEN_INPUTS = {
    "synthetic": (
        "validation-results/fast-hard/stage-0/frozen-inputs/synthetic.inputs.npz",
        "95940983ce6cb74c492117345701a320f8bbfee903eb25ec2daad60ad4dd63b2",
    ),
    "homogeneous_384": (
        "data/re2dc_70s_testdata/prepared/re2dc_70s_pose_benchmark.particles.mrcs",
        "725292936e7ea9c9a31c10c2ab18fb1ef1f68d94cfe47c7c20813cf5f8bffe45",
    ),
    "homogeneous_384_references": (
        "data/re2dc_70s_testdata/prepared/re2dc_70s_pose_benchmark.references.mrcs",
        "eb82e011c1b185e96e0a38f11e7cded3147fd401f35acaa8a40e723818cc93e5",
    ),
    "homogeneous_384_truth": (
        "data/re2dc_70s_testdata/prepared/re2dc_70s_pose_benchmark.truth.npz",
        "60d2f36a5fd0492203a19e347ad8b226aa8e181503f6eb5a095375e5ab845ca2",
    ),
    "mra1000": (
        "validation-results/fast-hard/stage-0/frozen-inputs/mra1000.inputs.npz",
        "4320c2777548d1d2c38597dd0476eeb73d41ec310f2296e5ae3b4e11e193da11",
    ),
    "mra1000_stack": (
        "data/re2dc_70s_testdata/prepared/re2dc_70s_n1000_s128.mrcs",
        "eb51dba8b3afdc593f95a2abc84c1a1e2b0c1c81c7f9e2ea3e1a4df572dcf519",
    ),
    "mra1000_references": (
        "validation-results/re2dc-70s-final-n1000.rf.seed-0.result.npz",
        "c0e255774a6f2b329aae0d6cfe8e9162d1c07363b87442ac6e6c465de68b91b5",
    ),
    "homogeneous_384_manifest": (
        "data/re2dc_70s_testdata/prepared/re2dc_70s_pose_benchmark.json",
        "cd2332dedff1a12927b41a690b70695f1607d54ba41b9ef9e7af10891a52dba8",
    ),
}
OUTPUT_BASE = ROOT / "validation-results/maintenance-2.3.1/baseline/fast3-n3050-2.3.0"
PARTICLES = ROOT / "data/local/test_align.mrcs"
REFERENCE = ROOT / "data/local/mu_aligned_mean.mrc"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stack(path: Path) -> tuple[np.ndarray, float | None]:
    with mrcfile.mmap(path, permissive=True, mode="r") as mrc:
        images = np.asarray(mrc.data, dtype=np.float32).copy()
        pixel_size = float(mrc.voxel_size.x)
    if images.ndim == 2:
        images = images[None]
    if images.ndim != 3 or images.shape[-1] != images.shape[-2]:
        raise ValueError(f"expected square 2-D image stack: {path}")
    return images, pixel_size if pixel_size > 0.0 else None


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def verify_baseline_source(manifest: dict) -> tuple[str, str]:
    base_files = {
        path: digest
        for path, digest in manifest["files"].items()
        if path not in {
            "tools/polar_231_baseline.py",
            "tests/gpu/test_polar_231_regressions.py",
        }
    }
    source_hash = hashlib.sha256(json.dumps(base_files, sort_keys=True).encode()).hexdigest()
    if source_hash not in ACCEPTED_SOURCE_SNAPSHOTS:
        raise RuntimeError(
            f"2.3.0 base source mismatch: {source_hash}; expected one of "
            f"{', '.join(ACCEPTED_SOURCE_SNAPSHOTS)}"
        )
    return source_hash, ACCEPTED_SOURCE_SNAPSHOTS[source_hash]


def main() -> int:
    report_path = OUTPUT_BASE.with_suffix(".json")
    result_path = OUTPUT_BASE.with_suffix(".npz")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists() or result_path.exists():
        raise FileExistsError("baseline outputs already exist; choose a fresh directory")
    if ai.__version__ != "2.3.0":
        raise RuntimeError(f"expected AlignImg 2.3.0, found {ai.__version__}")

    source_hash, source_snapshot = verify_baseline_source(source_manifest())
    frozen_hashes = {}
    for name, (relative_path, expected_hash) in FROZEN_INPUTS.items():
        path = ROOT / relative_path
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"frozen {name} input hash mismatch: {actual_hash} != {expected_hash}"
            )
        frozen_hashes[name] = {"path": relative_path, "sha256": actual_hash}

    images, pixel_size = load_stack(PARTICLES)
    reference_stack, _ = load_stack(REFERENCE)
    reference = reference_stack[0]
    input_hashes = {"particles": sha256(PARTICLES), "reference": sha256(REFERENCE)}
    if input_hashes != {
        "particles": EXPECTED_PARTICLES_SHA256,
        "reference": EXPECTED_REFERENCE_SHA256,
    }:
        raise RuntimeError(f"frozen 3,050-particle input hash mismatch: {input_hashes}")
    if images.shape != (3050, reference.shape[0], reference.shape[1]):
        raise ValueError(f"unexpected frozen workload dimensions: {images.shape}")

    import alignimg_gpu
    from alignimg_gpu.backend import _native_module

    native = _native_module()
    if alignimg_gpu.__version__ != "2.3.0" or native is None or native.__version__ != "2.3.0":
        raise RuntimeError("GPU package and native CUDA extension must both be 2.3.0")
    if not ai.available_alignment_backends().get("cuda", {}).get("available", False):
        raise RuntimeError("native CUDA backend is unavailable")

    import cupy as cp

    config = replace(
        ai.AlignmentConfig.preset("fast3"), batch_size=512, memory_fraction=0.8
    )

    def operation(stack):
        return ai.align_to_references(
            stack, reference[None], config=config, backend="cuda"
        )

    operation(images[:8])  # short CUDA warm-up; not included in timings
    timings = []
    results = []
    for repeat in range(3):
        cp.cuda.Stream.null.synchronize()
        started = time.perf_counter()
        result = operation(images)
        cp.cuda.Stream.null.synchronize()
        timings.append(time.perf_counter() - started)
        results.append(result)
        print(f"RUN Fast 3 baseline repeat={repeat + 1}/3: {timings[-1]:.3f} s", flush=True)

    for result in results[1:]:
        for field in ("angle_deg", "shift_y_px", "shift_x_px", "mirror"):
            np.testing.assert_array_equal(
                getattr(results[0].poses, field), getattr(result.poses, field)
            )
        for field in (
            "reference_assignments",
            "responsibilities",
            "references",
            "class_averages",
            "inlier_weights",
        ):
            np.testing.assert_array_equal(
                getattr(results[0], field), getattr(result, field)
            )
    final = results[-1]
    np.savez_compressed(
        result_path,
        angle_deg=final.poses.angle_deg,
        shift_y_px=final.poses.shift_y_px,
        shift_x_px=final.poses.shift_x_px,
        mirror=final.poses.mirror,
        assignments=final.reference_assignments,
        responsibilities=final.responsibilities,
        references=final.references,
        class_averages=final.class_averages,
        inlier_weights=final.inlier_weights,
    )
    report = {
        "schema": "alignimg.maintenance-2.3.1-baseline.v1",
        "status": "passed",
        "baseline": "AlignImg 2.3.0 Fast 3, CUDA, N=3050, K=1",
        "source": {
            "base_source_sha256": source_hash,
            "base_source_snapshot": source_snapshot,
            "working_tree_revision": _git_revision(),
            "versions": {
                "alignimg_module": ai.__version__,
                "alignimg_metadata": importlib.metadata.version("alignimg"),
                "alignimg_gpu_module": alignimg_gpu.__version__,
                "alignimg_gpu_metadata": importlib.metadata.version("alignimg-gpu"),
                "native_cuda": native.__version__,
            },
            "module_paths": {
                "alignimg": ai.__file__,
                "alignimg_gpu": alignimg_gpu.__file__,
                "native_cuda": native.__file__,
            },
            "native_binary_sha256": sha256(Path(native.__file__)),
        },
        "inputs": {
            "frozen_benchmark_inputs": frozen_hashes,
            "particles": str(PARTICLES),
            "particles_sha256": input_hashes["particles"],
            "reference": str(REFERENCE),
            "reference_sha256": input_hashes["reference"],
            "shape": list(images.shape),
            "pixel_size_angstrom": pixel_size,
        },
        "config": asdict(config),
        "timings_seconds": timings,
        "median_seconds": float(np.median(timings)),
        "deterministic_repeats": True,
        "final_metadata": final.metadata,
        "reference_correlation": correlation(final.class_averages[0], reference),
        "result_npz": str(result_path),
        "result_npz_sha256": sha256(result_path),
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "gpu_free_bytes_after": int(cp.cuda.runtime.memGetInfo()[0]),
            "gpu_total_bytes": int(cp.cuda.runtime.memGetInfo()[1]),
        },
    }
    report_path.write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Baseline saved: {report_path}", flush=True)
    print(f"Result saved: {result_path}", flush=True)
    return 0


def _git_revision() -> str | None:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
