"""Freeze source and small single-iteration E/M-step regression fixtures."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

import alignimg as ai
from alignimg._adaptive import infer_adaptive_candidates_cpu
from alignimg._engine import _update_references_fourier, run_soft_alignment_cpu
from alignimg._fourier import infer_top_candidates


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "validation-results/performance/stage-0/baseline-2.0.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(root: Path = ROOT) -> dict:
    roots = ("src", "packages", "tools", "tests", "docs", "examples")
    excluded = {"__pycache__", "build", "dist", ".DS_Store"}
    paths = [
        root / name
        for name in (
            "pyproject.toml",
            "MANIFEST.in",
            "README.md",
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
            "Reference/re2dc/LICENSE",
        )
    ]
    for name in roots:
        paths.extend(
            path
            for path in (root / name).rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and not any(
                part in excluded or part.endswith(".egg-info") for part in path.parts
            )
            and (
                path.name == "LICENSE"
                or path.suffix
                in {
                    ".py",
                    ".toml",
                    ".md",
                    ".txt",
                    ".cu",
                    ".cpp",
                    ".hpp",
                    ".h",
                    ".qss",
                }
            )
        )
    files = {str(path.relative_to(root)): sha256(path) for path in sorted(set(paths))}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"source_sha256": digest, "files": files}


def synthetic_inputs(count: int = 8, size: int = 24) -> dict[str, np.ndarray]:
    y, x = np.indices((size, size), dtype=np.float32)
    refs = []
    for offset in (0, 4):
        value = np.exp(-((y - size * 0.3) ** 2 + (x - size * 0.65) ** 2) / (size * 0.3))
        value += 0.7 * np.exp(
            -((y - size * 0.7) ** 2 + (x - size * 0.3 - offset) ** 2) / (size * 0.2)
        )
        refs.append(value.astype(np.float32))
    references = np.stack(refs)
    labels = np.arange(count, dtype=np.int32) % 2
    rng = np.random.default_rng(20260906)
    poses = ai.PoseSet(
        rng.uniform(-18, 18, count).astype(np.float32),
        rng.uniform(-1, 1, count).astype(np.float32),
        rng.uniform(-1, 1, count).astype(np.float32),
        np.zeros(count, dtype=bool),
    )
    images = ai.transform_images(references[labels], poses, backend="cpu")
    images += rng.normal(0, 0.03, images.shape).astype(np.float32)
    return {
        "images": images,
        "references": references,
        "labels": labels,
        "priors": ai.make_class_priors(assignments=labels, n_components=2),
        "initial_angle_deg": -poses.angle_deg,
        "initial_shift_y_px": -poses.shift_y_px,
        "initial_shift_x_px": -poses.shift_x_px,
        "initial_mirror": poses.mirror,
    }


def poses_from_inputs(values: dict) -> ai.PoseSet:
    return ai.PoseSet(
        *(
            values[f"initial_{name}"]
            for name in (
                "angle_deg",
                "shift_y_px",
                "shift_x_px",
                "mirror",
            )
        )
    )


def fixture_config(adaptive: bool) -> ai.AlignmentConfig:
    return ai.AlignmentConfig(
        max_iterations=1,
        top_l=4,
        angle_samples=24,
        proposal_angles_per_reference=4,
        translation_range=2,
        search_strategy="adaptive_posterior" if adaptive else "proposal",
        local_angle_range=6,
        local_shift_range=1,
        coarse_angle_step=6,
        coarse_shift_step=1,
        robust_weighting=True,
        halfset_diagnostics=True,
        center_references=True,
        random_seed=7,
        batch_size=16,
    )


def capture_iteration(
    values: dict, config: ai.AlignmentConfig
) -> dict[str, np.ndarray]:
    """Record E output before centering and the three original M-step calls."""
    captured = {}
    calls = 0
    adaptive = config.search_strategy == "adaptive_posterior"

    def inference(*args, rescue_mask=None):
        result = (
            infer_adaptive_candidates_cpu(*args, rescue_mask=rescue_mask)
            if adaptive
            else infer_top_candidates(*args)
        )
        captured.update({f"e_{key}": value.copy() for key, value in result.items()})
        return result

    def updater(images, candidates, weights, count, cfg, **kwargs):
        nonlocal calls
        prefix = ("full", "half_a", "half_b")[calls]
        if calls == 0:
            captured["inlier_weights"] = weights.copy()
        captured[f"{prefix}_subset"] = (
            np.ones(len(images), dtype=bool)
            if kwargs.get("subset") is None
            else kwargs["subset"].copy()
        )
        references, effective, shifts = _update_references_fourier(
            images, candidates, weights, count, cfg, **kwargs
        )
        captured[f"{prefix}_references"] = references.copy()
        captured[f"{prefix}_weights"] = effective.copy()
        captured[f"{prefix}_center_shifts"] = np.asarray(shifts)
        calls += 1
        return references, effective, shifts

    result = run_soft_alignment_cpu(
        values["images"],
        values["references"],
        config=config,
        class_priors=values["priors"],
        initial_poses=poses_from_inputs(values) if adaptive else None,
        workflow="refine" if adaptive else "global",
        _candidate_inference=inference,
        _reference_updater=updater,
    )
    captured["responsibilities"] = result.responsibilities
    captured["assignments"] = result.reference_assignments
    captured["frc"] = result.diagnostics[0]["frc"]
    captured["stable_cutoff"] = result.diagnostics[0][
        "frc_0143_stable_cutoff_cyc_per_px"
    ]
    return captured


def freeze_baseline(output: Path) -> dict:
    """Never overwrite a previous freeze; run before modifying engine code."""
    output.mkdir(parents=True, exist_ok=False)
    manifest = source_manifest()
    archive = output / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name in manifest["files"]:
            bundle.add(ROOT / name, arcname=name, recursive=False)
    manifest.update(
        {"alignimg_version": ai.__version__, "archive_sha256": sha256(archive)}
    )
    values = synthetic_inputs()
    fixture_hashes = {}
    for name, adaptive in (("global", False), ("adaptive", True)):
        config = fixture_config(adaptive)
        fixture = output / f"{name}.npz"
        np.savez_compressed(
            fixture,
            **values,
            config_json=np.asarray(json.dumps(asdict(config))),
            **capture_iteration(values, config),
        )
        fixture_hashes[fixture.name] = sha256(fixture)
    manifest["fixture_sha256"] = fixture_hashes
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args()
    print(json.dumps(freeze_baseline(args.output), indent=2))
