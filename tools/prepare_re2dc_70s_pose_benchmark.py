#!/usr/bin/env python3
"""Build disjoint RELION-reference and evaluation splits for pose benchmarking."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np

import alignimg as ai
from alignimg._geometry import CENTER_CONVENTION

if __package__:
    from tools.prepare_re2dc_70s_benchmark import read_particle_loop, write_star
    from tools.re2dc_70s_relion_benchmark import align_relion_metadata
else:
    from prepare_re2dc_70s_benchmark import read_particle_loop, write_star
    from re2dc_70s_relion_benchmark import align_relion_metadata


DEFAULT_STACK = Path(
    "data/re2dc_70s_testdata/prepared/re2dc_70s_n5000_s128.mrcs"
)
DEFAULT_PREPARED_STAR = Path(
    "data/re2dc_70s_testdata/prepared/re2dc_70s_n5000_s128.star"
)
DEFAULT_RELION_STAR = Path(
    "data/re2dc_70s_testdata/particles_Relion2Dclassification.star"
)
DEFAULT_OUTPUT = Path(
    "data/re2dc_70s_testdata/prepared/re2dc_70s_pose_benchmark"
)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def relion_to_alignimg_poses(
    angle_psi_deg: np.ndarray,
    origin_x_angstrom: np.ndarray,
    origin_y_angstrom: np.ndarray,
    pixel_size_angstrom: float,
) -> ai.PoseSet:
    """Convert RELION 2D poses into particle-to-reference AlignImg poses.

    RELION's Psi rotates a reference into an observation. Its origins shift the
    observation toward the reference before inverse rotation. AlignImg stores a
    particle-to-reference CCW rotation followed by a post-rotation y/x shift.
    """
    if pixel_size_angstrom <= 0.0:
        raise ValueError("pixel_size_angstrom must be positive")
    psi = np.asarray(angle_psi_deg, dtype=np.float64)
    origin_x = np.asarray(origin_x_angstrom, dtype=np.float64)
    origin_y = np.asarray(origin_y_angstrom, dtype=np.float64)
    if not (psi.shape == origin_x.shape == origin_y.shape) or psi.ndim != 1:
        raise ValueError("RELION pose fields must be one-dimensional arrays of equal shape")
    if not all(np.all(np.isfinite(value)) for value in (psi, origin_x, origin_y)):
        raise ValueError("RELION pose fields must contain only finite values")
    angle = -psi
    radians = np.deg2rad(angle)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    pre_x = origin_x / float(pixel_size_angstrom)
    pre_y = origin_y / float(pixel_size_angstrom)
    shift_x = cosine * pre_x + sine * pre_y
    shift_y = -sine * pre_x + cosine * pre_y
    return ai.PoseSet(
        angle.astype(np.float32),
        shift_y.astype(np.float32),
        shift_x.astype(np.float32),
        np.zeros(len(angle), dtype=np.bool_),
    )


def select_disjoint_splits(
    class_number: np.ndarray,
    confidence: np.ndarray,
    *,
    classes: tuple[int, ...],
    reference_count: int,
    evaluation_count: int,
    minimum_confidence: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select reproducible per-class reference/evaluation indices without overlap."""
    labels = np.asarray(class_number, dtype=np.int64)
    probability = np.asarray(confidence, dtype=np.float64)
    if labels.shape != probability.shape or labels.ndim != 1:
        raise ValueError("class_number and confidence must be equal one-dimensional arrays")
    if reference_count < 1 or evaluation_count < 1:
        raise ValueError("reference_count and evaluation_count must be positive")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be in [0, 1]")
    reference_rows: list[np.ndarray] = []
    evaluation_rows: list[np.ndarray] = []
    for class_value in classes:
        eligible = np.flatnonzero(
            (labels == class_value) & (probability >= minimum_confidence)
        )
        required = reference_count + evaluation_count
        if len(eligible) < required:
            raise ValueError(
                f"RELION class {class_value} has {len(eligible)} particles at confidence "
                f">= {minimum_confidence}, but {required} are required"
            )
        class_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(class_value)])
        )
        selected = class_rng.permutation(eligible)[:required]
        reference_rows.append(np.sort(selected[:reference_count]))
        evaluation_rows.append(np.sort(selected[reference_count:]))
    references = np.stack(reference_rows).astype(np.int64)
    evaluations = np.stack(evaluation_rows).astype(np.int64)
    if np.intersect1d(references, evaluations).size:
        raise AssertionError("reference and evaluation splits overlap")
    return references, evaluations


def stack_correlations(images: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(images, dtype=np.float64).reshape(len(images), -1)
    target = np.asarray(reference, dtype=np.float64).ravel()
    values -= values.mean(axis=1, keepdims=True)
    target -= target.mean()
    numerator = values @ target
    denominator = np.linalg.norm(values, axis=1) * np.linalg.norm(target)
    return numerator / np.maximum(denominator, 1e-12)


def build(args: argparse.Namespace) -> dict[str, Any]:
    classes = tuple(sorted(set(int(value) for value in args.classes)))
    if not classes or any(value < 1 for value in classes):
        raise ValueError("classes must contain one-based positive RELION class numbers")
    metadata = align_relion_metadata(args.relion_star, args.prepared_star)
    required = (
        "max_value_probability",
        "angle_psi_deg",
        "origin_x_angstrom",
        "origin_y_angstrom",
    )
    missing = [name for name in required if metadata[name] is None]
    if missing:
        raise ValueError(f"RELION STAR is missing pose benchmark fields: {missing}")

    with mrcfile.mmap(args.stack, permissive=True, mode="r") as mrc:
        stack = np.asarray(mrc.data, dtype=np.float32)
        pixel_size = float(mrc.voxel_size.x)
        if stack.ndim != 3 or stack.shape[1] != stack.shape[2]:
            raise ValueError("prepared stack must contain square 2D images")
        if stack.shape[1] % 2:
            raise ValueError("prepared stack image size must be even")
        if len(stack) != len(metadata["class_number"]):
            raise ValueError("prepared stack and aligned RELION metadata differ in length")
        if pixel_size <= 0.0:
            raise ValueError("prepared stack must record a positive pixel size")
        reference_indices, evaluation_indices = select_disjoint_splits(
            np.asarray(metadata["class_number"]),
            np.asarray(metadata["max_value_probability"]),
            classes=classes,
            reference_count=args.reference_count,
            evaluation_count=args.evaluation_count,
            minimum_confidence=args.minimum_confidence,
            seed=args.seed,
        )
        reference_images = np.asarray(stack[reference_indices.ravel()], dtype=np.float32)
        evaluation_images = np.asarray(stack[evaluation_indices.ravel()], dtype=np.float32)

    reference_poses = relion_to_alignimg_poses(
        np.asarray(metadata["angle_psi_deg"])[reference_indices.ravel()],
        np.asarray(metadata["origin_x_angstrom"])[reference_indices.ravel()],
        np.asarray(metadata["origin_y_angstrom"])[reference_indices.ravel()],
        pixel_size,
    )
    evaluation_poses = relion_to_alignimg_poses(
        np.asarray(metadata["angle_psi_deg"])[evaluation_indices.ravel()],
        np.asarray(metadata["origin_x_angstrom"])[evaluation_indices.ravel()],
        np.asarray(metadata["origin_y_angstrom"])[evaluation_indices.ravel()],
        pixel_size,
    )

    aligned_reference_particles = ai.transform_images(reference_images, reference_poses)
    aligned_reference_particles = aligned_reference_particles.reshape(
        len(classes), args.reference_count, stack.shape[1], stack.shape[2]
    )
    oracle_references = aligned_reference_particles.mean(axis=1).astype(np.float32)

    class_diagnostics: list[dict[str, Any]] = []
    for component, class_value in enumerate(classes):
        aligned = aligned_reference_particles[component]
        raw = reference_images[
            component * args.reference_count : (component + 1) * args.reference_count
        ]
        aligned_ncc = stack_correlations(aligned, oracle_references[component])
        raw_mean = raw.mean(axis=0)
        raw_ncc = stack_correlations(raw, raw_mean)
        selected_reference = reference_indices[component]
        selected_evaluation = evaluation_indices[component]
        class_diagnostics.append(
            {
                "relion_class_number": class_value,
                "eligible_count": int(
                    np.sum(
                        (np.asarray(metadata["class_number"]) == class_value)
                        & (
                            np.asarray(metadata["max_value_probability"])
                            >= args.minimum_confidence
                        )
                    )
                ),
                "reference_image_ids": np.asarray(metadata["image_id"])[selected_reference],
                "evaluation_image_ids": np.asarray(metadata["image_id"])[selected_evaluation],
                "reference_confidence_quantiles": np.quantile(
                    np.asarray(metadata["max_value_probability"])[selected_reference],
                    (0.0, 0.5, 1.0),
                ),
                "evaluation_confidence_quantiles": np.quantile(
                    np.asarray(metadata["max_value_probability"])[selected_evaluation],
                    (0.0, 0.5, 1.0),
                ),
                "raw_mean_particle_ncc": float(np.mean(raw_ncc)),
                "relion_aligned_mean_particle_ncc": float(np.mean(aligned_ncc)),
                "ncc_improvement": float(np.mean(aligned_ncc) - np.mean(raw_ncc)),
            }
        )

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    particle_path = Path(f"{output}.particles.mrcs")
    reference_path = Path(f"{output}.references.mrcs")
    star_path = Path(f"{output}.particles.star")
    truth_path = Path(f"{output}.truth.npz")
    manifest_path = Path(f"{output}.json")
    with mrcfile.new(particle_path, overwrite=True) as mrc:
        mrc.set_data(evaluation_images)
        mrc.voxel_size = pixel_size
        mrc.update_header_stats()
    with mrcfile.new(reference_path, overwrite=True) as mrc:
        mrc.set_data(oracle_references)
        mrc.voxel_size = pixel_size
        mrc.update_header_stats()

    labels, prepared_rows = read_particle_loop(args.prepared_star)
    selected_rows = [prepared_rows[int(index)] for index in evaluation_indices.ravel()]
    write_star(star_path, labels, selected_rows, particle_path.name)

    component_index = np.repeat(
        np.arange(len(classes), dtype=np.int32), args.evaluation_count
    )
    np.savez_compressed(
        truth_path,
        alignimg_version=np.asarray(ai.__version__),
        center_convention=np.asarray(CENTER_CONVENTION),
        pixel_size_angstrom=np.asarray(pixel_size, dtype=np.float64),
        relion_class_numbers=np.asarray(classes, dtype=np.int32),
        component_index=component_index,
        prepared_index_zero_based=evaluation_indices.ravel(),
        image_id=np.asarray(metadata["image_id"])[evaluation_indices.ravel()],
        relion_confidence=np.asarray(metadata["max_value_probability"])[
            evaluation_indices.ravel()
        ],
        relion_significant_samples=np.asarray(metadata["significant_samples"])[
            evaluation_indices.ravel()
        ],
        relion_angle_psi_deg=np.asarray(metadata["angle_psi_deg"])[
            evaluation_indices.ravel()
        ],
        relion_origin_x_angstrom=np.asarray(metadata["origin_x_angstrom"])[
            evaluation_indices.ravel()
        ],
        relion_origin_y_angstrom=np.asarray(metadata["origin_y_angstrom"])[
            evaluation_indices.ravel()
        ],
        relion_pose_angle_deg=evaluation_poses.angle_deg,
        relion_pose_shift_y_px=evaluation_poses.shift_y_px,
        relion_pose_shift_x_px=evaluation_poses.shift_x_px,
        reference_prepared_index_zero_based=reference_indices,
        reference_image_id=np.asarray(metadata["image_id"])[reference_indices],
        reference_relion_pose_angle_deg=reference_poses.angle_deg.reshape(
            len(classes), args.reference_count
        ),
        reference_relion_pose_shift_y_px=reference_poses.shift_y_px.reshape(
            len(classes), args.reference_count
        ),
        reference_relion_pose_shift_x_px=reference_poses.shift_x_px.reshape(
            len(classes), args.reference_count
        ),
    )

    report = {
        "schema": "alignimg.re2dc-70s-pose-benchmark-data.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope_note": (
            "RELION classes and poses define an external alignment reference, not "
            "biological ground truth. Reference-build particles are disjoint from "
            "evaluation particles."
        ),
        "inputs": {
            "stack": args.stack,
            "prepared_star": args.prepared_star,
            "relion_star": args.relion_star,
            "source_particle_count": len(metadata["class_number"]),
            "pixel_size_angstrom": pixel_size,
            "image_shape": list(evaluation_images.shape[1:]),
        },
        "selection": {
            "relion_class_numbers": classes,
            "minimum_relion_confidence": args.minimum_confidence,
            "reference_particles_per_class": args.reference_count,
            "evaluation_particles_per_class": args.evaluation_count,
            "seed": args.seed,
            "reference_evaluation_disjoint": True,
        },
        "pose_conversion": {
            "angle_deg": "-rlnAnglePsi",
            "origin": (
                "positive rlnOriginX/YAngst converted with prepared-stack pixel size "
                "and rotated by the particle-to-reference angle before storage as a "
                "post-rotation AlignImg shift"
            ),
            "mirror": False,
            "center_convention": CENTER_CONVENTION,
        },
        "class_diagnostics": class_diagnostics,
        "artifacts": {
            "particles": particle_path,
            "references": reference_path,
            "particles_star": star_path,
            "truth": truth_path,
        },
    }
    write_json(manifest_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", type=Path, default=DEFAULT_STACK)
    parser.add_argument("--prepared-star", type=Path, default=DEFAULT_PREPARED_STAR)
    parser.add_argument("--relion-star", type=Path, default=DEFAULT_RELION_STAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--classes", type=int, nargs="+", default=[4, 7, 9])
    parser.add_argument("--reference-count", type=int, default=64)
    parser.add_argument("--evaluation-count", type=int, default=128)
    parser.add_argument("--minimum-confidence", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    report = build(args)
    print(
        "Prepared "
        f"{len(report['selection']['relion_class_numbers'])} classes x "
        f"{report['selection']['evaluation_particles_per_class']} evaluation particles"
    )
    for item in report["class_diagnostics"]:
        print(
            f"class {item['relion_class_number']}: "
            f"RELION-aligned NCC={item['relion_aligned_mean_particle_ncc']:.4f}, "
            f"raw NCC={item['raw_mean_particle_ncc']:.4f}"
        )
    print(f"Manifest saved to: {Path(f'{args.output}.json').resolve()}")


if __name__ == "__main__":
    main(parse_args())
