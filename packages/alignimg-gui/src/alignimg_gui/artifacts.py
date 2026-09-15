"""MRC/NPZ/JSON IO for GUI runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import socket
from typing import Any
import warnings

import mrcfile
import numpy as np

import alignimg as ai
from alignimg import AlignmentResult, PoseSet
from alignimg._geometry import CENTER_CONVENTION


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def load_mrc_stack(
    path: str | Path, *, allow_single: bool = False
) -> tuple[np.ndarray, float | None]:
    with mrcfile.open(Path(path), permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32).copy()
        pixel_size = float(mrc.voxel_size.x)
    if data.ndim == 2 and allow_single:
        data = data[None]
    if data.ndim != 3 or len(data) == 0:
        raise ValueError(f"{path} must contain a non-empty 2D image stack.")
    if data.shape[1] != data.shape[2]:
        raise ValueError(f"{path} must contain square images.")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path} contains non-finite pixels.")
    return data, pixel_size if pixel_size > 0 else None


def inspect_mrc_stack(path: str | Path) -> dict[str, Any]:
    with mrcfile.open(Path(path), permissive=True) as mrc:
        shape = tuple(int(value) for value in mrc.data.shape)
        dtype = str(mrc.data.dtype)
        pixel_size = float(mrc.voxel_size.x)
    return {
        "shape": shape,
        "dtype": dtype,
        "pixel_size_angstrom": pixel_size if pixel_size > 0 else None,
        "file_size_bytes": Path(path).stat().st_size,
    }


def load_previous_result(
    path: str | Path, particle_count: int, image_size: int
) -> tuple[PoseSet, np.ndarray, np.ndarray | None]:
    with np.load(Path(path), allow_pickle=False) as saved:
        required = {"angle_deg", "shift_y_px", "shift_x_px", "mirror", "assignments"}
        missing = sorted(required.difference(saved.files))
        if missing:
            raise ValueError(f"previous result is missing arrays: {missing}")
        poses = PoseSet(
            saved["angle_deg"],
            saved["shift_y_px"],
            saved["shift_x_px"],
            saved["mirror"],
        )
        assignments = np.asarray(saved["assignments"], dtype=np.int32)
        responsibilities = (
            np.asarray(saved["responsibilities"], dtype=np.float32)
            if "responsibilities" in saved.files
            else None
        )
        stored_center = (
            str(np.asarray(saved["center_convention"]).item())
            if "center_convention" in saved.files
            else None
        )
    if len(poses) != particle_count or assignments.shape != (particle_count,):
        raise ValueError(
            "previous result particle count does not match the input stack."
        )
    if stored_center is None:
        warnings.warn(
            "Previous result has no center_convention; treating it as a pre-1.5 "
            "geometric-center result and converting its poses.",
            UserWarning,
            stacklevel=2,
        )
        poses = ai.convert_v1_4_poses_to_integer_center(poses, image_size)
    elif stored_center != CENTER_CONVENTION:
        raise ValueError(
            f"unsupported pose center convention {stored_center!r}; "
            f"expected {CENTER_CONVENTION!r}"
        )
    return poses, assignments, responsibilities


def save_result(
    run_directory: Path,
    result: AlignmentResult,
    *,
    pixel_size: float | None,
    reference_history: np.ndarray | None = None,
) -> dict[str, str]:
    references_path = run_directory / "references.mrcs"
    result_path = run_directory / "result.npz"
    with mrcfile.new(references_path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(result.class_averages, dtype=np.float32))
        if pixel_size is not None:
            mrc.voxel_size = pixel_size
    history = reference_history
    if history is None:
        history = (
            np.asarray(result.reference_history, dtype=np.float32)
            if result.reference_history
            else np.asarray(result.references[None], dtype=np.float32)
        )
    np.savez_compressed(
        result_path,
        alignimg_version=np.asarray(ai.__version__),
        center_convention=np.asarray(result.metadata["center_convention"]),
        reference_history=history,
        soft_references=np.asarray(result.references, dtype=np.float32),
        class_averages=np.asarray(result.class_averages, dtype=np.float32),
        angle_deg=result.poses.angle_deg,
        shift_y_px=result.poses.shift_y_px,
        shift_x_px=result.poses.shift_x_px,
        mirror=result.poses.mirror,
        assignments=result.reference_assignments,
        responsibilities=result.responsibilities,
        inlier_weights=result.inlier_weights,
        pose_entropy=result.pose_entropy,
        map_posterior=result.map_posterior,
    )
    return {"references": references_path.name, "result": result_path.name}


def base_report(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "alignimg-gui.pipeline.v2",
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "alignimg_version": ai.__version__,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "backends": ai.available_alignment_backends(),
        },
        "spec": spec,
    }


def load_result_bundle(
    report_path: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    artifact = report.get("artifacts", {}).get("result")
    if not artifact:
        raise ValueError("report does not contain a result artifact.")
    result_path = Path(artifact)
    if not result_path.is_absolute():
        result_path = report_path.parent / result_path
    with np.load(result_path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    return report, arrays
