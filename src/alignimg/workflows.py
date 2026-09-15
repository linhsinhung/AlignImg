"""AlignImg 2.x workflow-oriented public API."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from ._engine import run_soft_alignment_cpu, spectral_bootstrap
from ._fourier import soft_circular_mask
from ._transform import transform_stack
from ._profiling import profile_stage, profile_workflow
from .models import AlignmentConfig, AlignmentResult, PoseSet


def available_alignment_backends() -> dict[str, dict[str, Any]]:
    """Return availability of CPU, native CUDA, and CuPy alignment engines."""
    gpu_available = importlib.util.find_spec("alignimg_gpu") is not None
    result = {
        "cpu": {"available": True, "description": "CPU-authoritative soft Fourier engine."},
        "cuda": {"available": False, "description": "Native CUDA transform engine."},
        "cupy": {"available": False, "description": "CuPy fallback engine."},
    }
    if gpu_available:
        try:
            from alignimg_gpu import backend_status

            result.update(backend_status())
        except (ImportError, OSError, RuntimeError):
            pass
    result["auto"] = {
        "available": True,
        "description": "Select native CUDA, then CuPy, then CPU.",
    }
    result["gpu"] = {
        "available": result["cuda"]["available"] or result["cupy"]["available"],
        "description": "Compatibility alias selecting the best installed GPU engine.",
    }
    return result


def _load_gpu_function(name: str):
    try:
        module = __import__("alignimg_gpu", fromlist=[name])
        return getattr(module, name)
    except (ImportError, OSError, AttributeError) as error:
        raise RuntimeError(
            "The requested GPU backend requires alignimg-gpu and a compatible CUDA runtime."
        ) from error


@profile_stage("final_raw_average")
def _finalize_class_averages(
    result: AlignmentResult,
    images: np.ndarray,
    config: AlignmentConfig,
    backend: str,
) -> AlignmentResult:
    result.metadata["class_average_estimator"] = "soft_posterior_reference"
    if not config.apply_final_pose_to_raw:
        return result

    values = np.asarray(images, dtype=np.float32)
    reference_count = len(result.references)
    assignments = np.asarray(result.reference_assignments, dtype=np.int32)
    weights = np.asarray(result.inlier_weights, dtype=np.float64)

    if backend != "cpu":
        function = _load_gpu_function(f"_final_raw_class_averages_{backend}")
        averages, metadata = function(
            values,
            result.poses,
            assignments,
            weights,
            np.asarray(result.references, dtype=np.float32),
            config,
        )
        result._class_average_values = averages
        result.metadata.update(
            {
                "class_average_estimator": "final_map_pose_inlier_weighted_raw",
                **metadata,
            }
        )
        return result

    sums = np.zeros((reference_count, *values.shape[1:]), dtype=np.float64)
    total_weights = np.zeros(reference_count, dtype=np.float64)
    batch_size = int(config.batch_size or 512)

    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        batch_poses = PoseSet(
            result.poses.angle_deg[start:stop],
            result.poses.shift_y_px[start:stop],
            result.poses.shift_x_px[start:stop],
            result.poses.mirror[start:stop],
        )
        aligned = np.asarray(
            transform_images(values[start:stop], batch_poses, backend=backend),
            dtype=np.float32,
        )
        batch_assignments = assignments[start:stop]
        batch_weights = weights[start:stop]
        for reference_index in np.unique(batch_assignments):
            selected = batch_assignments == reference_index
            selected_weights = batch_weights[selected]
            sums[reference_index] += np.tensordot(
                selected_weights,
                aligned[selected].astype(np.float64),
                axes=(0, 0),
            )
            total_weights[reference_index] += float(np.sum(selected_weights))

    averages = np.asarray(result.references, dtype=np.float32).copy()
    nonempty = total_weights > 1e-8
    averages[nonempty] = (
        sums[nonempty] / total_weights[nonempty, None, None]
    ).astype(np.float32)
    averages[nonempty] *= soft_circular_mask(
        values.shape[-1], config.mask_radius, config.mask_soft_edge
    )
    result._class_average_values = averages
    result.metadata.update(
        {
            "class_average_estimator": "final_map_pose_inlier_weighted_raw",
            "class_average_transform_backend": "cpu",
            "class_average_batch_size": batch_size,
            "class_average_empty_components": np.flatnonzero(~nonempty),
        }
    )
    return result


def _dispatch(
    images: np.ndarray,
    references: np.ndarray,
    *,
    config: AlignmentConfig,
    class_priors: np.ndarray | None,
    initial_poses: PoseSet | None,
    workflow: str,
    backend: str,
) -> AlignmentResult:
    name = str(backend).strip().lower()
    if name == "cpu":
        return _finalize_class_averages(
            run_soft_alignment_cpu(
                images,
                references,
                config=config,
                class_priors=class_priors,
                initial_poses=initial_poses,
                workflow=workflow,
            ),
            images,
            config,
            name,
        )
    if name == "auto":
        status = available_alignment_backends()
        name = "cuda" if status["cuda"]["available"] else (
            "cupy" if status["cupy"]["available"] else "cpu"
        )
        if name == "cpu":
            return _finalize_class_averages(
                run_soft_alignment_cpu(
                    images,
                    references,
                    config=config,
                    class_priors=class_priors,
                    initial_poses=initial_poses,
                    workflow=workflow,
                ),
                images,
                config,
                name,
            )
    function_names = {
        "cuda": "run_soft_alignment_cuda",
        "cupy": "run_soft_alignment_cupy",
        "gpu": "run_soft_alignment_gpu",
    }
    if name not in function_names:
        raise ValueError("backend must be 'cpu', 'cuda', 'cupy', 'gpu', or 'auto'.")
    function = _load_gpu_function(function_names[name])
    return _finalize_class_averages(
        function(
            images,
            references,
            config=config,
            class_priors=class_priors,
            initial_poses=initial_poses,
            workflow=workflow,
        ),
        images,
        config,
        name,
    )


@profile_workflow
def reference_free_align(
    images: np.ndarray,
    *,
    n_components: int,
    config: AlignmentConfig | None = None,
    backend: str = "cpu",
) -> AlignmentResult:
    """Align particles without references or prior poses using K view components."""
    cfg = (config or AlignmentConfig.preset("reference_free")).normalized(
        workflow="reference_free"
    )
    initial_references, bootstrap_labels = spectral_bootstrap(images, n_components, cfg)
    result = _dispatch(
        images,
        initial_references,
        config=cfg,
        class_priors=None,
        initial_poses=None,
        workflow="reference_free",
        backend=backend,
    )
    result.metadata["bootstrap"] = "polar-harmonic-kmeans++-medoid"
    result.metadata["bootstrap_labels"] = bootstrap_labels
    return result


@profile_workflow
def align_to_references(
    images: np.ndarray,
    references: np.ndarray,
    *,
    class_priors: np.ndarray | None = None,
    config: AlignmentConfig | None = None,
    backend: str = "cpu",
) -> AlignmentResult:
    """Run global single- or multi-reference soft alignment."""
    cfg = (config or AlignmentConfig.preset("global_balanced")).normalized(
        workflow="global"
    )
    return _dispatch(
        images,
        references,
        config=cfg,
        class_priors=class_priors,
        initial_poses=None,
        workflow="global",
        backend=backend,
    )


@profile_workflow
def refine_alignment(
    images: np.ndarray,
    references: np.ndarray,
    initial_poses: PoseSet,
    *,
    class_priors: np.ndarray | None = None,
    config: AlignmentConfig | None = None,
    backend: str = "cpu",
) -> AlignmentResult:
    """Robustly refine poses, optionally allowing multi-reference reassignment."""
    cfg = config or AlignmentConfig.preset("refine")
    cfg = cfg.normalized(workflow="refine")
    return _dispatch(
        images,
        references,
        config=cfg,
        class_priors=class_priors,
        initial_poses=initial_poses,
        workflow="refine",
        backend=backend,
    )


def transform_images(
    images: np.ndarray,
    poses: PoseSet,
    *,
    backend: str = "cpu",
) -> np.ndarray:
    """Apply canonical poses to an image stack."""
    name = str(backend).strip().lower()
    if name == "cpu":
        return transform_stack(images, poses)
    if name == "auto":
        status = available_alignment_backends()
        name = "cuda" if status["cuda"]["available"] else (
            "cupy" if status["cupy"]["available"] else "cpu"
        )
        if name == "cpu":
            return transform_stack(images, poses)
    function_names = {
        "cuda": "transform_images_cuda",
        "cupy": "transform_images_cupy",
        "gpu": "transform_images_gpu",
    }
    if name not in function_names:
        raise ValueError("backend must be 'cpu', 'cuda', 'cupy', 'gpu', or 'auto'.")
    return _load_gpu_function(function_names[name])(images, poses)
