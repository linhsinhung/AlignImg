"""Optional native-CUDA and CuPy backends for AlignImg."""

from .backend import (
    _final_raw_class_averages_cuda as _final_raw_class_averages_cuda,
    _final_raw_class_averages_cupy as _final_raw_class_averages_cupy,
    _final_raw_class_averages_gpu as _final_raw_class_averages_gpu,
    backend_status,
    run_soft_alignment_cuda,
    run_soft_alignment_cupy,
    run_soft_alignment_gpu,
    transform_images_cuda,
    transform_images_cupy,
    transform_images_gpu,
)

__version__ = "2.2.0"
__all__ = [
    "backend_status",
    "run_soft_alignment_cuda",
    "run_soft_alignment_cupy",
    "run_soft_alignment_gpu",
    "transform_images_cuda",
    "transform_images_cupy",
    "transform_images_gpu",
]
