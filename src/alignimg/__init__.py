"""Public package interface for AlignImg."""

from .models import AlignmentConfig, AlignmentResult, CandidateSet, PoseSet
from .compat import (
    convert_v1_4_poses_to_integer_center,
    poses_from_legacy_params,
    poses_to_legacy_params,
)
from .priors import make_class_priors
from .workflows import (
    align_to_references,
    available_alignment_backends,
    reference_free_align,
    refine_alignment,
    transform_images,
)

__version__ = "2.2.0"

__all__ = [
    "__version__",
    "AlignmentConfig",
    "AlignmentResult",
    "CandidateSet",
    "PoseSet",
    "make_class_priors",
    "align_to_references",
    "available_alignment_backends",
    "reference_free_align",
    "refine_alignment",
    "transform_images",
    "poses_from_legacy_params",
    "poses_to_legacy_params",
    "convert_v1_4_poses_to_integer_center",
]
