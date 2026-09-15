"""Minimal public import smoke tests."""

from __future__ import annotations

import alignimg


def test_public_package_imports():
    expected_public = [
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

    assert alignimg.__version__ == "2.2.0"
    assert alignimg.__all__ == expected_public
    for name in expected_public:
        assert hasattr(alignimg, name)



def test_legacy_0_2_api_is_not_exported():
    for name in (
        "MAPEMConfig",
        "available_backends",
        "make_mapem_config",
        "run_alignment",
        "run_transform",
    ):
        assert not hasattr(alignimg, name)
