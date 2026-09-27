"""Frozen AlignImg 2.x workflow API contracts."""

from __future__ import annotations

from inspect import Parameter, signature

import numpy as np
import pytest

import alignimg as ai


def asymmetric_reference(size: int = 32) -> np.ndarray:
    y, x = np.indices((size, size), dtype=np.float32)
    image = np.exp(-((y - 10) ** 2 + (x - 19) ** 2) / 8.0)
    image += 0.6 * np.exp(-((y - 22) ** 2 + (x - 11) ** 2) / 4.0)
    return image.astype(np.float32)


def fast_config() -> ai.AlignmentConfig:
    return ai.AlignmentConfig(
        max_iterations=1,
        top_l=2,
        angle_samples=24,
        proposal_angles_per_reference=2,
        translation_range=1.0,
        halfset_diagnostics=False,
        center_references=False,
        batch_size=8,
    )


def parameter_contract(function) -> list[tuple[str, Parameter.kind]]:
    return [
        (name, value.kind) for name, value in signature(function).parameters.items()
    ]


def test_workflow_signatures_are_frozen():
    assert parameter_contract(ai.reference_free_align) == [
        ("images", Parameter.POSITIONAL_OR_KEYWORD),
        ("n_components", Parameter.KEYWORD_ONLY),
        ("config", Parameter.KEYWORD_ONLY),
        ("backend", Parameter.KEYWORD_ONLY),
    ]
    assert parameter_contract(ai.align_to_references) == [
        ("images", Parameter.POSITIONAL_OR_KEYWORD),
        ("references", Parameter.POSITIONAL_OR_KEYWORD),
        ("class_priors", Parameter.KEYWORD_ONLY),
        ("config", Parameter.KEYWORD_ONLY),
        ("backend", Parameter.KEYWORD_ONLY),
    ]
    assert parameter_contract(ai.refine_alignment) == [
        ("images", Parameter.POSITIONAL_OR_KEYWORD),
        ("references", Parameter.POSITIONAL_OR_KEYWORD),
        ("initial_poses", Parameter.POSITIONAL_OR_KEYWORD),
        ("class_priors", Parameter.KEYWORD_ONLY),
        ("config", Parameter.KEYWORD_ONLY),
        ("backend", Parameter.KEYWORD_ONLY),
    ]
    assert parameter_contract(ai.transform_images) == [
        ("images", Parameter.POSITIONAL_OR_KEYWORD),
        ("poses", Parameter.POSITIONAL_OR_KEYWORD),
        ("backend", Parameter.KEYWORD_ONLY),
    ]
    assert parameter_contract(ai.make_class_priors) == [
        ("assignments", Parameter.KEYWORD_ONLY),
        ("responsibilities", Parameter.KEYWORD_ONLY),
        ("n_components", Parameter.KEYWORD_ONLY),
        ("trust", Parameter.KEYWORD_ONLY),
    ]


def test_fourier_pipeline_defaults_are_frozen():
    config = ai.AlignmentConfig()
    assert config.candidate_scoring == "fourier"
    assert config.score_model == "fourier_ncc"
    assert config.reference_update == "fourier"


def test_global_and_refine_result_contracts():
    reference = asymmetric_reference()
    source = ai.PoseSet(
        angle_deg=np.asarray([15.0, -12.0], dtype=np.float32),
        shift_y_px=np.asarray([1.0, -1.0], dtype=np.float32),
        shift_x_px=np.asarray([-1.0, 1.0], dtype=np.float32),
        mirror=np.zeros(2, dtype=bool),
    )
    images = ai.transform_images(np.repeat(reference[None], 2, axis=0), source)
    result = ai.align_to_references(
        images, reference, config=fast_config(), backend="cpu"
    )

    assert isinstance(result, ai.AlignmentResult)
    assert result.references.shape == (1, 32, 32)
    assert result.responsibilities.shape == (2, 1)
    assert result.reference_assignments.shape == (2,)
    assert len(result.poses) == 2
    assert np.allclose(result.responsibilities, 1.0)

    refined = ai.refine_alignment(
        images,
        result.references,
        result.poses,
        config=fast_config(),
        backend="cpu",
    )
    assert isinstance(refined, ai.AlignmentResult)
    assert refined.responsibilities.shape == (2, 1)


def test_invalid_backend_is_rejected():
    reference = asymmetric_reference()
    with pytest.raises(ValueError, match="backend"):
        ai.align_to_references(
            reference[None], reference, config=fast_config(), backend="single"
        )
