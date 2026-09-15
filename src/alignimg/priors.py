"""Utilities for turning external classification feedback into class priors."""

from __future__ import annotations

import numpy as np


def make_class_priors(
    *,
    assignments: np.ndarray | None = None,
    responsibilities: np.ndarray | None = None,
    n_components: int | None = None,
    trust: float = 1.0,
) -> np.ndarray:
    """Return normalized ``(N, K)`` priors from hard or soft class feedback.

    Exactly one of ``assignments`` and ``responsibilities`` must be supplied.
    ``trust=1`` preserves the supplied feedback, while smaller values mix it
    with a uniform prior so alignment may correct external assignments.
    """
    if (assignments is None) == (responsibilities is None):
        raise ValueError(
            "exactly one of assignments and responsibilities must be provided."
        )
    trust = float(trust)
    if not np.isfinite(trust) or not 0.0 <= trust <= 1.0:
        raise ValueError("trust must be finite and in the range [0, 1].")

    if assignments is not None:
        labels = np.asarray(assignments)
        if labels.ndim != 1 or len(labels) == 0:
            raise ValueError("assignments must be a non-empty one-dimensional array.")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("assignments must contain integer class labels.")
        if n_components is None:
            raise ValueError("n_components is required with assignments.")
        component_count = int(n_components)
        if component_count < 1:
            raise ValueError("n_components must be positive.")
        labels = labels.astype(np.int64, copy=False)
        if np.any(labels < 0) or np.any(labels >= component_count):
            raise ValueError("assignments must be in the range [0, n_components).")
        priors = np.zeros((len(labels), component_count), dtype=np.float32)
        priors[np.arange(len(labels)), labels] = 1.0
    else:
        priors = np.asarray(responsibilities, dtype=np.float32)
        if priors.ndim != 2 or priors.shape[0] == 0 or priors.shape[1] == 0:
            raise ValueError("responsibilities must have non-empty shape (N, K).")
        if n_components is not None and int(n_components) != priors.shape[1]:
            raise ValueError(
                "n_components must match responsibilities.shape[1] when provided."
            )
        if not np.all(np.isfinite(priors)) or np.any(priors < 0.0):
            raise ValueError(
                "responsibilities must contain finite non-negative values."
            )
        row_sums = priors.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise ValueError("every responsibilities row must have a positive sum.")
        priors = priors / row_sums
        component_count = priors.shape[1]

    uniform = np.float32((1.0 - trust) / component_count)
    return (np.float32(trust) * priors + uniform).astype(np.float32)
