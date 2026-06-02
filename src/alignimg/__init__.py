"""Public package interface for AlignImg."""

from .api import (
    available_backends,
    make_mapem_config,
    run_alignment,
    run_transform,
)

__version__ = "0.1.0"

__all__ = [
    "available_backends",
    "make_mapem_config",
    "run_alignment",
    "run_transform",
]
