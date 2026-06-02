"""Public package interface for AlignImg."""

from .api import (
    MAPEMConfig,
    available_backends,
    make_mapem_config,
    run_alignment,
    run_transform,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "MAPEMConfig",
    "available_backends",
    "make_mapem_config",
    "run_alignment",
    "run_transform",
]
