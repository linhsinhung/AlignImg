"""Select an installed GPU package when available, else use the source fallback."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import sys


if find_spec("alignimg_gpu") is None:
    gpu_source = Path(__file__).resolve().parents[1] / "packages/alignimg-gpu/src"
    sys.path.insert(0, str(gpu_source))
