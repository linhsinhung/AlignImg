"""One bounded, immutable CPU table entry shared by Fourier operations."""

from functools import lru_cache

import numpy as np

from ._profiling import count


@lru_cache(maxsize=1)
def frequency_tables(size: int) -> tuple[np.ndarray, ...]:
    count("frequency_table_builds")
    frequency = np.fft.fftfreq(size)
    fy, fx = np.meshgrid(frequency, frequency, indexing="ij")
    by, bx = np.meshgrid(frequency * size, frequency * size, indexing="ij")
    indices = np.arange(size)
    phase = np.where((indices[:, None] + indices[None, :]) % 2 == 0, 1.0, -1.0)
    tables = (fy, fx, by, bx, phase)
    for table in tables:
        table.flags.writeable = False
    return tables
