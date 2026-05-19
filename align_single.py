#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
import numpy as np

from align_utils import get_geometry_context, transform_final_image, run_alignment_single_cpu, run_transform_single_cpu


def _demo_synthetic_stack(n=16, size=96, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.indices((size, size), dtype=np.float32)
    cy = (size - 1) / 2.0
    cx = (size - 1) / 2.0

    base = np.zeros((size, size), dtype=np.float32)
    for _ in range(8):
        ay = rng.uniform(-18, 18)
        ax = rng.uniform(-18, 18)
        s = rng.uniform(2.5, 6.0)
        amp = rng.uniform(0.6, 1.2)
        base += amp * np.exp(-(((y - (cy + ay)) ** 2 + (x - (cx + ax)) ** 2) / (2 * s * s)))

    geo = get_geometry_context((size, size))
    X = np.zeros((n, size, size), dtype=np.float32)
    for i in range(n):
        angle = rng.uniform(-25.0, 25.0)
        dy = rng.uniform(-6.0, 6.0)
        dx = rng.uniform(-6.0, 6.0)
        noisy = base + rng.normal(0.0, 0.08, size=(size, size)).astype(np.float32)
        X[i] = transform_final_image(noisy.astype(np.float32), geo, angle, dy, dx)

    initial_ref = np.mean(X, axis=0).astype(np.float32)
    return X, initial_ref


if __name__ == "__main__":
    X, initial_ref = _demo_synthetic_stack()
    t0 = time.perf_counter()
    final_ref, history_refs, params, meta = run_alignment_single_cpu(X, initial_ref, num_iterations=4, verbose=True)
    corrected = run_transform_single_cpu(X, params)
    elapsed = time.perf_counter() - t0

    print(f"final reference mean/std: {float(np.mean(final_ref)):.6f} / {float(np.std(final_ref)):.6f}")
    print(f"params shape: {params.shape}")
    print(f"corrected stack shape: {corrected.shape}")
    print(f"meta: {meta}")
    print(f"elapsed time: {elapsed:.3f}s")
