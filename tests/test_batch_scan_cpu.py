#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare experimental batched CPU angle scan against the scalar scan."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import alignimg.batch_cpu as batch
import alignimg.utils as au


def make_synthetic_pair(size=64):
    geo = au.get_geometry_context((size, size))
    y, x = np.indices((size, size), dtype=np.float32)
    ref = np.exp(-(((y - size * 0.42) ** 2) + ((x - size * 0.55) ** 2)) / (2.0 * 5.0 ** 2))
    ref += 0.65 * np.exp(-(((y - size * 0.64) ** 2) + ((x - size * 0.34) ** 2)) / (2.0 * 3.0 ** 2))
    ref = ref.astype(np.float32)

    img = au.transform_final_image(ref, geo, angle=-5.0, dy=2.0, dx=-3.0)
    mask = geo.get_circular_mask(diameter=size * 0.85)
    return img, ref, geo, mask


def top_key(cand):
    return (round(float(cand["angle"]), 6), round(float(cand["dy"]), 6), round(float(cand["dx"]), 6))


def main():
    img, ref, geo, mask = make_synthetic_pair()
    angles = np.arange(-10.0, 10.0 + 0.5, 0.5, dtype=np.float32)

    scalar = au.scan_joint_angles(img, ref, geo, angles, mask=mask)
    batched = batch.scan_joint_angles_batched_cpu(img, ref, geo, angles, mask=mask)

    s0 = scalar[0]
    b0 = batched[0]
    angle_diff = abs(float(au.angle_diff_deg(b0["angle"], s0["angle"])))
    dy_diff = abs(float(b0["dy"] - s0["dy"]))
    dx_diff = abs(float(b0["dx"] - s0["dx"]))
    score_diff = abs(float(b0["score"] - s0["score"]))

    scalar_top5 = {top_key(c) for c in scalar[:5]}
    batched_top5 = {top_key(c) for c in batched[:5]}
    top5_overlap = len(scalar_top5 & batched_top5)

    print("scalar best:", s0)
    print("batched best:", b0)
    print(
        "diffs:",
        f"angle={angle_diff:.6g}",
        f"dy={dy_diff:.6g}",
        f"dx={dx_diff:.6g}",
        f"score={score_diff:.6g}",
    )
    print(f"top-5 overlap: {top5_overlap}/5")

    ok = angle_diff <= 1e-6 and dy_diff <= 1e-6 and dx_diff <= 1e-6 and score_diff <= 1e-6
    if not ok:
        raise SystemExit("batched scan differs from scalar scan beyond tolerance")


def test_batch_scan_cpu():
    main()


if __name__ == "__main__":
    main()
