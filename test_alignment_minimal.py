#!/usr/bin/env python3
"""Minimal staged synthetic alignment benchmark.

This script intentionally avoids the production reference-building pipeline. It
uses a single asymmetric synthetic template and then validates transform and
estimator conventions one stage at a time.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def _install_minimal_scipy_fallback() -> None:
    """Install the tiny scipy surface align_utils needs when scipy is absent."""
    if importlib.util.find_spec("scipy") is not None:
        return

    scipy_mod = types.ModuleType("scipy")
    fft_mod = types.ModuleType("scipy.fft")
    ndimage_mod = types.ModuleType("scipy.ndimage")
    fft_mod.fft2 = np.fft.fft2
    fft_mod.ifft2 = np.fft.ifft2
    fft_mod.fftshift = np.fft.fftshift

    def center_of_mass(arr):
        data = np.asarray(arr, dtype=np.float64)
        total = float(np.sum(data))
        if abs(total) <= 1e-12:
            return tuple((np.array(data.shape) - 1.0) / 2.0)
        coords = np.indices(data.shape, dtype=np.float64)
        return tuple(float(np.sum(coords[i] * data) / total) for i in range(data.ndim))

    ndimage_mod.center_of_mass = center_of_mass
    scipy_mod.fft = fft_mod
    scipy_mod.ndimage = ndimage_mod
    sys.modules.setdefault("scipy", scipy_mod)
    sys.modules.setdefault("scipy.fft", fft_mod)
    sys.modules.setdefault("scipy.ndimage", ndimage_mod)


def _install_minimal_cv2_fallback() -> None:
    """Install a small cv2-compatible shim sufficient for this benchmark."""
    if importlib.util.find_spec("cv2") is not None:
        return

    cv2_mod = types.ModuleType("cv2")
    cv2_mod.INTER_LINEAR = 1
    cv2_mod.LINE_AA = 16
    cv2_mod.WARP_FILL_OUTLIERS = 8

    def _bilinear_sample(img, x, y):
        h, w = img.shape
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = x0 + 1
        y1 = y0 + 1
        valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
        x0c = np.clip(x0, 0, w - 1)
        x1c = np.clip(x1, 0, w - 1)
        y0c = np.clip(y0, 0, h - 1)
        y1c = np.clip(y1, 0, h - 1)
        wx = x - x0
        wy = y - y0
        out = (
            (1 - wx) * (1 - wy) * img[y0c, x0c]
            + wx * (1 - wy) * img[y0c, x1c]
            + (1 - wx) * wy * img[y1c, x0c]
            + wx * wy * img[y1c, x1c]
        )
        return np.where(valid, out, 0).astype(img.dtype, copy=False)

    def getRotationMatrix2D(center, angle, scale):
        cx, cy = center
        alpha = float(scale) * math.cos(math.radians(angle))
        beta = float(scale) * math.sin(math.radians(angle))
        return np.array(
            [[alpha, beta, (1 - alpha) * cx - beta * cy], [-beta, alpha, beta * cx + (1 - alpha) * cy]],
            dtype=np.float32,
        )

    def warpAffine(img, M, dsize, flags=1):
        width, height = dsize
        yy, xx = np.indices((height, width), dtype=np.float32)
        homog = np.vstack([M, [0, 0, 1]]).astype(np.float64)
        inv = np.linalg.inv(homog)[:2]
        src_x = inv[0, 0] * xx + inv[0, 1] * yy + inv[0, 2]
        src_y = inv[1, 0] * xx + inv[1, 1] * yy + inv[1, 2]
        return _bilinear_sample(np.asarray(img), src_x, src_y)

    def GaussianBlur(img, ksize, sigma):
        if sigma <= 0:
            return img
        radius = max(1, int(math.ceil(3.0 * float(sigma))))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-0.5 * (x / float(sigma)) ** 2)
        kernel /= np.sum(kernel)
        tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, np.asarray(img, dtype=np.float32))
        return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, tmp).astype(np.asarray(img).dtype, copy=False)

    def line(img, pt1, pt2, color, thickness=1, lineType=16):
        x1, y1 = pt1
        x2, y2 = pt2
        n = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
        rr = max(0, int(math.ceil(thickness / 2)))
        for t in np.linspace(0.0, 1.0, n):
            x = int(round(x1 + t * (x2 - x1)))
            y = int(round(y1 + t * (y2 - y1)))
            y0, y1c = max(0, y - rr), min(img.shape[0], y + rr + 1)
            x0, x1c = max(0, x - rr), min(img.shape[1], x + rr + 1)
            img[y0:y1c, x0:x1c] = color
        return img

    def linearPolar(src, center, maxRadius, flags=8):
        h, w = src.shape
        cy, cx = float(center[1]), float(center[0])
        out_y, out_x = np.indices((h, w), dtype=np.float32)
        theta = out_y / float(h) * 2.0 * math.pi
        radius = out_x / max(float(w - 1), 1.0) * float(maxRadius)
        src_x = cx + radius * np.cos(theta)
        src_y = cy + radius * np.sin(theta)
        return _bilinear_sample(np.asarray(src), src_x, src_y)

    def imwrite(path, img):
        # Dependency-free binary PGM payload. The extension may be .png, but the
        # overview remains readable by tools that inspect the file signature.
        arr = np.asarray(img, dtype=np.uint8)
        with open(path, "wb") as f:
            f.write(f"P5\n{arr.shape[1]} {arr.shape[0]}\n255\n".encode("ascii"))
            f.write(arr.tobytes())
        return True

    cv2_mod.getRotationMatrix2D = getRotationMatrix2D
    cv2_mod.warpAffine = warpAffine
    cv2_mod.GaussianBlur = GaussianBlur
    cv2_mod.line = line
    cv2_mod.linearPolar = linearPolar
    cv2_mod.imwrite = imwrite
    sys.modules.setdefault("cv2", cv2_mod)


_install_minimal_scipy_fallback()
_install_minimal_cv2_fallback()
cv2 = importlib.import_module("cv2")

from align_utils import (
    apply_circular_mask,
    apply_lowpass_filter,
    get_best_translation_fft,
    get_coarse_angle_fourier_mellin,
    get_geometry_context,
    normalize_angle,
    rotate_image,
    shift_image,
    transform_final_image,
)


@dataclass
class ResultRow:
    test_name: str
    applied_angle: float
    applied_dy: float
    applied_dx: float
    ideal_angle: float
    ideal_dy: float
    ideal_dx: float
    estimated_angle: float
    estimated_dy: float
    estimated_dx: float
    angle_error_deg: float
    dy_error_px: float
    dx_error_px: float
    corr: float
    nrmse: float
    passed: bool
    notes: str


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64) - float(np.mean(a))
    bb = np.asarray(b, dtype=np.float64) - float(np.mean(b))
    denom = math.sqrt(float(np.sum(aa * aa)) * float(np.sum(bb * bb)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(aa * bb) / denom)


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    rmse = math.sqrt(float(np.mean((aa - bb) ** 2)))
    scale = float(np.max(bb) - np.min(bb))
    if scale <= 1e-12:
        scale = float(np.std(bb)) + 1e-12
    return float(rmse / scale)


def angle_error(est: float, ideal: float) -> float:
    return abs(normalize_angle(float(est) - float(ideal)))


def normalize_template(img: np.ndarray, geo) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    img = apply_circular_mask(img, geo, diameter=geo.max_radius * 1.75, soft_edge=6)
    img -= float(np.mean(img))
    std = float(np.std(img))
    if std > 1e-12:
        img /= std
    return img.astype(np.float32, copy=False)


def add_gaussian(img: np.ndarray, cy: float, cx: float, sy: float, sx: float, amp: float) -> None:
    y, x = np.indices(img.shape, dtype=np.float32)
    img += amp * np.exp(-0.5 * (((y - cy) / sy) ** 2 + ((x - cx) / sx) ** 2)).astype(np.float32)


def make_asymmetric_template(size: int) -> np.ndarray:
    geo = get_geometry_context((size, size))
    img = np.zeros((size, size), dtype=np.float32)

    # Mixed-polarity, intentionally non-symmetric landmarks.
    add_gaussian(img, 0.30 * size, 0.36 * size, 5.0, 7.0, 2.8)  # large positive blob
    add_gaussian(img, 0.72 * size, 0.69 * size, 2.2, 2.8, 1.8)  # small positive blob
    add_gaussian(img, 0.60 * size, 0.26 * size, 4.0, 3.0, -2.2)  # negative blob

    # Off-center elongated diagonal bar.
    bar = np.zeros_like(img)
    p1 = (int(0.54 * size), int(0.18 * size))
    p2 = (int(0.85 * size), int(0.42 * size))
    cv2.line(bar, p1, p2, color=1.6, thickness=max(3, size // 24), lineType=cv2.LINE_AA)
    img += apply_lowpass_filter(bar, sigma=0.8)

    # L-shaped feature with unequal arms.
    lshape = np.zeros_like(img)
    cv2.line(
        lshape,
        (int(0.16 * size), int(0.78 * size)),
        (int(0.42 * size), int(0.78 * size)),
        color=1.2,
        thickness=max(2, size // 32),
        lineType=cv2.LINE_AA,
    )
    cv2.line(
        lshape,
        (int(0.16 * size), int(0.78 * size)),
        (int(0.16 * size), int(0.58 * size)),
        color=1.2,
        thickness=max(2, size // 32),
        lineType=cv2.LINE_AA,
    )
    img += apply_lowpass_filter(lshape, sigma=0.6)

    # A small notch/dot pair to break residual accidental symmetries.
    img[int(0.22 * size) : int(0.27 * size), int(0.76 * size) : int(0.81 * size)] -= 1.4
    img[int(0.84 * size), int(0.57 * size)] += 2.0

    return normalize_template(img, geo)


def add_noise(img: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    if noise <= 0:
        return img.astype(np.float32, copy=True)
    noisy = img + rng.normal(0.0, noise * float(np.std(img)), img.shape).astype(np.float32)
    return noisy.astype(np.float32, copy=False)


def ideal_correction_for_rotate_then_shift(applied_angle: float, applied_dy: float, applied_dx: float) -> tuple[float, float, float]:
    """Inverse for transformed = T(shift) after R(angle), applied as R(inv_angle) then T(inv_shift').

    transform_final_image applies rotation first and then image-space shift. To
    invert T(s) * R(a), the equivalent R(-a) then T(s') has s' = -R(-a) s.
    """
    ideal_angle = normalize_angle(-applied_angle)
    theta = math.radians(ideal_angle)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    # OpenCV/image coordinates: x' = cos*x + sin*y, y' = -sin*x + cos*y for positive angle.
    rot_dx = cos_t * applied_dx + sin_t * applied_dy
    rot_dy = -sin_t * applied_dx + cos_t * applied_dy
    return ideal_angle, -rot_dy, -rot_dx


def evaluate(test_name: str, gt: np.ndarray, corrected: np.ndarray, applied_angle: float, applied_dy: float, applied_dx: float,
             ideal_angle: float, ideal_dy: float, ideal_dx: float, estimated_angle: float, estimated_dy: float, estimated_dx: float,
             pass_condition: bool, notes: str, corr_threshold: float = 0.98) -> ResultRow:
    corr = normalized_correlation(corrected, gt)
    err = nrmse(corrected, gt)
    return ResultRow(
        test_name=test_name,
        applied_angle=float(applied_angle),
        applied_dy=float(applied_dy),
        applied_dx=float(applied_dx),
        ideal_angle=float(ideal_angle),
        ideal_dy=float(ideal_dy),
        ideal_dx=float(ideal_dx),
        estimated_angle=float(estimated_angle),
        estimated_dy=float(estimated_dy),
        estimated_dx=float(estimated_dx),
        angle_error_deg=float(angle_error(estimated_angle, ideal_angle)),
        dy_error_px=float(estimated_dy - ideal_dy),
        dx_error_px=float(estimated_dx - ideal_dx),
        corr=float(corr),
        nrmse=float(err),
        passed=bool(pass_condition and corr > corr_threshold),
        notes=notes,
    )


def write_failure_png(path: Path, gt: np.ndarray, transformed: np.ndarray, corrected: np.ndarray) -> None:
    def scale(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        lo, hi = np.percentile(x, [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        return np.clip((x - lo) / (hi - lo), 0, 1)

    diff = corrected - gt
    panels = [scale(gt), scale(transformed), scale(corrected), scale(np.abs(diff))]
    overview = np.concatenate(panels, axis=1)
    cv2.imwrite(str(path), (overview * 255).astype(np.uint8))


def brute_force_rotation(img: np.ndarray, ref: np.ndarray, geo, step: float = 2.0) -> tuple[float, float]:
    best_angle = 0.0
    best_corr = -float("inf")
    for cand in np.arange(-180.0, 180.0, step):
        rotated = rotate_image(img, geo, float(cand))
        corr = normalized_correlation(rotated, ref)
        if corr > best_corr:
            best_corr = corr
            best_angle = float(cand)
    return normalize_angle(best_angle), float(best_corr)


def append_result(rows: list[ResultRow], row: ResultRow, output_dir: Path, gt: np.ndarray, transformed: np.ndarray, corrected: np.ndarray) -> None:
    rows.append(row)
    if not row.passed:
        safe = f"{len(rows):03d}_{row.test_name}_a{row.applied_angle:g}_dy{row.applied_dy:g}_dx{row.applied_dx:g}.png".replace("-", "m")
        write_failure_png(output_dir / safe, gt, transformed, corrected)
        print(
            f"FAIL {row.test_name}: angle={row.applied_angle:g}, dy={row.applied_dy:g}, dx={row.applied_dx:g}, "
            f"corr={row.corr:.4f}, nrmse={row.nrmse:.4f}, notes={row.notes}"
        )


def run_transform_inverse(gt: np.ndarray, geo, rows: list[ResultRow], output_dir: Path) -> None:
    cases = [(0, 0, 0), (15, 0, 0), (-37, 0, 0), (0, 5, -7), (23, 4, -6), (-71, -5, 8), (135, 7, 3)]
    for angle, dy, dx in cases:
        transformed = shift_image(rotate_image(gt, geo, angle), geo, dy, dx)
        ideal_angle, ideal_dy, ideal_dx = ideal_correction_for_rotate_then_shift(angle, dy, dx)
        corrected = transform_final_image(transformed, geo, ideal_angle, ideal_dy, ideal_dx)
        row = evaluate(
            "transform_inverse_test", gt, corrected, angle, dy, dx,
            ideal_angle, ideal_dy, ideal_dx, ideal_angle, ideal_dy, ideal_dx,
            True, "analytic inverse via transform_final_image",
        )
        append_result(rows, row, output_dir, gt, transformed, corrected)


def run_translation_only(gt: np.ndarray, geo, rows: list[ResultRow], output_dir: Path) -> None:
    for dy, dx in [(0, 0), (5, -7), (-9, 4), (11, 12)]:
        transformed = shift_image(gt, geo, dy, dx)
        est_dy, est_dx, score = get_best_translation_fft(transformed, gt, geo)
        ideal_angle, ideal_dy, ideal_dx = 0.0, -dy, -dx
        corrected = transform_final_image(transformed, geo, 0.0, est_dy, est_dx)
        row = evaluate(
            "translation_only_oracle_test", gt, corrected, 0, dy, dx,
            ideal_angle, ideal_dy, ideal_dx, 0.0, est_dy, est_dx,
            abs(est_dy - ideal_dy) <= 1 and abs(est_dx - ideal_dx) <= 1,
            f"fft_score={score:.6g}; estimated correction shift vs ideal",
        )
        append_result(rows, row, output_dir, gt, transformed, corrected)


def run_rotation_only_bruteforce(gt: np.ndarray, geo, rows: list[ResultRow], output_dir: Path, step: float = 2.0) -> None:
    for angle in [0, 12, -34, 77, -123, 179]:
        transformed = rotate_image(gt, geo, angle)
        best_angle, best_corr = brute_force_rotation(transformed, gt, geo, step=step)
        ideal_angle = normalize_angle(-angle)
        corrected = transform_final_image(transformed, geo, best_angle, 0.0, 0.0)
        row = evaluate(
            "rotation_only_bruteforce_test", gt, corrected, angle, 0, 0,
            ideal_angle, 0, 0, best_angle, 0, 0,
            angle_error(best_angle, ideal_angle) <= step + 0.25,
            f"bruteforce_step={step:g}; best_corr={best_corr:.6f}",
        )
        append_result(rows, row, output_dir, gt, transformed, corrected)


def run_rotation_only_fm(gt: np.ndarray, geo, rows: list[ResultRow], output_dir: Path, step: float = 2.0) -> None:
    for angle in [0, 12, -34, 77, -123, 179]:
        transformed = rotate_image(gt, geo, angle)
        fm_angle = normalize_angle(get_coarse_angle_fourier_mellin(transformed, gt, geo))
        brute_angle, _ = brute_force_rotation(transformed, gt, geo, step=step)
        ideal_angle = normalize_angle(-angle)
        corrected = transform_final_image(transformed, geo, fm_angle, 0.0, 0.0)
        fm_to_brute = angle_error(fm_angle, brute_angle)
        row = evaluate(
            "rotation_only_fourier_mellin_test", gt, corrected, angle, 0, 0,
            ideal_angle, 0, 0, fm_angle, 0, 0,
            angle_error(fm_angle, ideal_angle) <= max(step, 360.0 / geo.H) + 1.0,
            f"bruteforce_angle={brute_angle:.3f}; fm_vs_bruteforce_error={fm_to_brute:.3f}",
        )
        append_result(rows, row, output_dir, gt, transformed, corrected)


def run_rotation_translation_oracle(gt: np.ndarray, geo, rows: list[ResultRow], output_dir: Path, step: float = 2.0) -> None:
    for angle, dy, dx in [(17, 5, -7), (-41, -8, 6), (93, 7, 9)]:
        transformed = shift_image(rotate_image(gt, geo, angle), geo, dy, dx)
        ideal_angle, ideal_dy, ideal_dx = ideal_correction_for_rotate_then_shift(angle, dy, dx)

        # Known ideal correction angle, estimate translation after rotation.
        rotated = rotate_image(transformed, geo, ideal_angle)
        est_dy, est_dx, score = get_best_translation_fft(rotated, gt, geo)
        corrected = transform_final_image(transformed, geo, ideal_angle, est_dy, est_dx)
        row = evaluate(
            "rotation_translation_oracle_angle_test", gt, corrected, angle, dy, dx,
            ideal_angle, ideal_dy, ideal_dx, ideal_angle, est_dy, est_dx,
            abs(est_dy - ideal_dy) <= 1.5 and abs(est_dx - ideal_dx) <= 1.5,
            f"known angle; fft_score={score:.6g}",
        )
        append_result(rows, row, output_dir, gt, transformed, corrected)

        # Estimate angle by image-space brute force, then estimate translation.
        best_angle, best_corr = brute_force_rotation(transformed, gt, geo, step=step)
        rotated = rotate_image(transformed, geo, best_angle)
        est_dy, est_dx, score = get_best_translation_fft(rotated, gt, geo)
        corrected = transform_final_image(transformed, geo, best_angle, est_dy, est_dx)
        row = evaluate(
            "rotation_translation_bruteforce_angle_test", gt, corrected, angle, dy, dx,
            ideal_angle, ideal_dy, ideal_dx, best_angle, est_dy, est_dx,
            angle_error(best_angle, ideal_angle) <= step + 0.25 and abs(est_dy - ideal_dy) <= 2.0 and abs(est_dx - ideal_dx) <= 2.0,
            f"bruteforce_step={step:g}; rotation_corr_before_translation={best_corr:.6f}; fft_score={score:.6g}",
        )
        append_result(rows, row, output_dir, gt, transformed, corrected)


def run_noise_sweep(gt: np.ndarray, geo, rows: list[ResultRow], output_dir: Path, step: float = 2.0) -> None:
    rng = np.random.default_rng(12345)
    angle, dy, dx = 23.0, 4.0, -6.0
    clean_transformed = shift_image(rotate_image(gt, geo, angle), geo, dy, dx)
    ideal_angle, ideal_dy, ideal_dx = ideal_correction_for_rotate_then_shift(angle, dy, dx)
    for noise in [0, 0.05, 0.1, 0.2, 0.35]:
        transformed = add_noise(clean_transformed, noise, rng)

        # Brute-force angle then FFT shift, deliberately simple staged estimator.
        best_angle, best_corr = brute_force_rotation(transformed, gt, geo, step=step)
        rotated = rotate_image(transformed, geo, best_angle)
        est_dy, est_dx, score = get_best_translation_fft(rotated, gt, geo)
        corrected = transform_final_image(transformed, geo, best_angle, est_dy, est_dx)
        corr = normalized_correlation(corrected, gt)
        pass_condition = corr > (0.98 if noise == 0 else 0.80)
        row = evaluate(
            "noise_sweep_test", gt, corrected, angle, dy, dx,
            ideal_angle, ideal_dy, ideal_dx, best_angle, est_dy, est_dx,
            pass_condition,
            f"noise={noise:g}; angle_by_bruteforce; pre_translation_corr={best_corr:.6f}; fft_score={score:.6g}",
            corr_threshold=(0.98 if noise == 0 else 0.80),
        )
        append_result(rows, row, output_dir, gt, transformed, corrected)


def write_csv(rows: list[ResultRow], output_dir: Path) -> Path:
    csv_path = output_dir / "minimal_alignment_tests.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return csv_path


def summarize(rows: list[ResultRow], csv_path: Path) -> None:
    print(f"\nWrote {csv_path}")
    print(f"Rows: {len(rows)}; passed: {sum(r.passed for r in rows)}; failed: {sum(not r.passed for r in rows)}")
    first_fail = next((r for r in rows if not r.passed), None)
    if first_fail is None:
        print("First failing stage: none")
    else:
        print(f"First failing stage: {first_fail.test_name} ({first_fail.notes})")
    for name in dict.fromkeys(r.test_name for r in rows):
        group = [r for r in rows if r.test_name == name]
        print(
            f"{name}: pass={sum(r.passed for r in group)}/{len(group)}, "
            f"median_corr={np.median([r.corr for r in group]):.4f}, "
            f"median_angle_err={np.median([r.angle_error_deg for r in group]):.3f}, "
            f"median_abs_shift_err=({np.median([abs(r.dy_error_px) for r in group]):.3f}, "
            f"{np.median([abs(r.dx_error_px) for r in group]):.3f})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=96, help="Synthetic square image size (default: 96).")
    parser.add_argument("--output-dir", type=Path, default=Path("minimal_report"), help="Directory for CSV and failure PNGs.")
    parser.add_argument("--angle-step", type=float, default=2.0, help="Brute-force rotation grid step in degrees.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    geo = get_geometry_context((args.size, args.size))
    gt = make_asymmetric_template(args.size)

    rows: list[ResultRow] = []
    run_transform_inverse(gt, geo, rows, args.output_dir)
    run_translation_only(gt, geo, rows, args.output_dir)
    run_rotation_only_bruteforce(gt, geo, rows, args.output_dir, step=args.angle_step)
    run_rotation_only_fm(gt, geo, rows, args.output_dir, step=args.angle_step)
    run_rotation_translation_oracle(gt, geo, rows, args.output_dir, step=args.angle_step)
    run_noise_sweep(gt, geo, rows, args.output_dir, step=args.angle_step)

    csv_path = write_csv(rows, args.output_dir)
    summarize(rows, csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
