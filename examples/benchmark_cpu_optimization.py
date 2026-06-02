#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark CPU alignment optimization candidates.

Compares:
1. multicore Phase-3 global MAP-EM
2. warm-start refine with the scalar scan
3. warm-start refine with experimental batched local scan

The batched path is experimental and disabled in production defaults.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import alignimg.api as api
import alignimg.utils as au


def load_mrc_stack(path: Path, n: int) -> np.ndarray | None:
    try:
        import mrcfile
    except ImportError:
        return None

    if not path.exists():
        return None

    with mrcfile.mmap(path, permissive=True, mode="r") as mrc:
        arr = np.asarray(mrc.data)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(f"Expected 2D image or 3D stack, got {arr.shape}")
        end = arr.shape[0] if int(n) <= 0 else min(arr.shape[0], int(n))
        return np.asarray(arr[:end], dtype=np.float32).copy()


def make_synthetic_stack(n=64, size=64):
    geo = au.get_geometry_context((size, size))
    y, x = np.indices((size, size), dtype=np.float32)
    ref = np.exp(-(((y - size * 0.42) ** 2) + ((x - size * 0.58) ** 2)) / (2.0 * 5.0 ** 2))
    ref += 0.55 * np.exp(-(((y - size * 0.65) ** 2) + ((x - size * 0.35) ** 2)) / (2.0 * 3.0 ** 2))
    ref = ref.astype(np.float32)

    rng = np.random.default_rng(123)
    stack = np.empty((int(n), size, size), dtype=np.float32)
    for i in range(int(n)):
        angle = float(rng.uniform(-8.0, 8.0))
        dy = float(rng.integers(-3, 4))
        dx = float(rng.integers(-3, 4))
        stack[i] = au.transform_final_image(ref, geo, angle, dy, dx)
    return stack


def normalized_corr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    aa = a - np.mean(a)
    bb = b - np.mean(b)
    den = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb))) + 1e-8
    return float(np.sum(aa * bb) / den)


def summarize(name, elapsed, corrected_mean, params, meta, baseline_mean=None, baseline_params=None):
    weights = np.asarray(meta.get("last_weights", np.ones(params.shape[0])), dtype=np.float32)
    shift_mag = np.sqrt(params[:, 1].astype(np.float32) ** 2 + params[:, 2].astype(np.float32) ** 2)
    row = {
        "name": name,
        "elapsed_s": float(elapsed),
        "effective_n": float(meta.get("iterations", [{}])[-1].get("effective_n", np.sum(weights))),
        "weight_mean": float(np.mean(weights)),
        "shift_mean": float(np.mean(shift_mag)),
        "large_shift_gt20_count": int(np.sum(shift_mag > 20.0)),
        "corrected_mean_corr_vs_baseline": np.nan,
        "angle_abs_diff_mean": np.nan,
        "shift_abs_diff_mean": np.nan,
        "search_mode": meta.get("search_mode", ""),
        "use_batched_scan": bool(meta.get("config", {}).get("use_batched_scan", False)),
    }
    if baseline_mean is not None:
        row["corrected_mean_corr_vs_baseline"] = normalized_corr(corrected_mean, baseline_mean)
    if baseline_params is not None:
        angle_diff = np.abs(au.angle_diff_deg(params[:, 0], baseline_params[:, 0]))
        shift_diff = np.sqrt(
            (params[:, 1] - baseline_params[:, 1]) ** 2
            + (params[:, 2] - baseline_params[:, 2]) ** 2
        )
        row["angle_abs_diff_mean"] = float(np.mean(angle_diff))
        row["shift_abs_diff_mean"] = float(np.mean(shift_diff))
    return row


def run_case(name, X, ref, cfg, num_iterations, n_jobs, chunksize, initial_params=None, search_mode=None):
    t0 = time.perf_counter()
    final_ref, history, params, meta = api.run_alignment(
        X,
        ref,
        backend="multicore",
        algorithm="mapem",
        config=cfg,
        num_iterations=num_iterations,
        initial_params=initial_params,
        search_mode=search_mode,
        n_jobs=n_jobs,
        chunksize=chunksize,
        verbose=True,
    )
    corrected = api.run_transform(X, params, backend="multicore", algorithm="mapem", n_jobs=n_jobs)
    corrected_mean = np.mean(corrected, axis=0).astype(np.float32)
    elapsed = time.perf_counter() - t0
    return {
        "name": name,
        "final_ref": final_ref,
        "history": history,
        "params": params,
        "meta": meta,
        "corrected_mean": corrected_mean,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="test_align.mrcs")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--n-jobs", type=int, default=24)
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--global-iters", type=int, default=4)
    parser.add_argument("--refine-iters", type=int, default=1)
    args = parser.parse_args()

    X = load_mrc_stack(Path(args.data), args.n)
    if X is None:
        print("[benchmark] real-data stack unavailable; using synthetic stack")
        X = make_synthetic_stack(n=args.n, size=64)
    X = np.asarray(X, dtype=np.float32)
    init_ref = np.mean(X, axis=0).astype(np.float32)

    cfg_base = api.make_mapem_config(phase=3, weight_mode="sigmoid", lambda_shift=0.01, lambda_angle=0.0)
    cfg_batch = api.make_mapem_config(
        phase=3,
        weight_mode="sigmoid",
        lambda_shift=0.01,
        lambda_angle=0.0,
        use_batched_scan=True,
    )

    global_case = run_case(
        "multicore_phase3_global",
        X,
        init_ref,
        cfg_base,
        args.global_iters,
        args.n_jobs,
        args.chunksize,
    )
    refine_case = run_case(
        "warmstart_refine_scalar",
        X,
        global_case["final_ref"],
        cfg_base,
        args.refine_iters,
        args.n_jobs,
        args.chunksize,
        initial_params=global_case["params"],
        search_mode="refine",
    )
    batch_case = run_case(
        "warmstart_refine_batched",
        X,
        global_case["final_ref"],
        cfg_batch,
        args.refine_iters,
        args.n_jobs,
        args.chunksize,
        initial_params=global_case["params"],
        search_mode="refine",
    )

    rows = [
        summarize(
            global_case["name"],
            global_case["elapsed"],
            global_case["corrected_mean"],
            global_case["params"],
            global_case["meta"],
        ),
        summarize(
            refine_case["name"],
            refine_case["elapsed"],
            refine_case["corrected_mean"],
            refine_case["params"],
            refine_case["meta"],
            baseline_mean=refine_case["corrected_mean"],
            baseline_params=refine_case["params"],
        ),
        summarize(
            batch_case["name"],
            batch_case["elapsed"],
            batch_case["corrected_mean"],
            batch_case["params"],
            batch_case["meta"],
            baseline_mean=refine_case["corrected_mean"],
            baseline_params=refine_case["params"],
        ),
    ]

    print("\n=== CPU optimization benchmark ===")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
