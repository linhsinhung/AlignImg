#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-core CPU backend for MAP-EM alignment.

This first parallel backend intentionally preserves the single-process MAP-EM
algorithm and only parallelizes the per-particle E-step inside each iteration.
Reference updates, inlier-weight estimation, and metadata aggregation remain in
this parent process.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import os

import numpy as np

import align_utils as au


def _align_particle_mapem_worker(args):
    """Run one MAP-EM particle E-step in a worker process."""
    (
        i,
        img,
        ref_match,
        prev_angle,
        schedule,
        cfg_dict,
        mask_diameter,
        mask_soft_edge,
    ) = args

    cfg = au.MAPEMConfig(**cfg_dict).normalized()
    geo = au.get_geometry_context(img.shape)
    mask = geo.get_circular_mask(diameter=mask_diameter, soft_edge=mask_soft_edge)

    params_i, aligned_i, best = au.align_one_mapem_cpu(
        img,
        ref_match,
        geo,
        prev_angle=float(prev_angle),
        schedule=schedule,
        cfg=cfg,
        mask=mask,
    )

    image_score = float(best.get("image_score", best.get("score", 0.0)))
    posterior_score = float(best.get("posterior_score", best.get("score", 0.0)))
    diagnostics_record = {
        "particle": int(i),
        "angle": float(params_i[0]),
        "dy": float(params_i[1]),
        "dx": float(params_i[2]),
        "posterior_score": float(posterior_score),
        "image_score": float(image_score),
        "translation_prior_score": float(best.get("translation_prior_score", 0.0)),
        "angle_prior_score": float(best.get("angle_prior_score", 0.0)),
        "n_evaluations": int(best.get("n_evaluations", 0)),
    }

    return i, params_i, aligned_i, image_score, posterior_score, diagnostics_record


def run_alignment_mapem_multicore(
    X,
    initial_ref,
    num_iterations=4,
    mask_diameter=None,
    config=None,
    verbose=True,
    n_jobs=None,
    chunksize=1,
):
    """Run robust MAP-EM alignment with a process-parallel E-step."""
    cfg = (config or au.MAPEMConfig()).normalized()

    X = np.asarray(X, dtype=np.float32)
    initial_ref = np.asarray(initial_ref, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected X shape (N,H,W), got {X.shape}")
    if initial_ref.ndim != 2:
        raise ValueError(f"Expected initial_ref shape (H,W), got {initial_ref.shape}")

    n = int(X.shape[0])
    geo = au.get_geometry_context(X.shape)
    # Build once in the parent to mirror the single backend's mask construction.
    geo.get_circular_mask(diameter=mask_diameter, soft_edge=cfg.mask_soft_edge)

    params = np.zeros((n, 4), dtype=np.float32)
    current_ref = au.apply_circular_mask(
        initial_ref,
        geo,
        diameter=mask_diameter,
        soft_edge=cfg.mask_soft_edge,
    )
    history_refs = [current_ref.copy()]

    if n_jobs is None:
        n_jobs = max(1, min(os.cpu_count() or 1, n))
    else:
        n_jobs = max(1, int(n_jobs))
    chunksize = max(1, int(chunksize))

    meta = {
        "backend": "multicore",
        "engine": "align-mapem-multicore",
        "algorithm": "mapem",
        "config": asdict(cfg),
        "iterations": [],
        "num_particles": n,
        "num_iterations": int(num_iterations),
        "mask_diameter": mask_diameter,
        "n_jobs": n_jobs,
        "chunksize": chunksize,
        "implemented": True,
    }

    weights = np.ones(n, dtype=np.float32)
    image_scores = np.zeros(n, dtype=np.float32)
    posterior_scores = np.zeros(n, dtype=np.float32)

    for it in range(int(num_iterations)):
        schedule = au.mapem_iter_schedule(it, int(num_iterations))
        if verbose:
            print(
                f"[multicore iter {it + 1}/{num_iterations}] "
                f"phase={cfg.phase} mode={schedule['mode']} "
                f"lp_sigma={schedule['lp_sigma']} weight={cfg.weight_mode} "
                f"n_jobs={n_jobs}"
            )

        ref_match = au.apply_lowpass_filter(current_ref, sigma=float(schedule["lp_sigma"]))
        aligned_stack = np.empty_like(X, dtype=np.float32)
        diagnostics = []
        cfg_dict = asdict(cfg)

        worker_args = [
            (
                i,
                X[i],
                ref_match,
                params[i, 0],
                schedule,
                cfg_dict,
                mask_diameter,
                cfg.mask_soft_edge,
            )
            for i in range(n)
        ]

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = executor.map(_align_particle_mapem_worker, worker_args, chunksize=chunksize)
            for i, params_i, aligned_i, image_score, posterior_score, diagnostics_record in results:
                params[i] = params_i
                aligned_stack[i] = aligned_i
                image_scores[i] = float(image_score)
                posterior_scores[i] = float(posterior_score)
                if i < int(cfg.diagnostics_n):
                    diagnostics.append(diagnostics_record)

        weights, weight_info = au.estimate_inlier_weights(posterior_scores, cfg)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-8:
            weights = np.ones(n, dtype=np.float32)
            weight_sum = float(n)

        new_ref = np.sum(aligned_stack * weights[:, None, None], axis=0) / weight_sum
        new_ref = au.apply_circular_mask(
            new_ref,
            geo,
            diameter=mask_diameter,
            soft_edge=cfg.mask_soft_edge,
        )

        if cfg.normalize_reference:
            new_ref = (new_ref - np.mean(new_ref)) / (np.std(new_ref) + 1e-8)

        current_ref = new_ref.astype(np.float32, copy=False)
        history_refs.append(current_ref.copy())

        meta["iterations"].append({
            "iteration": it,
            "schedule": dict(schedule),
            "image_score_mean": float(np.mean(image_scores)),
            "posterior_score_mean": float(np.mean(posterior_scores)),
            "weight_mean": float(np.mean(weights)),
            "weight_min": float(np.min(weights)),
            "weight_max": float(np.max(weights)),
            **weight_info,
            "diagnostics": diagnostics,
        })

    meta["last_weights"] = weights.astype(np.float32)
    meta["last_image_scores"] = image_scores.astype(np.float32)
    meta["last_posterior_scores"] = posterior_scores.astype(np.float32)
    return current_ref, history_refs, params, meta


def run_transform_multicore(X, params, n_jobs=None):
    """Apply final transforms; currently delegates to the single CPU backend."""
    del n_jobs
    return au.run_transform_single_cpu(X, params)


def _make_smoke_stack(n=4, size=32):
    """Create a tiny deterministic synthetic stack for the module smoke test."""
    geo = au.get_geometry_context((size, size))
    y, x = np.indices((size, size), dtype=np.float32)
    base = np.exp(-(((y - size * 0.45) ** 2) + ((x - size * 0.55) ** 2)) / (2.0 * 4.0 ** 2))
    base += 0.5 * np.exp(-(((y - size * 0.65) ** 2) + ((x - size * 0.35) ** 2)) / (2.0 * 2.5 ** 2))
    base = base.astype(np.float32)

    transforms = [(-6.0, 1.0, -1.0), (3.0, -1.0, 2.0), (8.0, 2.0, 1.0), (-2.0, 0.0, -2.0)]
    stack = np.empty((n, size, size), dtype=np.float32)
    for i in range(n):
        angle, dy, dx = transforms[i % len(transforms)]
        stack[i] = au.transform_final_image(base, geo, angle, dy, dx)
    return stack, base


if __name__ == "__main__":
    X, init_ref = _make_smoke_stack()
    cfg = au.MAPEMConfig(
        global_step=30.0,
        mid_range=6.0,
        mid_step=3.0,
        fine_range=1.5,
        fine_step=1.5,
    )

    single_ref, _, single_params, _ = au.run_alignment_mapem_cpu(
        X,
        init_ref,
        num_iterations=2,
        mask_diameter=28,
        config=cfg,
        verbose=False,
    )
    multi_ref, _, multi_params, _ = run_alignment_mapem_multicore(
        X,
        init_ref,
        num_iterations=2,
        mask_diameter=28,
        config=cfg,
        verbose=False,
        n_jobs=2,
    )

    corr = float(np.corrcoef(single_ref.ravel(), multi_ref.ravel())[0, 1])
    print("final_ref shape", multi_ref.shape)
    print("params shape", multi_params.shape)
    print("single params shape", single_params.shape)
    print("single final_ref mean/std", float(np.mean(single_ref)), float(np.std(single_ref)))
    print("multicore final_ref mean/std", float(np.mean(multi_ref)), float(np.std(multi_ref)))
    print("correlation between single and multicore final_ref", corr)
