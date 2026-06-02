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
from multiprocessing import shared_memory
import os
import time

import numpy as np

from . import _utils as au


_SHM_X = None
_SHM_REF = None
_SHARED_X = None
_SHARED_REF = None


def _create_float32_shared_array(array):
    """Create shared memory and copy a float32 array into it."""
    array_c = np.ascontiguousarray(array, dtype=np.float32)
    shm = shared_memory.SharedMemory(create=True, size=array_c.nbytes)
    try:
        shared_array = np.ndarray(array_c.shape, dtype=np.float32, buffer=shm.buf)
        shared_array[:] = array_c
    except Exception:
        shm.close()
        shm.unlink()
        raise
    return shm, array_c.shape


def _init_shared_mapem_worker(x_shm_name, x_shape, ref_shm_name, ref_shape):
    """Attach worker process globals to parent-created shared memory blocks."""
    global _SHM_X, _SHM_REF, _SHARED_X, _SHARED_REF

    _SHM_X = shared_memory.SharedMemory(name=x_shm_name)
    _SHARED_X = np.ndarray(x_shape, dtype=np.float32, buffer=_SHM_X.buf)
    _SHM_REF = shared_memory.SharedMemory(name=ref_shm_name)
    _SHARED_REF = np.ndarray(ref_shape, dtype=np.float32, buffer=_SHM_REF.buf)


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

    # cfg_dict is produced from an already-normalized parent config.
    cfg = au.MAPEMConfig(**cfg_dict)
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


def _align_particle_mapem_shared_worker(args):
    """Run one MAP-EM particle E-step using shared input arrays."""
    i, prev_angle, schedule, cfg_dict, mask_diameter, mask_soft_edge = args

    cfg = au.MAPEMConfig(**cfg_dict)
    img = _SHARED_X[int(i)]
    ref_match = _SHARED_REF

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
    initial_params=None,
    search_mode=None,
    use_shared_memory=False,
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

    params, has_initial_params = au.initialize_mapem_params(initial_params, n)
    requested_search_mode = "auto" if search_mode is None else str(search_mode).strip().lower()
    if requested_search_mode not in {"auto", "global", "refine"}:
        raise ValueError("search_mode must be 'auto', 'global', or 'refine'.")
    if requested_search_mode == "refine" and not has_initial_params:
        raise ValueError("search_mode='refine' requires initial_params.")
    use_refine_schedule = requested_search_mode == "refine" or (
        requested_search_mode == "auto" and has_initial_params
    )
    resolved_search_mode = "refine" if use_refine_schedule else "global"

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
        "engine": "alignimg-mapem-multicore",
        "config": asdict(cfg),
        "iterations": [],
        "num_particles": n,
        "num_iterations": int(num_iterations),
        "mask_diameter": mask_diameter,
        "n_jobs": n_jobs,
        "chunksize": chunksize,
        "has_initial_params": bool(has_initial_params),
        "search_mode": resolved_search_mode,
        "warm_start": bool(use_refine_schedule),
        "use_shared_memory": bool(use_shared_memory),
        "implemented": True,
    }

    weights = np.ones(n, dtype=np.float32)
    image_scores = np.zeros(n, dtype=np.float32)
    posterior_scores = np.zeros(n, dtype=np.float32)

    x_shm = None
    x_shape = None
    timing_shared_x_setup_s = 0.0
    timing_shared_x_cleanup_s = 0.0
    try:
        if use_shared_memory:
            shared_x_t0 = time.perf_counter()
            x_shm, x_shape = _create_float32_shared_array(X)
            timing_shared_x_setup_s = time.perf_counter() - shared_x_t0

        for it in range(int(num_iterations)):
            iter_t0 = time.perf_counter()
            if use_refine_schedule:
                schedule = au.mapem_warm_start_iter_schedule(it, int(num_iterations))
            else:
                schedule = au.mapem_iter_schedule(it, int(num_iterations))
            if verbose:
                print(
                    f"[multicore iter {it + 1}/{num_iterations}] "
                    f"phase={cfg.phase} mode={schedule['mode']} "
                    f"lp_sigma={schedule['lp_sigma']} weight={cfg.weight_mode} "
                    f"n_jobs={n_jobs}"
                )

            ref_t0 = time.perf_counter()
            ref_match = au.apply_lowpass_filter(current_ref, sigma=float(schedule["lp_sigma"]))
            timing_ref_prepare_s = time.perf_counter() - ref_t0

            timing_shared_setup_s = 0.0
            timing_shared_cleanup_s = 0.0
            ref_shm = None
            ref_shape = None
            if use_shared_memory:
                timing_shared_setup_s += timing_shared_x_setup_s if it == 0 else 0.0
                shared_ref_t0 = time.perf_counter()
                ref_shm, ref_shape = _create_float32_shared_array(ref_match)
                timing_shared_setup_s += time.perf_counter() - shared_ref_t0

            task_build_t0 = time.perf_counter()
            aligned_stack = np.empty_like(X, dtype=np.float32)
            diagnostics = []
            cfg_dict = asdict(cfg)

            if use_shared_memory:
                worker_args = [
                    (
                        i,
                        params[i, 0],
                        schedule,
                        cfg_dict,
                        mask_diameter,
                        cfg.mask_soft_edge,
                    )
                    for i in range(n)
                ]
            else:
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
            timing_task_build_s = time.perf_counter() - task_build_t0

            worker_t0 = time.perf_counter()
            try:
                if use_shared_memory:
                    with ProcessPoolExecutor(
                        max_workers=n_jobs,
                        initializer=_init_shared_mapem_worker,
                        initargs=(x_shm.name, x_shape, ref_shm.name, ref_shape),
                    ) as executor:
                        results = executor.map(
                            _align_particle_mapem_shared_worker,
                            worker_args,
                            chunksize=chunksize,
                        )
                        collect_t0 = time.perf_counter()
                        for (
                            i,
                            params_i,
                            aligned_i,
                            image_score,
                            posterior_score,
                            diagnostics_record,
                        ) in results:
                            params[i] = params_i
                            aligned_stack[i] = aligned_i
                            image_scores[i] = float(image_score)
                            posterior_scores[i] = float(posterior_score)
                            if i < int(cfg.diagnostics_n):
                                diagnostics.append(diagnostics_record)
                        timing_collect_s = time.perf_counter() - collect_t0
                else:
                    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                        results = executor.map(
                            _align_particle_mapem_worker,
                            worker_args,
                            chunksize=chunksize,
                        )
                        collect_t0 = time.perf_counter()
                        for (
                            i,
                            params_i,
                            aligned_i,
                            image_score,
                            posterior_score,
                            diagnostics_record,
                        ) in results:
                            params[i] = params_i
                            aligned_stack[i] = aligned_i
                            image_scores[i] = float(image_score)
                            posterior_scores[i] = float(posterior_score)
                            if i < int(cfg.diagnostics_n):
                                diagnostics.append(diagnostics_record)
                        timing_collect_s = time.perf_counter() - collect_t0
            finally:
                if ref_shm is not None:
                    shared_cleanup_t0 = time.perf_counter()
                    ref_shm.close()
                    ref_shm.unlink()
                    timing_shared_cleanup_s += time.perf_counter() - shared_cleanup_t0
            timing_worker_map_s = time.perf_counter() - worker_t0

            weight_t0 = time.perf_counter()
            weights, weight_info = au.estimate_inlier_weights(posterior_scores, cfg)
            timing_weight_s = time.perf_counter() - weight_t0

            ref_update_t0 = time.perf_counter()
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
            timing_reference_update_s = time.perf_counter() - ref_update_t0
            timing_total_iter_s = time.perf_counter() - iter_t0

            meta["iterations"].append({
                "iteration": it,
                "schedule": dict(schedule),
                "image_score_mean": float(np.mean(image_scores)),
                "posterior_score_mean": float(np.mean(posterior_scores)),
                "weight_mean": float(np.mean(weights)),
                "weight_min": float(np.min(weights)),
                "weight_max": float(np.max(weights)),
                **weight_info,
                "timing_ref_prepare_s": float(timing_ref_prepare_s),
                "timing_task_build_s": float(timing_task_build_s),
                "timing_worker_map_s": float(timing_worker_map_s),
                "timing_collect_s": float(timing_collect_s),
                "timing_estep_s": float(timing_task_build_s + timing_worker_map_s),
                "timing_weight_s": float(timing_weight_s),
                "timing_reference_update_s": float(timing_reference_update_s),
                "timing_total_iter_s": float(timing_total_iter_s),
                "timing_shared_setup_s": float(timing_shared_setup_s),
                "timing_shared_cleanup_s": float(timing_shared_cleanup_s),
                "diagnostics": diagnostics,
            })

            if verbose:
                print(
                    "  timing: "
                    f"ref_prepare={timing_ref_prepare_s:.3f}s "
                    f"task_build={timing_task_build_s:.3f}s "
                    f"worker_map={timing_worker_map_s:.3f}s "
                    f"collect={timing_collect_s:.3f}s "
                    f"weight={timing_weight_s:.3f}s "
                    f"reference_update={timing_reference_update_s:.3f}s "
                    f"total={timing_total_iter_s:.3f}s"
                )
    finally:
        if x_shm is not None:
            shared_cleanup_t0 = time.perf_counter()
            x_shm.close()
            x_shm.unlink()
            timing_shared_x_cleanup_s = time.perf_counter() - shared_cleanup_t0
            if meta["iterations"]:
                meta["iterations"][-1]["timing_shared_cleanup_s"] += float(
                    timing_shared_x_cleanup_s
                )

    meta["last_weights"] = weights.astype(np.float32)
    meta["last_image_scores"] = image_scores.astype(np.float32)
    meta["last_posterior_scores"] = posterior_scores.astype(np.float32)
    meta["timing_shared_x_setup_s"] = float(timing_shared_x_setup_s)
    meta["timing_shared_x_cleanup_s"] = float(timing_shared_x_cleanup_s)
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
