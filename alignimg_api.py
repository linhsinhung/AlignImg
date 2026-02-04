#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CryoEM 2D Alignment Unified API
Integrates Serial (CPU), Parallel (CPU), and Batch (GPU) implementations.

Usage:
    from alignimg_api import run_alignment
    
    # Auto (CPU Parallel):
    ref, history, params, meta = run_alignment(X, init_ref)
    
    # Force Single Core:
    run_alignment(..., n_jobs=1)
    
    # GPU (with fallback):
    run_alignment(..., use_gpu=True)
"""

import os
import gc
import time
import numpy as np
import multiprocessing
from tqdm import tqdm
from functools import partial

# === Imports & GPU Check ===
import align_utils as au

try:
    import cupy as cp
    import align_utils_gpu as aug
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    aug = None

# =============================================================================
# [Multiprocessing Workers]
# Must be at module level for pickling
# =============================================================================

def _worker_calc_com(data_pack):
    """ Worker: Calculate Center of Mass (CPU Parallel) """
    img, geo, sigma = data_pack
    dy, dx = au.calculate_center_of_mass_shift(img, geo, sigma=sigma)
    return np.array([dy, dx], dtype=np.float32)

def _worker_align_particle(data_pack):
    """ Worker: Single Particle Alignment Logic (CPU Parallel) """
    (idx, img, current_ref_smooth, 
     geo, mask_diameter, lp_sigma, 
     is_global_search, search_range, search_step, 
     param_angle, param_dy, param_dx, 
     com_dy, com_dx) = data_pack

    # A. Apply Shift (Pre-rotation)
    if is_global_search:
        current_bias_y, current_bias_x = com_dy, com_dx
        current_angle = 0.0
    else:
        current_bias_y, current_bias_x = param_dy, param_dx
        current_angle = param_angle

    img_centered = au.shift_image(img, geo, current_bias_y, current_bias_x)
    
    # Prepare for matching
    img_masked = au.apply_circular_mask(img_centered, geo, diameter=mask_diameter)
    img_for_matching = au.apply_lowpass_filter(img_masked, sigma=lp_sigma)

    # B. Determine Angle
    if is_global_search:
        raw_angle = au.get_coarse_angle_fourier_mellin(img_for_matching, current_ref_smooth, geo)
        center_angle, _ = au.check_180_ambiguity(img_for_matching, current_ref_smooth, raw_angle, geo)
    else:
        center_angle = current_angle

    # C. Fine Tuning (Residuals in Post-rotation frame)
    best = au.fine_alignment_search(
        img_for_matching, current_ref_smooth, center_angle, geo,
        search_range=search_range, step=search_step
    )

    # D. Update State (Logic Fix)
    res_dy, res_dx = best['dy'], best['dx']
    best_angle = best['angle']
    best_score = best['score']

    # Back rotation of residuals
    rad = np.deg2rad(-best_angle)
    res_dx_pre = res_dx * np.cos(rad) - res_dy * np.sin(rad)
    res_dy_pre = res_dx * np.sin(rad) + res_dy * np.cos(rad)

    new_bias_y = current_bias_y - res_dy_pre
    new_bias_x = current_bias_x - res_dx_pre

    # E. Generate Final Image
    final_shifted = au.shift_image(img, geo, new_bias_y, new_bias_x)
    aligned_img = au.rotate_image(final_shifted, geo, best_angle)

    new_params = np.array([best_angle, new_bias_y, new_bias_x, best_score], dtype=np.float32)
    return idx, new_params, aligned_img


# =============================================================================
# [Engine 1] Serial Implementation (CPU)
# =============================================================================
def run_stateful_alignment_serial(X, initial_ref, num_iterations=4, mask_diameter=None):
    """ Standard serial processing compatible with align_utils.py """
    print(">> Mode: Serial (CPU Single Core)")
    N, H, W = X.shape
    geo = au.get_geometry_context((H, W))
    
    # 1. CoM
    print("   [Step 0] Pre-calculating CoM...")
    com_offsets = np.zeros((N, 2), dtype=np.float32)
    for i in tqdm(range(N), desc="   CoM"):
        com_offsets[i] = au.calculate_center_of_mass_shift(X[i], geo, sigma=5)
    
    state_params = np.zeros((N, 4), dtype=np.float32) # [angle, dy, dx, score]
    current_ref = au.apply_circular_mask(initial_ref.copy(), geo, diameter=mask_diameter)
    history_refs = [current_ref]
    
    for it in range(num_iterations):
        lp_sigma = 3.0 if it == 0 else (1.0 if it == 1 else 0.0)
        is_global_search = (it == 0)
        search_range = 15 if is_global_search else 5 
        search_step = 2.0 if it < num_iterations - 1 else 0.5
        
        print(f"   [Iter {it+1}/{num_iterations}] Global={is_global_search}, LP={lp_sigma}")
        
        ref_accumulator = np.zeros((H, W), dtype=np.float32)
        scores_sum = 0
        ref_for_matching = au.apply_lowpass_filter(current_ref, sigma=lp_sigma)
        
        for i in tqdm(range(N), desc=f"   Aligning"):
            # Reuse logic via direct call or inline? Inline for speed in serial.
            # (Logic identical to worker, kept inline for simplicity in serial engine)
            
            if is_global_search:
                curr_bias_y, curr_bias_x = com_offsets[i, 0], com_offsets[i, 1]
                curr_angle = 0.0
            else:
                curr_angle = state_params[i, 0]
                curr_bias_y, curr_bias_x = state_params[i, 1], state_params[i, 2]
                
            img_centered = au.shift_image(X[i], geo, curr_bias_y, curr_bias_x)
            img_masked = au.apply_circular_mask(img_centered, geo, diameter=mask_diameter)
            img_match = au.apply_lowpass_filter(img_masked, sigma=lp_sigma)
            
            if is_global_search:
                raw = au.get_coarse_angle_fourier_mellin(img_match, ref_for_matching, geo)
                cen_ang, _ = au.check_180_ambiguity(img_match, ref_for_matching, raw, geo)
            else:
                cen_ang = curr_angle
            
            best = au.fine_alignment_search(img_match, ref_for_matching, cen_ang, geo, 
                                            search_range=search_range, step=search_step)
            
            # Update Logic
            rad = np.deg2rad(-best['angle'])
            rdx = best['dx'] * np.cos(rad) - best['dy'] * np.sin(rad)
            rdy = best['dx'] * np.sin(rad) + best['dy'] * np.cos(rad)
            
            nb_y = curr_bias_y - rdy
            nb_x = curr_bias_x - rdx
            
            state_params[i] = [best['angle'], nb_y, nb_x, best['score']]
            
            # Accumulate
            final_s = au.shift_image(X[i], geo, nb_y, nb_x)
            aligned = au.rotate_image(final_s, geo, best['angle'])
            ref_accumulator += aligned
            scores_sum += best['score']

        new_ref = ref_accumulator / N
        new_ref = (new_ref - np.mean(new_ref)) / (np.std(new_ref) + 1e-8)
        current_ref = au.apply_circular_mask(new_ref, geo, diameter=mask_diameter)
        history_refs.append(current_ref)
        
    return current_ref, history_refs, state_params, com_offsets


# =============================================================================
# [Engine 2] Parallel Implementation (CPU Multiprocessing)
# =============================================================================
def run_stateful_alignment_parallel(X, initial_ref, num_iterations=4, mask_diameter=None, n_jobs=-1):
    """ Parallelized Alignment Driver """
    
    # Determine workers
    max_cores = multiprocessing.cpu_count()
    if n_jobs is None or n_jobs < 1:
        num_workers = max(1, max_cores - 1) # Reserve 1 core
    else:
        num_workers = min(n_jobs, max_cores)
        
    print(f">> Mode: Parallel CPU (Workers={num_workers})")
    
    N, H, W = X.shape
    geo = au.get_geometry_context((H, W))
    
    # 1. CoM Parallel
    print("   [Step 0] Parallel CoM Calculation...")
    tasks_com = [(X[i], geo, 5) for i in range(N)]
    com_offsets = np.zeros((N, 2), dtype=np.float32)
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(_worker_calc_com, tasks_com, chunksize=10), total=N, desc="   CoM"))
    for i, res in enumerate(results):
        com_offsets[i] = res
        
    state_params = np.zeros((N, 4), dtype=np.float32)
    current_ref = au.apply_circular_mask(initial_ref.copy(), geo, diameter=mask_diameter)
    history_refs = [current_ref]

    for it in range(num_iterations):
        lp_sigma = 3.0 if it == 0 else (1.0 if it == 1 else 0.0)
        is_global_search = (it == 0)
        search_range = 15 if is_global_search else 5 
        search_step = 2.0 if it < num_iterations - 1 else 0.5
        
        print(f"   [Iter {it+1}/{num_iterations}] Global={is_global_search}, LP={lp_sigma}")
        ref_for_matching = au.apply_lowpass_filter(current_ref, sigma=lp_sigma)
        
        # Prepare Tasks
        tasks_align = []
        for i in range(N):
            task = (
                i, X[i], ref_for_matching, 
                geo, mask_diameter, lp_sigma, 
                is_global_search, search_range, search_step,
                state_params[i,0], state_params[i,1], state_params[i,2], 
                com_offsets[i,0], com_offsets[i,1]
            )
            tasks_align.append(task)
            
        ref_accumulator = np.zeros((H, W), dtype=np.float32)
        scores_sum = 0.0
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            iterator = pool.imap(_worker_align_particle, tasks_align, chunksize=5)
            for idx, new_params, aligned_img in tqdm(iterator, total=N, desc="   Aligning"):
                state_params[idx] = new_params
                ref_accumulator += aligned_img
                scores_sum += new_params[3]
                
        new_ref = ref_accumulator / N
        new_ref = (new_ref - np.mean(new_ref)) / (np.std(new_ref) + 1e-8)
        current_ref = au.apply_circular_mask(new_ref, geo, diameter=mask_diameter)
        history_refs.append(current_ref)
        gc.collect()

    return current_ref, history_refs, state_params, com_offsets


# =============================================================================
# [Engine 3] GPU Implementation (CuPy)
# =============================================================================
def run_batch_alignment_gpu(X_cpu, initial_ref_cpu, num_iterations=4, mask_diameter=None, batch_size=256):
    """ GPU Alignment Driver using align_utils_gpu """
    print(f">> Mode: GPU Accelerated (CuPy), Batch Size={batch_size}")
    
    # Init GPU resources
    N, H, W = X_cpu.shape
    mempool = cp.get_default_memory_pool()
    geo = aug.GeometryContext((H, W))
    
    state_params = np.zeros((N, 4), dtype=np.float32) 
    com_offsets = np.zeros((N, 2), dtype=np.float32)
    
    current_ref_gpu = cp.array(initial_ref_cpu, dtype=cp.float32)
    current_ref_gpu = aug.apply_circular_mask_batch(current_ref_gpu, geo, diameter=mask_diameter)
    
    # 1. CoM Batch
    print("   [Step 0] GPU CoM Calculation...")
    for start_idx in tqdm(range(0, N, batch_size), desc="   CoM"):
        end_idx = min(start_idx + batch_size, N)
        batch_gpu = cp.array(X_cpu[start_idx:end_idx], dtype=cp.float32)
        offsets_gpu = aug.calculate_com_batch(batch_gpu, geo)
        com_offsets[start_idx:end_idx] = cp.asnumpy(offsets_gpu)
        del batch_gpu, offsets_gpu
        mempool.free_all_blocks()
        
    history_refs = []
    
    for it in range(num_iterations):
        lp_sigma = 3.0 if it == 0 else (1.0 if it == 1 else 0.0)
        is_global_search = (it == 0)
        search_range = 15 if is_global_search else 5 
        search_step = 2.0 if it < num_iterations - 1 else 0.5
        
        print(f"   [Iter {it+1}/{num_iterations}] Global={is_global_search}, LP={lp_sigma}")
        
        ref_masked = aug.get_circular_mask(geo, diameter=mask_diameter) * current_ref_gpu
        ref_match = aug.apply_lowpass_batch(ref_masked, sigma=lp_sigma)
        ref_accum_gpu = cp.zeros((H, W), dtype=cp.float32)
        
        for start_idx in tqdm(range(0, N, batch_size), desc="   Aligning"):
            end_idx = min(start_idx + batch_size, N)
            curr_bs = end_idx - start_idx
            
            # Load
            img_batch = cp.array(X_cpu[start_idx:end_idx], dtype=cp.float32)
            
            # Get State
            if is_global_search:
                bias_y = cp.array(com_offsets[start_idx:end_idx, 0])
                bias_x = cp.array(com_offsets[start_idx:end_idx, 1])
                curr_angle = cp.zeros(curr_bs, dtype=cp.float32)
            else:
                p_batch = cp.array(state_params[start_idx:end_idx])
                curr_angle = p_batch[:, 0]
                bias_y, bias_x = p_batch[:, 1], p_batch[:, 2]
                
            # Shift
            img_centered = aug.warp_affine_batch(img_batch, geo, cp.zeros_like(curr_angle), bias_y, bias_x)
            
            # Match Prep
            mask = geo.get_circular_mask(diameter=mask_diameter)
            img_match = aug.apply_lowpass_batch(img_centered * mask, sigma=lp_sigma)
            
            # Angle
            if is_global_search:
                raw_ang = aug.get_coarse_angle_fm_batch(img_match, ref_match, geo)
                center_ang, _ = aug.check_180_ambiguity_batch(img_match, ref_match, raw_ang, geo)
            else:
                center_ang = curr_angle
            
            # Fine Search
            best = aug.fine_alignment_search_batch(img_match, ref_match, center_ang, geo, 
                                                   search_range=search_range, step=search_step)
            
            # Update Logic
            res_dy, res_dx = best['dy'], best['dx']
            final_ang = best['angle']
            
            rad = cp.deg2rad(-final_ang)
            cos_r, sin_r = cp.cos(rad), cp.sin(rad)
            res_dx_pre = res_dx * cos_r - res_dy * sin_r
            res_dy_pre = res_dx * sin_r + res_dy * cos_r
            
            new_by = bias_y - res_dy_pre
            new_bx = bias_x - res_dx_pre
            
            # Store State
            new_params = np.stack([
                cp.asnumpy(final_ang), cp.asnumpy(new_by), cp.asnumpy(new_bx), cp.asnumpy(best['score'])
            ], axis=1)
            state_params[start_idx:end_idx] = new_params
            
            # Accumulate
            aligned_batch = aug.warp_affine_batch(img_batch, geo, final_ang, new_by, new_bx)
            ref_accum_gpu += cp.sum(aligned_batch, axis=0)
            
            del img_batch, img_centered, img_match, aligned_batch
            mempool.free_all_blocks()
            
        new_ref = ref_accum_gpu / N
        new_ref = (new_ref - cp.mean(new_ref)) / (cp.std(new_ref) + 1e-8)
        current_ref_gpu = new_ref * geo.get_circular_mask(diameter=mask_diameter)
        
        history_refs.append(cp.asnumpy(current_ref_gpu))
        
    return cp.asnumpy(current_ref_gpu), history_refs, state_params, com_offsets


# =============================================================================
# [Main API] The Unified Interface
# =============================================================================
def run_alignment(X, initial_ref, num_iterations=4, mask_diameter=None, 
                  use_gpu=False, n_jobs=None, batch_size=512):
    """
    Unified entry point for 2D Alignment.
    
    Args:
        X (np.ndarray): Particle stack (N, H, W).
        initial_ref (np.ndarray): Initial reference image (H, W).
        num_iterations (int): Number of alignment iterations.
        mask_diameter (int): Diameter for circular mask (pixels).
        use_gpu (bool): If True, attempts to use GPU. Fallback to CPU if failed.
        n_jobs (int): CPU parallelism control. 
                      1 = Serial (Single Core). 
                      None or -1 = Use all available cores (Parallel).
        batch_size (int): Batch size for GPU processing.
        
    Returns:
        final_ref (np.ndarray): Aligned reference.
        history (list): List of references per iteration.
        params (np.ndarray): Alignment parameters [Angle, Dy, Dx, Score].
        meta (dict): Metadata containing CoM offsets and run configuration
            (keys: com_offsets, engine, num_iterations, mask_diameter).
    """
    engine = None
    final_ref = history = params = com_offsets = None

    # 1. GPU Logic with Fallback
    if use_gpu:
        if HAS_GPU:
            try:
                # Attempt to run on GPU
                final_ref, history, params, com_offsets = run_batch_alignment_gpu(
                    X, initial_ref, 
                    num_iterations=num_iterations, 
                    mask_diameter=mask_diameter,
                    batch_size=batch_size
                )
                engine = "gpu"
            except Exception as e:
                print(f"\n[WARNING] GPU execution failed: {e}")
                print("Switching to CPU mode automatically...\n")
        else:
            print("\n[WARNING] GPU requested but 'cupy' or 'align_utils_gpu' not found.")
            print("Switching to CPU mode automatically...\n")
            
    # 2. CPU Logic (Dispatch based on n_jobs)
    if engine != "gpu":
        if n_jobs == 1:
            final_ref, history, params, com_offsets = run_stateful_alignment_serial(
                X, initial_ref, 
                num_iterations=num_iterations, 
                mask_diameter=mask_diameter
            )
            engine = "cpu-serial"
        else:
            # n_jobs = None, -1, or > 1 all imply parallel
            final_ref, history, params, com_offsets = run_stateful_alignment_parallel(
                X, initial_ref, 
                num_iterations=num_iterations, 
                mask_diameter=mask_diameter,
                n_jobs=n_jobs
            )
            engine = "cpu-parallel"

    meta = {
        "com_offsets": com_offsets,
        "engine": engine,
        "num_iterations": num_iterations,
        "mask_diameter": mask_diameter,
    }
    return final_ref, history, params, meta


# =============================================================================
# [Utilities] Test & I/O
# =============================================================================
def _transform_worker(args):
    """Worker: Apply transform to a single image (CPU)."""
    img, angle, dy, dx = args
    geo = au.get_geometry_context(img.shape)
    return au.transform_final_image(img, geo, angle, dy, dx)

def run_transform(X, params, engine=None):
    """
    Apply alignment parameters from run_alignment to a data stack X.

    Args:
        X (np.ndarray): Particle stack (N, H, W).
        params (np.ndarray): Alignment parameters [Angle, Dy, Dx, Score].
        engine (str | None): Alignment engine ("gpu", "cpu-serial", "cpu-parallel").

    Returns:
        np.ndarray: Aligned particle stack (N, H, W).
    """
    use_gpu_transform = engine == "gpu"

    if use_gpu_transform:
        if not HAS_GPU or aug is None:
            raise RuntimeError("GPU transform requested but GPU utilities are unavailable.")

        X_gpu = X if hasattr(X, "get") else cp.asarray(X)
        params_gpu = params if hasattr(params, "get") else cp.asarray(params)

        N, H, W = X_gpu.shape
        geo = aug.GeometryContext((H, W))
        angles = params_gpu[:, 0]
        dys = params_gpu[:, 1]
        dxs = params_gpu[:, 2]

        X_corrected_gpu = aug.warp_affine_batch(X_gpu, geo, angles, dys, dxs)
        return cp.asnumpy(X_corrected_gpu)

    if hasattr(X, "get"):
        X = X.get()
    if hasattr(params, "get"):
        params = params.get()

    N, H, W = X.shape
    X_corrected = np.empty_like(X, dtype=np.float32)

    if engine == "cpu-parallel":
        work_items = [(X[i], params[i, 0], params[i, 1], params[i, 2]) for i in range(N)]
        with multiprocessing.Pool() as pool:
            results = pool.map(_transform_worker, work_items)
        X_corrected[:] = np.stack(results, axis=0)
        return X_corrected

    geo = au.get_geometry_context((H, W))
    for i in range(N):
        angle, dy, dx = params[i, 0], params[i, 1], params[i, 2]
        X_corrected[i] = au.transform_final_image(X[i], geo, angle, dy, dx)

    return X_corrected

def generate_synthetic_data(N=200, H=128, W=128, noise_level=1.0):
    print(f"Generating synthetic data (N={N}, Size={H}x{W})...")
    geo = au.get_geometry_context((H, W))
    y, x = np.ogrid[-3:3:H*1j, -3:3:W*1j]
    ground_truth = np.exp(-(x**2 + (y-1.0)**2)*2) + 0.8 * np.exp(-((x-1.0)**2 + y**2)*2)
    ground_truth = au.apply_circular_mask(ground_truth, geo)

    X_mock = np.zeros((N, H, W), dtype=np.float32)
    for i in range(N):
        ang, dx, dy = np.random.uniform(0,360), np.random.uniform(-5,5), np.random.uniform(-5,5)
        # We simulate the transform
        # Create a temp transform: Rotate then Shift
        img_rot = au.rotate_image(ground_truth, geo, ang)
        img_final = au.shift_image(img_rot, geo, dy, dx)
        X_mock[i] = img_final + np.random.normal(0, noise_level, (H, W))
        
    return ground_truth, X_mock

def plot_results(final_ref, X_raw, gt_img=None, save_path="alignment_result.png"):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3 if gt_img is not None else 2, figsize=(15, 5))
    
    axes[0].imshow(np.mean(X_raw[:min(200, len(X_raw))], axis=0), cmap='gray')
    axes[0].set_title("Raw Average")
    axes[0].axis('off')
    
    axes[1].imshow(final_ref, cmap='gray')
    axes[1].set_title("Aligned Reference")
    axes[1].axis('off')
    
    if gt_img is not None:
        axes[2].imshow(gt_img, cmap='gray')
        axes[2].set_title("Ground Truth")
        axes[2].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    print(f">> Plot saved to {save_path}")

def save_alignment_results(params, com_offsets, output_path="alignment_results.csv"):
    if hasattr(params, 'get'): params = params.get()
    if hasattr(com_offsets, 'get'): com_offsets = com_offsets.get()
       
    N = params.shape[0]
    df = pd.DataFrame({
        'Particle_Idx': np.arange(N),
        'CoM_Dy': com_offsets[:, 0],
        'CoM_Dx': com_offsets[:, 1],
        'Align_Dy': params[:, 1],
        'Align_Dx': params[:, 2],
        'Angle_Psi': params[:, 0],
        'Score': params[:, 3]
    })
    df['Total_Dy'] = df['CoM_Dy'] + df['Align_Dy']
    df['Total_Dx'] = df['CoM_Dx'] + df['Align_Dx']
    
    df.to_csv(output_path, index=False, float_format='%.6f')
    print(f">> Parameters saved to: {os.path.abspath(output_path)}")
    return df

def load_mrc_stack(path):
    mrc = mrcfile.mmap(path, permissive=True, mode='r')
    return mrc.data

# =============================================================================
# [Main Block] Self-Test
# =============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import mrcfile
    import pandas as pd

    multiprocessing.freeze_support()
    
    # --- Configuration ---
    USE_GPU_FLAG = True   # Change this to test switching
    N_JOBS = -1           # 1 for Serial, -1 for Parallel
    
    path_data = './test_align.mrcs'
    path_gt = './mu_aligned_mean.mrc'
    
    # --- Load Data ---
    if os.path.exists(path_data):
        print(f"Loading real data: {path_data}")
        X_data = load_mrc_stack(path_data)
        gt_img = np.squeeze(mrcfile.open(path_gt).data) if os.path.exists(path_gt) else None
        if gt_img is not None:
            # Simple fix for GT orientation if needed
            geo_temp = au.get_geometry_context(gt_img.shape)
            gt_img = au.rotate_image(gt_img, geo_temp, 180)
        
        init_ref = np.mean(X_data[:100], axis=0).astype(np.float32)
    else:
        print("Real data not found. Using synthetic generator.")
        gt_img, X_data = generate_synthetic_data(N=200)
        init_ref = np.mean(X_data, axis=0).astype(np.float32)

    # --- Run API ---
    start_time = time.time()
    
    final_ref, history, params, meta = run_alignment(
        X_data, 
        init_ref, 
        num_iterations=4, 
        mask_diameter=75,
        use_gpu=USE_GPU_FLAG,
        n_jobs=N_JOBS
    )
    
    elapsed = time.time() - start_time
    print(f"\n>> Alignment Finished in {elapsed:.2f} seconds.")
    
    # --- Output ---
    plot_results(final_ref, X_data, gt_img=gt_img, save_path="api_result.png")
    save_alignment_results(params, meta["com_offsets"], "api_params.csv")

    # --- Apply Transform & Visual Check ---
    X_corrected = run_transform(X_data, params, engine=meta["engine"])
    plt.figure(figsize=(5, 5))
    plt.imshow(np.mean(X_corrected, axis=0), cmap='gray')
    plt.title("Corrected Average")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("api_corrected_mean.png")
    print(">> Corrected mean saved to api_corrected_mean.png")
