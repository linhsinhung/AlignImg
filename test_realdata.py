#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:05:02 2026

@author: linhsinhung
"""

import os
import gc
import time
import numpy as np
from tqdm import tqdm

# === Imports ===
import align_utils as au
from alignimg_api import run_alignment, run_transform, available_backends
    


def load_mrc_stack(path):
    mrc = mrcfile.mmap(path, permissive=True, mode='r')
    return mrc.data

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

def save_alignment_results(params, output_path="alignment_results.csv"):
    N = params.shape[0]
    df = pd.DataFrame({
        "Particle_Idx": np.arange(N),
        "Angle_Psi": params[:, 0],
        "Align_Dy": params[:, 1],
        "Align_Dx": params[:, 2],
        "Score": params[:, 3],
    })
    df.to_csv(output_path, index=False, float_format="%.6f")
    print(f">> Parameters saved to: {os.path.abspath(output_path)}")
    return df



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import mrcfile
    import pandas as pd

    # --- Configuration ---
    BACKEND = "single"   # available: single / multicore / gpu
    
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
    
    print("Available backends:", available_backends())
    final_ref, history, params, meta = run_alignment(
        X_data,
        init_ref,
        num_iterations=4,
        mask_diameter=75,
        backend=BACKEND,
    )
    
    elapsed = time.time() - start_time
    print(f"\n>> Alignment Finished in {elapsed:.2f} seconds.")
    
    # --- Output ---
    plot_results(final_ref, X_data, gt_img=gt_img, save_path="api_result.png")
    save_alignment_results(params, "api_params.csv")

    # --- Apply Transform & Visual Check ---
    X_corrected = run_transform(X_data, params, engine=meta["engine"])
    plt.figure(figsize=(5, 5))
    plt.imshow(np.mean(X_corrected, axis=0), cmap='gray')
    plt.title("Corrected Average")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("api_corrected_mean.png")
    print(">> Corrected mean saved to api_corrected_mean.png")
