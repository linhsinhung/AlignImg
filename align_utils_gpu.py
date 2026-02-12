#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CryoEM 2D Alignment Utilities (GPU Version - CuPy)
Fully vectorized implementation with GeometryContext caching.
"""

# import numpy as np
import cupy as cp
from cupyx.scipy import ndimage as ndi
from cupyx.scipy.fft import fft2, ifft2, fftshift, fft, ifft

class GeometryContext:
    def __init__(self, shape):
        """
        Pre-allocates all static geometry grids on GPU.
        """
        if len(shape) == 3:
            self.H, self.W = shape[1], shape[2]
        else:
            self.H, self.W = shape
            
        self.cy = self.H / 2.0
        self.cx = self.W / 2.0
        self.center = cp.array([self.cy, self.cx])
        
        # 1. Standard Coordinate Grid (for CoM and general use)
        # origin at (0,0)
        y_idx, x_idx = cp.indices((self.H, self.W), dtype=cp.float32)
        self.grid_y = y_idx
        self.grid_x = x_idx
        
        # Centered Grid (for Masking)
        self.dist_sq = (self.grid_x - self.cx)**2 + (self.grid_y - self.cy)**2
        self.max_radius = min(self.H, self.W) / 2.0
        
        # 2. Window Function (Hanning)
        hann_h = cp.hanning(self.H)
        hann_w = cp.hanning(self.W)
        self.window = cp.outer(hann_h, hann_w).astype(cp.float32)
        
        # 3. Log-Polar Pre-calculation (For Fourier-Mellin)
        # Pre-calculate the sample coordinates (source_y, source_x) 
        # that map the image to log-polar space.
        self._init_log_polar_grid()
        self._mask_cache = {}

    def _init_log_polar_grid(self):
        """
        Creates a sampling grid for cv2.linearPolar equivalent.
        """
        # Desired output size for Polar Transform (keep same as input)
        ph, pw = self.H, self.W
        
        max_r = self.max_radius
        # Log-Polar logic: 
        # x-axis in transformed image corresponds to log(radius)
        # y-axis corresponds to angle
        
        # Create grid in Log-Polar space
        idx_r = cp.arange(pw, dtype=cp.float32)
        idx_theta = cp.arange(ph, dtype=cp.float32)
        grid_theta, grid_logr = cp.meshgrid(idx_theta, idx_r, indexing='ij')
        
        # Map back to source Radius and Angle
        # Formula matching cv2.WARP_FILL_OUTLIERS + linearPolar mechanics
        # K = pw / log(max_r)
        K = pw / cp.log(max_r)
        r = cp.exp(grid_logr / K)
        theta = (grid_theta / ph) * 2 * cp.pi
        
        # Convert to Cartesian (Source Coordinates)
        # These are the (y, x) coordinates in the original FFT image we need to sample from
        self.lp_sample_y = self.cy + r * cp.sin(theta)
        self.lp_sample_x = self.cx + r * cp.cos(theta)
        
        # Flatten for map_coordinates
        self.lp_coords = cp.stack([self.lp_sample_y.flatten(), self.lp_sample_x.flatten()])

    def get_circular_mask(self, diameter=None, soft_edge=5):
        """ Generates or retrieves a cached mask. """
        key = (None if diameter is None else float(diameter), int(soft_edge))
        if key in self._mask_cache:
            return self._mask_cache[key]

        radius = (diameter / 2.0) if diameter else (self.max_radius - 1)
        inner = radius - soft_edge
        
        mask = cp.zeros((self.H, self.W), dtype=cp.float32)
        mask_indices = self.dist_sq < (inner**2)
        edge_indices = (self.dist_sq >= (inner**2)) & (self.dist_sq < (radius**2))
        
        mask[mask_indices] = 1.0
        
        # Soft edge calculation
        dists = cp.sqrt(self.dist_sq[edge_indices])
        norm_dists = (dists - inner) / soft_edge
        mask[edge_indices] = 0.5 * (1 + cp.cos(norm_dists * cp.pi))
        
        self._mask_cache[key] = mask
        return mask

# === 1. Pre-processing & Utilities ===

def get_circular_mask(geo: GeometryContext, diameter=None, soft_edge=5):
    """
    Wrapper function to get the mask directly from the geometry object.
    Used when main.py calls aug.get_circular_mask(geo, ...)
    """
    return geo.get_circular_mask(diameter=diameter, soft_edge=soft_edge)

def apply_circular_mask_batch(img_stack, geo: GeometryContext, diameter=None, soft_edge=5):
    """
    Applies circular mask to a batch of images (N, H, W) or single image (H, W).
    """
    # 1. Get the (H, W) mask from GeometryContext
    mask = geo.get_circular_mask(diameter=diameter, soft_edge=soft_edge)
    
    # 2. Apply via Broadcasting
    # If img_stack is (N, H, W), CuPy automatically broadcasts (H, W) mask to all N.
    return img_stack * mask

def apply_lowpass_batch(img_stack, sigma=2.0):
    """
    Apply Gaussian Lowpass Filter.
    Auto-detects 2D (Single Image) or 3D (Batch of Images).
    """
    if sigma <= 0: 
        return img_stack
        
    # Determine input dimensionality
    if img_stack.ndim == 2:
        # Case 1: Single Image (H, W) -> Apply isotropic sigma
        return ndi.gaussian_filter(img_stack, sigma=sigma)
        
    elif img_stack.ndim == 3:
        # Case 2: Batch (N, H, W) -> Don't filter across batch axis (0)
        return ndi.gaussian_filter(img_stack, sigma=(0, sigma, sigma))
        
    else:
        raise ValueError(f"Input must be 2D or 3D, got shape {img_stack.shape}")

def calculate_com_batch(img_stack, geo: GeometryContext):
    """
    Vectorized Center of Mass calculation.
    Returns: offsets (N, 2) where [dy, dx]
    """
    # Normalize to avoid numerical issues (pseudo-probability)
    # shape: (N, H, W)
    batch_min = cp.min(img_stack, axis=(1, 2), keepdims=True)
    img_pos = img_stack - batch_min
    mass = cp.sum(img_pos, axis=(1, 2))
    
    # Avoid division by zero
    mass = cp.maximum(mass, 1e-6)
    
    # Vectorized moment calculation
    # grid_y, grid_x are (H, W), broaden to (1, H, W)
    # Sum(I * y) / Sum(I)
    cy_com = cp.sum(img_pos * geo.grid_y, axis=(1, 2)) / mass
    cx_com = cp.sum(img_pos * geo.grid_x, axis=(1, 2)) / mass
    
    dy = geo.cy - cy_com
    dx = geo.cx - cx_com
    
    return cp.stack([dy, dx], axis=1) # (N, 2)

# === 2. Geometric Transformations (Batch) ===

def warp_affine_batch(img_stack, geo: GeometryContext, angles, dys, dxs):
    """
    Apply Rotate + Shift to a batch of images using map_coordinates.
    This replaces individual cv2.warpAffine.
    
    Order: Rotate around center -> Shift
    """
    N, H, W = img_stack.shape
    
    # 1. Prepare Transformation Matrices (Inverse for sampling)
    # We want to sample pixel (y, x) from source (y', x')
    # Forward: p_new = R * p_old + T
    # Inverse: p_old = R^T * (p_new - T)
    
    # Convert angles to radians (negative for inverse rotation direction logic check)
    # We use standard convention: Positive Angle = Counter-Clockwise
    theta = cp.deg2rad(angles)
    cos_t = cp.cos(theta)
    sin_t = cp.sin(theta)
    
    # 2. Create Grid of Destination Coordinates (N, H, W)
    # Center grid so rotation is around (0,0)
    y_centered = geo.grid_y - geo.cy # (H, W)
    x_centered = geo.grid_x - geo.cx
    
    # Flatten for broadcasting: (1, H*W)
    y_flat = y_centered.flatten()
    x_flat = x_centered.flatten()
    
    # 3. Inverse Transform Calculation (Vectorized over N)
    # Source X = (x_dest - dx) * cos + (y_dest - dy) * sin + cx
    # Source Y = -(x_dest - dx) * sin + (y_dest - dy) * cos + cy
    
    # Adjust for shift (T) first
    # shapes: x_flat (M,), dxs (N, 1) -> (N, M)
    x_shifted = x_flat[None, :] - dxs[:, None]
    y_shifted = y_flat[None, :] - dys[:, None]
    
    # Apply Inverse Rotation
    src_x = (x_shifted * cos_t[:, None]) + (y_shifted * sin_t[:, None]) + geo.cx
    src_y = -(x_shifted * sin_t[:, None]) + (y_shifted * cos_t[:, None]) + geo.cy
    
    # 4. Map Coordinates
    # map_coordinates does not support batching coordinates against batch images efficiently in old versions
    # However, we can flatten everything into 2D or loop slightly optimized.
    # Best Vectorized approach in CuPy: Treat (N, H, W) as flat volume? No, rotation differs per image.
    # Optimization: Use cupyx.scipy.ndimage.map_coordinates with `coordinates` shape (2, N, H, W) is NOT supported.
    
    # Workaround for Batch Warping: 
    # Since N can be large, we loop over N but keep data on GPU. 
    # Or typically for Alignment N is small (batch size 100-200).
    # Ideally, we write a custom CUDA kernel, but for pure CuPy:
    
    output = cp.zeros_like(img_stack)
    
    # Coordinates shape needed for map_coordinates: (2, H*W)
    # We loop because map_coordinates requires matching specific image to specific coords
    for i in range(N):
        coords = cp.stack([src_y[i], src_x[i]])
        output[i] = ndi.map_coordinates(img_stack[i], coords, order=1, mode='constant', cval=0.0).reshape(H, W)
        
    return output

# === 3. Alignment Logic ===

def get_coarse_angle_fm_batch(img_stack, ref_img, geo: GeometryContext):
    """
    Fourier-Mellin implementation using Pre-calculated Log-Polar grid.
    """
    N, H, W = img_stack.shape
    
    # 1. FFT -> Power Spectrum
    # Windowing
    img_w = img_stack * geo.window
    ref_w = ref_img * geo.window
    
    F_img = cp.abs(fftshift(fft2(img_w), axes=(1, 2)))
    F_ref = cp.abs(fftshift(fft2(ref_w)))
    
    # Log transform (whitening)
    F_img = cp.log(F_img + 1)
    F_ref = cp.log(F_ref + 1)
    
    # 2. Log-Polar Transform (Using cached coordinates)
    # Ref Polar
    pol_ref = ndi.map_coordinates(F_ref, geo.lp_coords, order=1, mode='constant').reshape(H, W)
    prof_ref = cp.sum(pol_ref, axis=1) # Sum over log-radius
    
    # Batch Img Polar
    # Loop required for map_coordinates on batch unless we stack images into 3D volume (which they are)
    # but coords are same for all images. map_coordinates can map 1 coord set to 1 volume?
    # No, map_coordinates inputs must match rank.
    # Strategy: Reshape stack to (N*H*W), mapping is tricky.
    # Simple loop is fast enough because map_coordinates is the heavy lifting.
    
    prof_imgs = cp.zeros((N, H), dtype=cp.float32)
    
    # Optimization: If N is large, this loop is the bottleneck.
    # But F-M is usually done once per global search.
    for i in range(N):
        p_img = ndi.map_coordinates(F_img[i], geo.lp_coords, order=1, mode='constant').reshape(H, W)
        prof_imgs[i] = cp.sum(p_img, axis=1)
        
    # 3. 1D Correlation (FFT based)
    pad_len = H
    # Pad for linear correlation
    prof_imgs_pad = cp.pad(prof_imgs, ((0,0), (0, pad_len)))
    prof_ref_pad = cp.pad(prof_ref, (0, pad_len))
    
    fft_len = int(2**cp.ceil(cp.log2(2*H - 1)))
    
    fp_img = fft(prof_imgs_pad, n=fft_len, axis=1)
    fp_ref = fft(prof_ref_pad, n=fft_len) # Broadcasts
    
    corr = cp.real(ifft(fp_img * cp.conj(fp_ref), axis=1))
    shift_idx = cp.argmax(corr, axis=1)
    
    # 4. Convert lag to angle
    # Lag > fft_len/2 means negative shift
    lags = cp.where(shift_idx > fft_len//2, shift_idx - fft_len, shift_idx)
    angles = -(lags.astype(cp.float32) / H) * 360.0
    
    # Normalize
    angles = (angles + 180.0) % 360.0 - 180.0
    return angles

def get_translation_fft_batch(img_stack, ref_img, geo: GeometryContext):
    """
    Computes best (dy, dx) for a batch of images against one reference.
    """
    N, H, W = img_stack.shape
    
    f_img = fft2(img_stack, axes=(1, 2))
    f_ref = fft2(ref_img) # Auto broadcast
    
    cc_spec = f_img * cp.conj(f_ref)
    cc_map = cp.real(ifft2(cc_spec, axes=(1, 2)))
    cc_map = fftshift(cc_map, axes=(1, 2))
    
    # Find argmax in 2D
    # Flatten last two dims to find max index
    flat_map = cc_map.reshape(N, -1)
    max_indices = cp.argmax(flat_map, axis=1)
    scores = cp.max(flat_map, axis=1)
    
    cy_idxs, cx_idxs = cp.unravel_index(max_indices, (H, W))
    
    # Calculate dy, dx (Standard FFT center convention)
    # Note: cx_int, cy_int from geo are Python floats/ints, cast to cupy if needed
    dy = cy_idxs.astype(cp.float32) - int(geo.cy) # int cast matches CPU logic
    dx = cx_idxs.astype(cp.float32) - int(geo.cx)
    
    return dy, dx, scores

def check_180_ambiguity_batch(img_stack, ref_img, candidate_angles, geo: GeometryContext):
    """
    Checks angle vs angle+180.
    """
    N = img_stack.shape[0]
    zeros = cp.zeros(N, dtype=cp.float32)
    
    # 1. Rotate at Candidate
    rot_1 = warp_affine_batch(img_stack, geo, candidate_angles, zeros, zeros)
    _, _, score_1 = get_translation_fft_batch(rot_1, ref_img, geo)
    
    # 2. Rotate at Candidate + 180
    rot_2 = warp_affine_batch(img_stack, geo, candidate_angles + 180, zeros, zeros)
    _, _, score_2 = get_translation_fft_batch(rot_2, ref_img, geo)
    
    final_angles = cp.where(score_2 > score_1, candidate_angles + 180, candidate_angles)
    final_scores = cp.maximum(score_1, score_2)
    
    return (final_angles + 180.0) % 360.0 - 180.0, final_scores

def refine_subpixel_parabolic_batch(scores_matrix, max_indices, step):
    """
    Vectorized Parabolic Refinement.
    
    Args:
        scores_matrix: (N, K) scores for K angles.
        max_indices: (N,) indices of the max score.
        step: angle step size.
        
    Returns:
        deltas: (N,) adjustment to the angle.
    """
    N, K = scores_matrix.shape
    rows = cp.arange(N)
    
    # 1. Handle Edges: If max is at 0 or K-1, we cannot interpolate.
    # Create a mask for valid inner peaks
    is_inner = (max_indices > 0) & (max_indices < K - 1)
    
    # 2. Gather Neighbors (Vectorized Indexing)
    # Only compute for valid inner indices to avoid index out of bounds, 
    # but for vectorization simplicty, we clip indices and mask result later.
    
    # Clip indices to stay within bounds for gathering (safe gathering)
    safe_idx = cp.clip(max_indices, 1, K - 2)
    
    y_c = scores_matrix[rows, safe_idx]
    y_l = scores_matrix[rows, safe_idx - 1]
    y_r = scores_matrix[rows, safe_idx + 1]
    
    # 3. Parabolic Formula
    # delta = 0.5 * (y_l - y_r) / (y_l - 2*y_c + y_r)
    numerator = 0.5 * (y_l - y_r)
    denominator = y_l - 2 * y_c + y_r
    
    # Avoid division by zero
    denominator = cp.where(cp.abs(denominator) < 1e-6, 1e-6, denominator)
    
    deltas = numerator / denominator
    
    # 4. Apply refinement only to valid inner peaks
    # If edge peak, delta = 0
    final_deltas = cp.where(is_inner, deltas, 0.0)
    
    # Scale by step
    return final_deltas * step

def fine_alignment_search_batch(img_stack, ref_img, coarse_angles, geo: GeometryContext, search_range=5, step=1):
    """
    Scans angles and performs sub-pixel parabolic refinement.
    """
    N = img_stack.shape[0]
    
    # Generate search grid
    offsets = cp.arange(-search_range, search_range + step, step, dtype=cp.float32)
    K = len(offsets)
    
    # Storage for all attempts: (N, K)
    # Memory usage: For N=512, K=21 -> ~43KB (Tiny)
    scores_matrix = cp.zeros((N, K), dtype=cp.float32)
    dys_matrix = cp.zeros((N, K), dtype=cp.float32)
    dxs_matrix = cp.zeros((N, K), dtype=cp.float32)
    
    zeros = cp.zeros(N, dtype=cp.float32)
    
    # Loop to fill the matrix
    for i, off in enumerate(offsets):
        # Current test angles
        test_angles = coarse_angles + off
        
        # Rotate
        rot_img = warp_affine_batch(img_stack, geo, test_angles, zeros, zeros)
        
        # Shift Search (FFT)
        dy, dx, sc = get_translation_fft_batch(rot_img, ref_img, geo)
        
        # Store
        scores_matrix[:, i] = sc
        dys_matrix[:, i] = dy
        dxs_matrix[:, i] = dx
        
    # === Refinement Stage ===
    
    # 1. Find Integer Max
    max_indices = cp.argmax(scores_matrix, axis=1) # (N,)
    
    # 2. Extract Best Coarse Parameters
    rows = cp.arange(N)
    best_coarse_offset = offsets[max_indices]
    best_dy = dys_matrix[rows, max_indices]
    best_dx = dxs_matrix[rows, max_indices]
    best_score_raw = scores_matrix[rows, max_indices]
    
    # 3. Calculate Sub-pixel Delta
    delta_angles = refine_subpixel_parabolic_batch(scores_matrix, max_indices, step)
    
    # 4. Final Parameters
    final_angles = coarse_angles + best_coarse_offset + delta_angles
    
    # Note: We stick to the integer dy/dx associated with the peak.
    # While angle changes slightly, re-calculating dy/dx for 0.X degree change 
    # usually yields the same pixel shift or requires sub-pixel shift logic (complex).
    # Using the dy/dx of the peak is standard practice.
    
    return {
        'angle': final_angles,
        'dy': best_dy,
        'dx': best_dx,
        'score': best_score_raw
    }
