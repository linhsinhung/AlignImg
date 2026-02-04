#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CryoEM 2D Alignment Utilities (CPU Fundamental v9)
Unified Math: Uses Float Center and FFT Correlation to match GPU logic.
"""

import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import center_of_mass 

# === 0. Helper & Geometry Context ===

def normalize_angle(angle):
    return (angle + 180.0) % 360.0 - 180.0

def get_geometry_context(shape):
    if len(shape) == 3:
        H, W = shape[1], shape[2]
    else:
        H, W = shape
        
    # [FIX] Unified Float Center (Matches GPU)
    cy = H / 2.0
    cx = W / 2.0
    
    # Integers for FFT shift reference (if needed)
    cy_int = int(H // 2)
    cx_int = int(W // 2)
    
    max_radius = min(H, W) / 2.0
    
    hann_h = np.hanning(H)
    hann_w = np.hanning(W)
    window = np.outer(hann_h, hann_w)
    
    return {
        'H': H, 'W': W,
        'cy': cy, 'cx': cx,
        'cy_int': cy_int, 'cx_int': cx_int,
        'max_radius': max_radius,
        'window': window
    }

# === 1. Pre-processing ===

def apply_circular_mask(img, geo, diameter=None, soft_edge=5):
    h, w = geo['H'], geo['W']
    cy, cx = geo['cy'], geo['cx']
    
    if diameter is None:
        radius = geo['max_radius'] - 1
    else:
        radius = diameter / 2.0
    
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    mask = np.zeros_like(img)
    inner_radius = radius - soft_edge
    
    mask_indices = dist_from_center < inner_radius
    edge_indices = (dist_from_center >= inner_radius) & (dist_from_center < radius)
    
    mask[mask_indices] = 1.0
    normalized_dist = (dist_from_center[edge_indices] - inner_radius) / soft_edge
    mask[edge_indices] = 0.5 * (1 + np.cos(normalized_dist * np.pi))
    
    return img * mask

def apply_lowpass_filter(img, sigma=2.0):
    if sigma <= 0: return img
    return cv2.GaussianBlur(img, (0, 0), sigma)

def calculate_center_of_mass_shift(img, geo, sigma=5):
    # [FIX] Use cv2 to match Multi version
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    blurred = blurred - np.min(blurred)
    cy_com, cx_com = center_of_mass(blurred)
    
    dy = geo['cy'] - cy_com
    dx = geo['cx'] - cx_com
    return float(dy), float(dx)

# === 2. Geometric Transformations ===

def shift_image(img, geo, dy, dx):
    h, w = geo['H'], geo['W']
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

def rotate_image(img, geo, angle):
    h, w = geo['H'], geo['W']
    center = (geo['cx'], geo['cy'])
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

def transform_final_image(img, geo, angle, dy, dx):
    h, w = geo['H'], geo['W']
    angle = normalize_angle(angle)
    
    # 1. Rotate
    M_rot = cv2.getRotationMatrix2D((geo['cx'], geo['cy']), angle, 1.0)
    
    # 2. Total Shift
    total_dx = -dx
    total_dy = -dy
        
    # Combine (Approximation for speed, or strict two-step)
    # Let's do strict two-step to avoid matrix math errors manually
    rot_img = cv2.warpAffine(img, M_rot, (w, h), flags=cv2.INTER_LINEAR)
    M_shift = np.float32([[1, 0, -total_dx], [0, 1, -total_dy]])
    
    return cv2.warpAffine(rot_img, M_shift, (w, h), flags=cv2.INTER_LINEAR)

# === 3. Alignment Logic ===

def get_coarse_angle_fourier_mellin(img, ref, geo):
    h, w = img.shape
    window = geo['window']
    
    F_img = np.abs(fftshift(fft2(img * window)))
    F_ref = np.abs(fftshift(fft2(ref * window)))
    F_img = np.log(F_img + 1)
    F_ref = np.log(F_ref + 1)
    
    center = (geo['cx'], geo['cy'])
    max_radius = geo['max_radius']
    
    polar_img = cv2.linearPolar(F_img, center, max_radius, cv2.WARP_FILL_OUTLIERS)
    polar_ref = cv2.linearPolar(F_ref, center, max_radius, cv2.WARP_FILL_OUTLIERS)
    
    prof_img = np.sum(polar_img, axis=1)
    prof_ref = np.sum(polar_ref, axis=1)
    
    # [FIX] Use FFT Correlation to match GPU logic
    pad_len = h
    prof_img_pad = np.pad(prof_img, (0, pad_len))
    prof_ref_pad = np.pad(prof_ref, (0, pad_len))
    
    fft_len = int(2**np.ceil(np.log2(2*h - 1)))
    fp_img = np.fft.fft(prof_img_pad, n=fft_len)
    fp_ref = np.fft.fft(prof_ref_pad, n=fft_len)
    
    corr = np.real(np.fft.ifft(fp_img * np.conj(fp_ref)))
    shift_idx = np.argmax(corr)
    
    # Circular Lag Logic
    if shift_idx > fft_len // 2:
        lag = shift_idx - fft_len
    else:
        lag = shift_idx
        
    angle_est = - (float(lag) / h) * 360.0
    return normalize_angle(angle_est)

def get_best_translation_fft(img, ref, geo):
    f_img = fft2(img)
    f_ref = fft2(ref)
    
    cc_spec = f_img * np.conj(f_ref)
    cc_map = np.real(ifft2(cc_spec))
    cc_map = fftshift(cc_map)
    
    cy, cx = np.unravel_index(np.argmax(cc_map), cc_map.shape)
    
    # Use Integer Center for FFT shift reference (Standard FFT behavior)
    dy = float(cy) - geo['cy_int']
    dx = float(cx) - geo['cx_int']
    max_cc = float(cc_map[cy, cx])
    
    return dy, dx, max_cc

def check_180_ambiguity(img, ref, angle_candidate, geo):
    rot_1 = rotate_image(img, geo, angle_candidate)
    _, _, score_1 = get_best_translation_fft(rot_1, ref, geo)
    
    angle_180 = angle_candidate + 180
    rot_2 = rotate_image(img, geo, angle_180)
    _, _, score_2 = get_best_translation_fft(rot_2, ref, geo)
    
    if score_2 > score_1:
        return normalize_angle(angle_180), score_2
    else:
        return normalize_angle(angle_candidate), score_1

def refine_subpixel_parabolic(scores, center_idx, step):
    n = len(scores)
    if center_idx <= 0 or center_idx >= n - 1:
        return 0.0
    y_l = scores[center_idx - 1]
    y_c = scores[center_idx]
    y_r = scores[center_idx + 1]
    denom = y_l - 2 * y_c + y_r
    if abs(denom) < 1e-6: return 0.0
    delta = 0.5 * (y_l - y_r) / denom
    return delta * step

def fine_alignment_search(img, ref, coarse_angle, geo, search_range=5, step=1):
    best_score = -float('inf')
    best_params = {'angle': coarse_angle, 'dy': 0, 'dx': 0, 'score': -float('inf')}
    angles = np.arange(coarse_angle - search_range, coarse_angle + search_range + step, step)
    scores = []
    
    for ang in angles:
        rot_img = rotate_image(img, geo, ang)
        dy, dx, score = get_best_translation_fft(rot_img, ref, geo)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_params = {'angle': ang, 'dy': dy, 'dx': dx, 'score': score}
            best_idx = len(scores) - 1
            
    delta = refine_subpixel_parabolic(scores, best_idx, step)
    best_params['angle'] = normalize_angle(best_params['angle'] + delta)
    return best_params

