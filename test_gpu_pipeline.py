import unittest
import numpy as np

import alignimg_api as api

try:
    import cupy as cp
    import align_utils_gpu as aug
    HAS_GPU = True
except Exception:
    HAS_GPU = False


def _legacy_gpu_single_iter(X_cpu, initial_ref_cpu, mask_diameter=None, batch_size=4):
    """Reference implementation equivalent to the pre-optimization GPU loop (1 iter)."""
    N, H, W = X_cpu.shape
    geo = aug.GeometryContext((H, W))

    state_params = np.zeros((N, 4), dtype=np.float32)
    com_offsets = np.zeros((N, 2), dtype=np.float32)

    current_ref_gpu = cp.array(initial_ref_cpu, dtype=cp.float32)
    current_ref_gpu = aug.apply_circular_mask_batch(current_ref_gpu, geo, diameter=mask_diameter)

    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        batch_gpu = cp.array(X_cpu[start_idx:end_idx], dtype=cp.float32)
        offsets_gpu = aug.calculate_com_batch(batch_gpu, geo)
        com_offsets[start_idx:end_idx] = cp.asnumpy(offsets_gpu)

    lp_sigma, is_global_search, search_range, search_step = api.iter_params(0, 1)

    ref_masked = aug.get_circular_mask(geo, diameter=mask_diameter) * current_ref_gpu
    ref_match = aug.apply_lowpass_batch(ref_masked, sigma=lp_sigma)
    ref_accum_gpu = cp.zeros((H, W), dtype=cp.float32)

    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        curr_bs = end_idx - start_idx

        img_batch = cp.array(X_cpu[start_idx:end_idx], dtype=cp.float32)
        bias_y = cp.array(com_offsets[start_idx:end_idx, 0])
        bias_x = cp.array(com_offsets[start_idx:end_idx, 1])
        curr_angle = cp.zeros(curr_bs, dtype=cp.float32)

        img_centered = aug.warp_affine_batch(img_batch, geo, cp.zeros_like(curr_angle), bias_y, bias_x)
        mask = geo.get_circular_mask(diameter=mask_diameter)
        img_match = aug.apply_lowpass_batch(img_centered * mask, sigma=lp_sigma)

        raw_ang = aug.get_coarse_angle_fm_batch(img_match, ref_match, geo)
        center_ang, _ = aug.check_180_ambiguity_batch(img_match, ref_match, raw_ang, geo)

        best = aug.fine_alignment_search_batch(
            img_match,
            ref_match,
            center_ang,
            geo,
            search_range=search_range,
            step=search_step,
        )

        res_dy, res_dx = best["dy"], best["dx"]
        final_ang = best["angle"]

        rad = cp.deg2rad(-final_ang)
        cos_r, sin_r = cp.cos(rad), cp.sin(rad)
        res_dx_pre = res_dx * cos_r - res_dy * sin_r
        res_dy_pre = res_dx * sin_r + res_dy * cos_r

        new_by = bias_y - res_dy_pre
        new_bx = bias_x - res_dx_pre

        new_params = np.stack(
            [cp.asnumpy(final_ang), cp.asnumpy(new_by), cp.asnumpy(new_bx), cp.asnumpy(best["score"])],
            axis=1,
        )
        state_params[start_idx:end_idx] = new_params

        aligned_batch = aug.warp_affine_batch(img_batch, geo, final_ang, new_by, new_bx)
        ref_accum_gpu += cp.sum(aligned_batch, axis=0)

    new_ref = ref_accum_gpu / N
    new_ref = (new_ref - cp.mean(new_ref)) / (cp.std(new_ref) + 1e-8)
    current_ref_gpu = new_ref * geo.get_circular_mask(diameter=mask_diameter)

    return cp.asnumpy(current_ref_gpu), state_params, com_offsets


class TestGpuPipeline(unittest.TestCase):
    def test_cpu_path_unchanged(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(12, 24, 24)).astype(np.float32)
        init_ref = X.mean(axis=0)

        ref, _, params, meta = api.run_alignment(X, init_ref, num_iterations=1, n_jobs=1, use_gpu=False)

        self.assertEqual(meta["engine"], "cpu-serial")
        self.assertEqual(ref.shape, init_ref.shape)
        self.assertEqual(params.shape, (12, 4))

    @unittest.skipUnless(HAS_GPU, "CuPy/GPU unavailable")
    def test_optimized_gpu_matches_legacy_reference(self):
        if cp.cuda.runtime.getDeviceCount() < 1:
            self.skipTest("No CUDA device available")

        rng = np.random.default_rng(1)
        X = rng.normal(size=(8, 32, 32)).astype(np.float32)
        init_ref = X.mean(axis=0).astype(np.float32)

        ref_opt, _, params_opt, com_opt, _ = api.run_batch_alignment_gpu(
            X,
            init_ref,
            num_iterations=1,
            batch_size=4,
            profile_gpu=True,
        )
        ref_legacy, params_legacy, com_legacy = _legacy_gpu_single_iter(
            X,
            init_ref,
            batch_size=4,
        )

        np.testing.assert_allclose(com_opt, com_legacy, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(params_opt, params_legacy, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(ref_opt, ref_legacy, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
