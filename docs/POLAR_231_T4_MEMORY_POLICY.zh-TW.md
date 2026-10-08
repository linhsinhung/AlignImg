# AlignImg 2.3.1 T4：polar resident／streaming

本文件記錄 T4 的資料供應與配置預算。T3 `_gpu_polar_batch_solver` 本體、CPU sampling authority、mirror 修正、搜尋數學、public API、既有 tolerance 均不變。T4 當時以 2.3.0 版本號驗證；T7 的正式 2.3.1 artifacts 已完成實機驗收，詳見 [封版狀態](RELEASE_FREEZE_2_3_1.md)。

## Policy

- 目標 batch = `min(N, requested batch)`；未指定時保留 automatic soft cap 256。
- 完整 FP32 spatial particles＋共用配置＋目標 batch 工作區皆能放下，才選 resident。resident 僅活在一次 inference 呼叫，不跨 iteration 保留。
- 否則選 streaming，以預算可容納的最大 particle batch 運算。每批只上傳 B 張影像、B 列 grids／valid masks／priors／fixed-reference indices；source indices 改為 `0..B-1`，但共用同一個 batch solver。
- references、polar offsets 本次呼叫共用；只下載 B×11 的精簡結果並依原 particle 順序填回 host。完整 correlation maps 不下載。
- Streaming GPU 工作區依 B、K、box、angles、centers 變動，不隨 N 配置完整 particle stack。完整 inputs、grid 與結果仍可存在 host RAM；不是磁碟 streaming。

## 共同預算

```
workflow_remaining = initial_workflow_budget - live_Fourier_cache
current_available = current_free_VRAM - reserve
polar_budget = max(0, min(workflow_remaining, current_available))
```

`current_free_VRAM` 已反映 cache 占用，不能再扣一次。workspace 只提供初始 budget 與 live cache bytes；polar 配置和 batch 工作區另列。量測 free VRAM 前，沿用既有行為釋放 CuPy pool 未使用的 blocks。

若 streaming batch 1 不可行，依序釋放可重建的 update、scoring cache，每次釋放後重新量測／規劃。一旦可行就停止釋放；仍不可行則在 polar upload 前拋出明確的 MemoryError。

## 逐項 inventory

`alignimg_gpu/_polar_memory.py` 記錄共用與每 particle 配置，包含：

- FP32 spatial images／rings／normalization scratch；
- FP64 polar offsets／centers／priors；
- complex64 FFT、conjugates、einsum layout materialization、cross spectra；
- FP32 correlation curves 與 peak selector 的 FP64 copy；
- fixed-reference gather、expanded reference slots、ordering／runner-up／winner scratch；
- reference、full-batch、tail-batch 的 cuFFT workspace 預留。

cuFFT 預留使用 [NVIDIA CUDA 12.8.1 文件](https://docs.nvidia.com/cuda/archive/12.8.1/cufft/index.html) 的 workspace 上界，不沿用整份工作區乘任意倍率。這是保守容量估計，不是實測值；實際 batch 可能因此小於 requested 512。先由 server 取得容量與耗時證據，不能把這個上界當作平常的實際用量。

計畫記錄包含 `particle_storage_policy`、`target_particle_batch`、`batch_size`、`live_cache_bytes`、兩種可用預算、fixed/item components、`estimated_peak_bytes`。估計 peak 不含另列的 live Fourier caches。既有 profiler 的 pool allocation peak、device usage samples、H2D/D2H bytes 與時間保持分開。

## 驗證範圍

- 純尺寸模擬：8 GiB、80%、128×128、N=1,000／105,000／140,000，不配置巨大影像陣列。
- resident 選擇的精確 budget 邊界；requested/automatic batch；cache 不重複扣款；外部 VRAM 使用量；batch-1 拒絕；cache eviction 順序。
- CPU/emulation 與實機：batch 1／7／256／512、尾批、fractional center、mirror、fixed/open priors 的既有契約。
- `tools/polar_231_streaming_validation.py` 沿用 T3 的 515-particle fixture、frozen T2 solver、hash、comparison、JSON helpers。每組 warm-up 一次、unprofiled 三次、額外 profile 一次。
- runner-scoped 限額必須落在同資料的 `streaming minimum < cap < resident minimum`，並與真實可用預算取較小值；結束或例外時還原，不改 AlignmentConfig 或 CUDA 全域環境。
- 實機檢查測得的 pool live peak 不超過估計，觀察到的 device 增量不超過 budget；device samples 只是 peak 下界，不宣稱連續硬體峰值證明。受限預算驗證不能代表實體 8 GiB 卡的效能。

## 後續階段狀態（2026-10-07）

T4 streaming 與 T5 OOM recovery 已有 server 驗收紀錄。本機仍無 CUDA，不能用 CPU/emulation 通過替代新版 artifact 的 server gate。

T5 已加入完整 OOM recovery（resident → streaming、cache eviction、batch 減半、traceback 釋放、整次重跑）。單純 budget 測試不取代 allocator OOM recovery 測試。

T6 以已記錄的桌面記憶體與舊 baseline 相容性例外接受；原始 FAIL／inconclusive 保留，不改寫成 PASS。T7 已以正式安裝的 2.3.1 artifacts 通過 CUDA／CuPy conformance、resources、representative、quick 及 472 項產品測試；本輪 resources 未使用新的例外。CUDA 3050-particle 中位耗時 7.204 秒，通過原始 7.516 秒 baseline 的 5% 上限。正式證據與測試來源 hash 見封版紀錄；後續文件狀態更新不改動已測試 artifacts。
