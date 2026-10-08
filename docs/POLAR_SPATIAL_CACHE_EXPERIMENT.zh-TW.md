# Polar spatial particles 跨 iteration 常駐實驗

狀態（2026-10-08）：2026-10-07 的 server 驗證完成，依使用者決定保留並封存。
這是獨立的 source／evidence snapshot，不改 `2.3.1` 版本字串，也不覆寫原版發布檔。
驗證數據、效能限制與封存入口見 [封版紀錄](RELEASE_FREEZE_POLAR_SPATIAL.md)。

## 範圍

只延長 GPU 上 prepared spatial particles 的生命週期：從一次 polar candidate
inference，延長為同一次 alignment workflow。這是已正規化、套 mask 的 FP32
影像，不是 raw stack；不能供 final raw average 直接使用。

搜尋、mirror sampling、排序、pose convention、precision、public config 與 native
CUDA kernels 都不改。RF/global 使用 polar search 時可受益；quadratic refine
不使用這份 spatial cache。只有多 iteration、Fourier M-step 的 workflow 才考慮
常駐；舊 spatial M-step 維持原來的生命週期，不擴張其記憶體規劃。

## 生命週期與預算

- 沿用 workflow workspace 的 80% 預算，和 scoring/update Fourier caches 共用。
- 每輪仍重建 reference 與 translation grids；只重用不變的 particle spatial source。
- 完整 spatial stack 是否常駐，與 solver batch 多大是兩回事。常駐來源可供小 batch
  重複讀取，`batch_size=512` 仍只是使用者要求的上限。
- 第一次 admission 把完整 source 算成新增 fixed allocation。之後它算在 live cache；
  當前 free VRAM 已反映其占用，不重複扣除。
- 每輪重新量測。放不下時釋放 spatial cache，沿用原 resident/streaming 路徑；
  一旦釋放，本次 workflow 不再反覆嘗試建立它。
- Fourier M-step 優先。若 spatial cache 限縮更新的 batch 或使 Fourier cache 無法
  admission，先釋放 spatial，再重算 update plan。
- Candidate OOM 先清除失敗 frame 的 device references，釋放 spatial cache，再依
  2.3.1 的 streaming/cache eviction/batch halving 順序重試；不改搜尋數學。
- 正常完成及例外退出都由 workspace 關閉。GPU memory 回到 CuPy pool，不代表
  立即歸還作業系統；records 分開記錄估算與實測。

不新增跨 workflow cache、磁碟 streaming、public cache switch 或新的調參機制。

## 預期效益與取捨

Fast 3 最多省掉後兩輪的完整 spatial H2D，不能省 reference 更新、sampling、FFT
或 correlation。若搜尋本身占主要時間，整體加速可能很小。

Cache 也會吃掉可用 workspace，因此可能縮小 solver batch。驗證已同時記錄
上傳量、cache hits、各輪 batch、cache eviction 與 wall time；不能只看 H2D 減少
就宣稱更快。如果 cache 經常被 M-step 釋放，該 workload 對本優化不具代表性。

## A/B 與停止條件

`tools/polar_spatial_cache_validation.py` 在同一 process 中分別載入雜湊固定的
2.3.1 GPU Python backend 和當前實驗 backend，共用未修改的 native kernels。
原始封存 source、JSON、NPZ 與 artifacts 不覆寫。

- Smoke：mirror off/on 與 K=3 synthetic；先確認 upload/reuse、結果與完整生命週期。
- Representative：既有 384 fixed-MRA、1000 open MRA、3050 K=1 Fast 3 inputs。
- 各 variant warm-up 一次、正式三次及獨立一次 profiling；耗時比較不用 profiled run。
- 沿用既有 array/candidate tolerance、無 GPU fallback、完整 correlation-map D2H=0、
  workspace 正常關閉與 memory checks。
- 同 policy、同 batch 的重跑保持既有 deterministic 契約。自動 batch 若受可用 VRAM
  影響，另做固定 batch 複查，不能把 streaming 複查冒充 cached policy 的重現證據。
- Cache 未命中時不標成優化成功；wall median 退步超過 5% 暫停採納；改善至少 5%
  才標記為明確效益，其餘報告為中性結果。

Server 產品測試 515 passed、0 skipped，CUDA/CuPy quick 各 22/22；兩個 backend 的
smoke 與 representative A/B 全部通過既定數值與資源門檻。三輪中只上傳一次
spatial stack、命中兩次，representative 整體 H2D 少約 33%。耗時變化有限且依
workload 而異；不宣稱普遍加速，不再展開更多微調。本期效能工作到此收斂。
