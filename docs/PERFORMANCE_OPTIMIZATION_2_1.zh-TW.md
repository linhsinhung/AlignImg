# AlignImg 2.1：分階段效能與資源管理優化

紀錄日期：2026-09-06（Asia/Taipei）  
狀態：階段 0 至 7 已完成 RC 驗收；Stage 6 mixed 實驗不採用，Stage 7 RC1 已接受並進入正式 `2.1.0` package/native identity check。

後續進度（2026-09-06）：階段 0 已開始實作；交付與驗收狀態另見 [階段 0 紀錄](PERFORMANCE_STAGE_0_2_1.zh-TW.md)。以下保留原先確認的方案。

後續進度（2026-09-07）：階段 0 已驗收封存。階段 1 開始實作，交付及測試方式見 [階段 1 紀錄](PERFORMANCE_STAGE_1_2_1.zh-TW.md)。

後續進度（2026-09-07）：階段 1 已通過 clean-GPU CUDA A/B 與 CPU／CuPy conformance 後封存；正式判定見 `validation-results/performance/stage-1/ACCEPTED.md`。下一步為階段 2 GPU workspace 與 VRAM 管理。

後續進度（2026-09-08）：階段 2 已通過 workflow-scoped scoring／raw-M-step FFT cache、共享 VRAM budget、device-cache／host-streaming、CUDA／CuPy A/B 與 1000-particle RF → feedback 驗收後封存；正式判定見 `validation-results/performance/stage-2/ACCEPTED.md`。下一步為階段 3 adaptive 跨 particles 批次化。

後續進度（2026-09-08）：階段 3 使用 `2.1.0.dev5` 開發包；dev4 server preflight 找到並已修正 GPU metadata entry 的未定義變數。實作與驗證方式見 [階段 3 紀錄](PERFORMANCE_STAGE_3_2_1.zh-TW.md)。

後續進度（2026-09-08）：階段 3 已通過 clean-GPU CUDA／CuPy dev3→dev5 A/B、384-particle K=1／fixed-MRA 等價性與 3050-particle known-reference 驗收後封存；正式判定見 `validation-results/performance/stage-3/ACCEPTED.md`。下一步為階段 4 full／half-set 共用 M-step 累加。

後續進度（2026-09-08）：階段 4 使用 `2.1.0.dev6` 開發包，將 full statistics 定義為正規化前的 half-A／half-B sums 與 weights 之和，每個 candidate 只執行一次 transform；實作與 server 驗證方式見 [階段 4 紀錄](PERFORMANCE_STAGE_4_2_1.zh-TW.md)。

後續進度（2026-09-09）：階段 4 已通過 CUDA／CuPy A/B 與第二輪 1000-particle RF → feedback 驗收並封存。階段 5 的第一個 `2.1.0.dev7` 增量讓 native CUDA kernel 以 particle index 直接讀取 workflow-cached FFT，先獨立驗證移除 device gather 的收益；見 [階段 5 紀錄](PERFORMANCE_STAGE_5_2_1.zh-TW.md)。

後續進度（2026-09-09）：`2.1.0.dev7` 已通過數值、VRAM 與效能驗收。第二個 `2.1.0.dev8` 增量雖保持逐值一致並減少 74–79% rotation interpolation，但 clean-GPU 重測後 representative workload 仍慢 6.50–8.66%，因此不採用並撤回；Stage 5 主線保留 dev7 indexed path，下一個增量不建立在 grouped rotation reuse 上。

後續進度（2026-09-09）：第三個 `2.1.0.dev9` 增量以 dev7 為基礎，將 native indexed Fourier transform 與 FP64 weighted M-step accumulation 融合；不採用 dev8 grouped rotation reuse。此增量先移除 candidate-sized transformed stack 與獨立 CuPy scatter，再以 dev7 做 CUDA 數值、VRAM、傳輸與 throughput A/B。

後續進度（2026-09-09）：`2.1.0.dev9` 已通過 clean-GPU representative A/B 並接受為 Stage 5 新 baseline。Captured scientific arrays 相對 dev7 為零差異，local refine／pose-fixed MRA median 分別為 dev7 的 0.9897／0.9722，目標 M-step 約減少 71.5%／74.1%。8-particle global smoke 的 +1.18 ms 差異保留於驗證紀錄，但不代表實際規模 throughput regression。下一個 Stage 5 增量為 GPU final raw-average accumulation。

後續進度（2026-09-10）：`2.1.0.dev10` 開始實作 GPU final raw-average accumulation。它只影響 `apply_final_pose_to_raw=True` 的輸出重建：保留既有 CUDA／CuPy spatial transform，以 FP64 在 GPU 依 hard assignment 與 inlier weight 累加，最後只下載 K 張 averages。輸入仍是記憶體中的 NumPy stack，未新增磁碟 I/O。是否保留此增量由 1000／3000／5000 particles 隔離 A/B 決定，最大案例至少快 5% 才認列實質收益。

後續進度（2026-09-10）：`2.1.0.dev10` 已通過 CUDA／CuPy quick、dev9→dev10 workflow A/B 與 1000／3000／5000 particles 隔離 scale A/B。Captured scientific arrays 與三個 scale outputs 的最大誤差均為零；5000-particle raw-average 為 dev9 的 0.1865 倍（約 5.36×），D2H 由 327.68 MB 降至 0.655 MB。Representative end-to-end 為 dev9 的 1.0083 倍，仍在 5% gate 內。dev10 正式接受，Stage 5 完成並封存。

後續進度（2026-09-11）：階段 6 使用 `2.1.0.dev11` 實驗 GPU Fourier M-step mixed accumulation：每個 VRAM batch 先以 FP32 累加 K 組部分和，再合併至 workflow 持有的 FP64 sums；weight denominator 維持 FP64。FP64 仍是預設及權威路徑，mixed 不影響 scoring、posterior、CPU、spatial M-step 或 final raw-average。實作與 server 驗證方式見 [階段 6 紀錄](PERFORMANCE_STAGE_6_2_1.zh-TW.md)。

後續進度（2026-09-11）：dev11 通過 CUDA／CuPy quick、極端數值與完整 scientific-array tolerance，但 mixed 沒有實質 throughput 收益。Local refine／pose-fixed MRA end-to-end 分別為 FP64 的 1.0027／0.9962 倍，而目標 shared M-step 反而為 1.5572／1.2947 倍。原因是每個 batch 的 K-sized FP32 partial zeroing 與 FP64 merge 成本大於 FP32 atomic 的收益。依預定 gate 不執行 1000／3050-particle 延伸測試，mixed 分支與公開設定已撤回；Stage 5 dev10 FP64 engine 繼續作為 accepted baseline。

後續進度（2026-09-11）：階段 7 使用 `2.1.0rc1`，不修改 dev10 alignment 計算檔。Release preflight 逐檔驗證 compute source 與 dev10 snapshot 相同，並檢查安裝中的 core、Python GPU package、native CUDA extension、production defaults，以及 rejected mixed option 確實未公開。正式 gate 固定為 5000 particles、K=10、20 iterations、seeds 0／1／2；結果帶回檢討後才決定是否凍結正式 `2.1.0`。

後續進度（2026-09-11）：RC1 已通過 CUDA／CuPy quick、representative 與正式 5000-particle 三-seed gate。三組 assignment 對 frozen baseline 的 ARI 均為 1.0，occupancy、responsibility 與 RELION diagnostics 相同；median runtime 從 429.322 秒降至 220.544 秒，約 1.95× throughput。RC1 正式接受，已只變更版本標記並建置 `2.1.0` artifacts；最後只需確認 server 安裝的 core、GPU package 與 native extension identity。

## 目標與工作節奏

保留目前 alignment 數學與公開 workflow，以減少重複計算、資料搬移與 GPU 等待為主要目標。混合精度納入後段獨立實驗，通過驗收後仍為選配，預設保持現有精度。

每階段固定採用：

**修改 → 本機測試 → 交付 server 測試包 → CUDA 驗證 → 帶回 JSON／artifacts 檢討 → 通過後封存，才進下一階段。**

每次只處理一類主要變動。失敗時在該階段修正並重測，不疊加下一階段，也不透過縮小搜尋範圍、減少 iterations 或降低診斷頻率換取速度。

## 分階段改動與驗收

| 階段 | 改動範圍 | 該階段驗收重點 |
|---|---|---|
| **0．封存效能 baseline、補齊量測** | 封存目前已驗證、包含 raw-average 選項的原始碼；加入分階段計時、資料搬移計數與記憶體紀錄。建立固定輸入及單輪 E／M-step fixtures。 | Profiling 開關不影響數值結果；報告可區分 CPU、GPU、傳輸、FRC 與輸出時間；取得同環境暖機後 baseline。 |
| **1．低風險計算重用** | 快取 particle／reference polar FFT、固定 frequency grids、center phase 與可重用的 reference norms。CPU、CUDA、CuPy 共用正確的失效規則。 | 候選排序、scores、posterior 與輸出維持既有容許誤差；重複 FFT／建表次數確實下降。跑 quick 與 384-particle 的 global／fixed-MRA。 |
| **2．GPU workspace 與 VRAM 管理** | 建立一次 workflow 內共用的 GPU workspace，分別保留 scoring FFT 與 raw M-step FFT；跨 iterations／halfsets 重用。統一 cache、暫存、workspace 與 raw averaging 的預算管理。 | 快取與串流模式結果一致；cache 不再每輪重傳；正常 3090 工作負載維持 batch 512。驗證預算不足、縮批、釋放與連續呼叫；完成第一輪 1000-particle workflow。 |
| **3．Adaptive 跨 particles 批次化** | 多顆 particles 一起做 coarse scoring，再批次建立與評估 fine candidates。先保留 CPU posterior controller 的數學邏輯，減少逐顆 GPU 呼叫與下載等待。 | Coarse／fine cells、posterior-mass selection、tie ordering、rescue 行為與完整 M-step posterior 保持一致。跑 adaptive quick、384-particle adaptive／K=1 等價性與 3050-particle known-reference。 |
| **4．Full／half-set 共用 M-step 累加** | 同一次粒子變換同步累加 full、half A、half B 的未正規化加權和，避免 half-set 再處理一次完整粒子量。CPU 與 GPU 都實作。 | 與舊 updater 比對 weights、references、center shifts、FRC curves、stable cutoff 及低 occupancy masking；特別驗證非零中心位移、mask、lowpass 與空 component。完成第二輪 1000-particle workflow。 |
| **5．Native CUDA 熱點整合** | 以 particle index 直接讀取 cached FFT，減少 broadcast／gather 複製；重用相同 angle／mirror 的 rotation。實作分塊加權 reduction，減少完整影像暫存與 scatter；保留 FP64 累加。Raw average 改成 GPU 累加後只下載 K 張 averages。 | 各項改動分別測試後才整合；CUDA／CuPy／CPU conformance、完整 posterior、raw-average 契約不變；量測中間陣列峰值、傳輸量及實際吞吐量。 |
| **6．混合精度選配實驗** | 在已驗證的 Fourier M-step reduction 上加入 FP32 分塊部分和、FP64 最終累加。與階段 5 的 FP64 路徑直接 A/B。 | 固定 candidate／posterior 的單輪誤差先通過，再跑 384、1000 與 3050-particle tests；比較高動態範圍、訊號抵消、低權重與 batch 改變。無穩定收益或誤差不合格，就不納入正式選配。 |
| **7．整體驗收與 release freeze** | 停止局部改動，執行正式比較，整理安裝、效能與資源管理文件，建置 release artifacts。 | 比較原始 baseline 與 dev10 優化 FP64 engine；完成 5000 particles、K=10、20 iterations、seeds 0／1／2，再封存 2.1。Stage 6 mixed 已拒絕，不進入 release candidate。 |

階段 4 必須保留現行中心修正前後的座標與處理順序；不能直接把最後輸出的 half averages 合併當成等價實作。

## 介面與實作界線

- RF／global／refine／class-feedback 的函式簽名與 `AlignmentResult` 既有資料契約不變；`apply_final_pose_to_raw=False` 維持預設。
- 新增 `AlignmentConfig.profile_execution=False`。啟用後提供分階段效能 metadata；舊計時欄位保留原語意，新增完整 iteration／workflow 時間，避免既有報告誤讀。
- Stage 6 曾實驗 `gpu_accumulation_precision`，但未通過效能 gate，已從 `AlignmentConfig` 與主線實作撤回。正式 2.1 只保留 dev10 FP64 accumulation engine。
- GPU workspace 僅限單次 workflow、單一 device 使用，結束後釋放所持資源；不新增公開 session object，也不建立無上限的全域資料快取。
- VRAM 管理維持 `memory_fraction=0.8`。不足一個工作單位時明確回報，不能把 batch 1 當作安全保證；記憶體紀錄區分已追蹤配置峰值與裝置使用量採樣，並揭露外部程序影響。
- 本輪保持單一 GPU、單一主要執行 stream。多 GPU、多 stream 傳輸重疊、大型 I/O／GUI 重構與新的分類機制延後。

## 分層驗證與通過規則

### 每階段必跑

- 本機完整 pytest；新增該階段針對性的回歸測試。
- Server CUDA quick；CuPy 跑受影響區塊的 conformance。正式 CUDA 驗收不得靜默 fallback。
- 相同輸入、config、seed，比較上一個已通過階段及階段 0 baseline。
- 固定 class assignments 不變、responsibilities 正規化、結果 finite、K=1／fixed-MRA 等價性與 raw-average 開關不影響 inference 的契約持續成立。
- 所有 accepted FP64 路徑都必須通過既有 CPU／GPU tolerance，不為了讓優化過關而放寬既有測試。另記錄 relative error，避免只靠 reference correlation 掩蓋振幅差異。

### 真實資料分工

- **384 particles**：檢查 RELION pose、homogeneous RF、fixed-MRA 與 adaptive refinement。
- **3050-particle local stack**：檢查 K=1 refinement 熱點及 soft／raw average 輸出。
- **1000 particles**：已在 workspace 與共用 M-step 等 accepted 關鍵里程碑驗證 RF → feedback 的完整流程；rejected mixed experiment 未進入此 gate。
- **5000 particles、三 seeds**：最後正式驗收才執行，不在每階段重跑。

真實資料報告列出 gauge-aware pose error、reference 差異、NCC、occupancy 與 FRC trajectory。ARI 不作為效能優化的通過門檻；出現明顯退化時回查該階段，不新增 outlier constraints 補救。

### 效能比較

- 小型效能案例每個版本先暖機一次，再量測三次，使用 median 並保留原始數據；詳細 profiling 與正式 throughput 分開跑。
- 小於 5% 的耗時差異不宣稱加速。若代表性案例慢超過 5%，重測一次；仍退步則先修正，不直接推進。
- 同時檢查計算／傳輸次數等直接指標，避免把環境波動誤認為優化成果。
- 不預先承諾整體加速倍數；混合精度只有在數值合格且有可重現收益時才發布為選配。

## 交付與封存

以 **2.1.0** 為本輪目標版本；開發測試包使用 core／GPU 同步的 `2.1.0.devN`，每次 server 交付遞增，避免不同實作都顯示相同版本。

每次交付包含測試指令、原始碼與輸入 SHA-256、環境／build 資訊、增量 JSON，以及必要的 NPZ／MRCS。報告分開標示數值檢查、效能檢查及待人工檢視項目，不把 CLI 沒有 fail 當作全部驗收完成。

舊 baseline 不覆寫、不刪除；完成一階段後保留可還原的 source／build 與結果，再開始下一階段。
