# AlignImg 2.1.0.dev1：階段 1 計算重用

依據 [分階段計畫](PERFORMANCE_OPTIMIZATION_2_1.zh-TW.md)。本階段已於 2026-09-07 通過 server A/B 驗收並封存；正式 acceptance record 位於 `validation-results/performance/stage-1/ACCEPTED.md`。

## 實作與失效規則

- Prepared particle/reference stack 惰性保存 angular polar FFT。Particles 在同一 workflow 的 iterations 間共用；更新 references 時建立新的 prepared stack，其 FFT 隨之重建。Mirror 的 polar FFT 每顆、每輪計算一次，供允許的 references 共用。
- CPU 的 frequency grids、DFT-bin grids、integer-center phase 共用一個唯讀 LRU entry。Key 為 box size，最多保留一個 size 的五張 float64 表（40 × size² bytes）；切換 size 即替換。CUDA／CuPy 的 native transform 本來就直接計算 DFT indices 與 parity phase，沿用該路徑。
- CPU reference norms 依 prepared reference、prepared particle stack 與 weight-profile index 快取；GPU norms 限單次 inference 呼叫。Uniform NCC 的 K 個 norms 共用於所有 particles；whitening 僅保留目前 particle 的一份 norms，coarse/fine 共用，換 particle 即替換。Reference 更新、box/config 改變及新的 workflow 都重新建立相應資料。
- GPU global scorer 共用 uniform norms；particle-wise whitening 的 global scorer 保留逐批計算。Adaptive CUDA／CuPy 共用同一個 cache 規則。GPU memory plan 加入常駐 norm/weight 所需空間，沒有跨 workflow 保留 device arrays。
- 新增 `polar_fft_cache_bytes`、`frequency_table_builds`、`reference_norm_requests`、`reference_norm_cache_hits` 等 profiling counters。既有 `polar_angular_fft_calls`、`reference_norm_evaluations` 記錄實際計算次數。

Polar FFT 保留 NumPy 原有輸出 dtype；host cache 增量為 N × angular_samples × radial_bins × complex_itemsize，並隨 prepared stack 釋放。新資料不應就地修改 prepared stack 內部 arrays；更新流程應重新 prepare，既有 engine 即採此方式。

階段 0 的來源、報告與 baseline 不覆寫。階段 2 的跨 E/M/half-set GPU workspace、階段 3 的 adaptive batching 仍依原計畫後續實作。

## 本機與 Server 驗收範圍

本機交付前結果：pytest 181 passed、17 skipped；CPU quick 18／18 passed。五組 CPU A/B（含完整 E-step arrays）最大數值差異皆為零；384-particle global 的 poses、candidates、references 與 FRC 差異也為零，三次計時 median 從 13.778 秒降至 12.500 秒（約 9.3%）。這些是本機 CPU 觀察，CUDA／CuPy 仍待 server 驗收。

本機測試涵蓋 cached／uncached global、RF、refine 的兩輪結果、raster/Fourier scoring、mirror、whitening、reference 更新、不同 box size 與 cache 計算次數。沿用 frozen dev0 E/M fixtures，另以完整 dev0 source snapshot 執行同環境 A/B。

新工具 `tools/stage1_validation.py` 自動依序執行：

1. 驗證封存 source manifest 與 archive，於臨時目錄展開 dev0。
2. 在獨立 Python process 使用 dev0 core/GPU Python code，warm-up、三次計時、一次 profiling。
3. 在目前安裝的 dev1 執行完全相同的 inputs/config 與量測。
4. 驗證輸入／輸出 hashes、逐欄比較 poses、top-L/full E-step scores/posterior、assignments、references、raw averages 與 FRC。
5. 比較計算次數與 median time；慢於 baseline 超過 5% 時回報 `repeat_required`，需針對該 case 再跑一次。小於 5% 的差異不宣稱加速。

兩版共用目前安裝的 native binary。工具會逐檔確認 native sources 與 CMake 設定只差版本 stamp；本階段 kernel 算式相同。這個 A/B 比較的是 dev0/dev1 Python 計算重用，並非兩次 CUDA 編譯的效能差異。CUDA 明確指定時不能 fallback。

`--suite smoke` 包含 global、adaptive、RF、兩輪 mirror global、兩輪 whitened adaptive。`--suite representative` 包含 384 顆固定 class 的 global pose search、384 顆 fixed-MRA refinement，以及既有 local global／local refine 各 256 顆。Global case 的 RELION poses 只保存於輸入紀錄，未用來限制 global pose search。

## Server 部署

在既有 `align-dev` 環境與專案根目錄執行。以 binary FTP 上傳最新版 src、tools、tests、docs、root pyproject 及 GPU package；這次也有新檔，請包含完整相關資料夾。不要再執行限定 dev0 digest 的 `server_stage0_source_cleanup.sh`。

保留並上傳兩組 frozen 資料夾：

- `validation-results/performance/stage-0/baseline-2.0.0/`：小型 E/M-step fixture。
- `validation-results/performance/stage-0/dev0-source-clean-delivery/`：A/B 的原始 dev0 source archive 與 manifest，必須保留原檔。

另需現有 prepared 384-particle benchmark、`data/local` 與 `validation-results/local-known-reference-raw-average/result.npz`。

```bash
python -m pip install --force-reinstall --no-deps -e .

CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" \
python -m pip install -v --force-reinstall --no-build-isolation --no-deps \
  packages/alignimg-gpu/dist/alignimg_gpu-2.1.0.dev1.tar.gz
```

依序執行，避免同時跑其他效能測試：

```bash
python tools/server_validation.py \
  --suite quick --backend cuda --batch-size 512 \
  --output validation-results/performance/stage-1/dev1-cuda-quick.json

python tools/stage1_validation.py \
  --suite smoke --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-1/dev1-cuda-smoke-ab.json

python tools/stage1_validation.py \
  --suite representative --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-1/dev1-cuda-real-ab.json

python tools/stage1_validation.py \
  --suite smoke --backend cupy --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-1/dev1-cupy-smoke-ab.json
```

每個 A/B 命令產生總報告、`.baseline.json`、`.current.json`，以及兩版的 `.inputs.npz`／`.result.npz`。請將整個 `stage-1` 結果資料夾帶回；不需再取得 dev0 安裝包或切換 Conda 環境。

失敗、數值不一致或計算次數未下降會回傳非零 exit code 並保留錯誤 JSON。`repeat_required` 也回傳非零，代表效能需要重測；使用 `--only case_name` 與新的 output 名稱，不覆寫首次報告。Full candidate arrays 的複製只在 diagnostic run 執行，會增加該 run 的時間與 host memory；勿把 profiled wall time 當 throughput。

## 2026-09-07 Server 驗收與封存

最初下載的效能報告顯示 CuPy 有不一致的退步；後續確認當時有其他運算程序共用 GPU，因此保留原報告供診斷，但不納入正式 throughput 判定。清除額外 compute workloads 後，以 `dev1-clean-gpu-*` 完整重跑 CUDA／CuPy smoke 與 representative suites。

- CUDA quick 18／18 passed；clean-run 的 72 個 input／result NPZ hashes 全部符合報告。
- Dev0／dev1 的完整 E-step arrays 與所有輸出最大數值差異皆為零。
- CUDA smoke 五個 cases 改善 3.98% 至 20.98%；CUDA representative 四個 cases 均未出現超過 5% 的退步。
- Polar FFT accounting 在 local global 從 512 降至 257；reference norms 在 local refine 從 512 降至 1、fixed MRA 從 2304 降至 3。
- 所有 clean-run 均無 OOM retry，H2D／D2H calls 與 bytes 沒有增加；最大觀測 pool live allocation peak 為 560.66 MiB。
- CuPy conformance 完全通過。`global_mirror` tiny smoke（+10.40%，4.15 ms）與 `local_global`（+7.44%）保留 `repeat_required` timing note；依原計畫 CuPy 是 portable fallback/conformance，正式 GPU performance gate 為 CUDA，因此不為這兩項加入 backend 特例。

Frozen `2.1.0.dev1` source digest 為 `5e4cc3b501fda6578da4169e547c0a38769264acd124a53df9baa0f6db6f776a`，native CUDA binary digest 為 `d95cab0928e5d085ef86c1aecb70bac1e2691e3ff633457aeb53fac550aa1b11`。階段 1 至此完成，後續修改屬於階段 2。
