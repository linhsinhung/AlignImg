# AlignImg 2.1.0.dev0：階段 0 交付與驗收

日期：2026-09-06。對應 [2.1 分階段計畫](PERFORMANCE_OPTIMIZATION_2_1.zh-TW.md)。

本階段只封存 baseline、加入觀測與建立測試工具；尚未進行 cache 重用、批次化、M-step 合併或混合精度優化。CPU／CUDA／CuPy 使用相同的候選與數學規則。2026-09-07 的 server 數值與執行驗收已通過；清理 FTP 合併後殘留的舊 source files 並確認來源一致後，才開始階段 1。

## 本機驗證與 frozen baseline

- 修改前 pytest：142 passed、13 skipped。
- 交付前 pytest：159 passed、15 skipped；另啟用 real-data integration test：1 passed。
- 新增 profiling、錯誤清理、allocation-hook accounting、版本／來源檢查及 runner tests。
- CPU quick：18/18 passed；performance smoke：global、adaptive、reference-free 均通過。
- Global／adaptive 單輪 E-step 與 full／half A／half B M-step，比對修改前 fixture 的數值差異為零（本機相同環境）。
- 本機沒有 NVIDIA CUDA；native CUDA／CuPy GPU 數值、事件與配置 hook 的實際運作仍需 server 驗證。CPU mock tests 不代表 CUDA 已通過。

正式的修改前基準放在 `validation-results/performance/stage-0/baseline-2.0.0/`：

| 檔案 | 用途／SHA-256 |
|---|---|
| `source.tar.gz` | 修改前 engine、build 設定、工具、測試與文件；`4e37ec8c3fc22fe653841db9ccf151997332a9ab886046b78e89cce7dda9b212` |
| `manifest.json` | 每個 source file 的 SHA-256、archive 與 fixtures hashes；source digest：`38be9779685a76bf0219673739214aecae238136b300d1b1450dd42a4a8dc031` |
| `global.npz` | 固定 images、references、class priors、E-step 與三次 M-step outputs；`8af856937ce59b45c339056c0bb78b29ff11a7e83c6a481182f9620c5f539708` |
| `adaptive.npz` | 同上，另包含 initial poses 與完整 adaptive posterior；`380085981d01e4362e44a8339d049f6f720b7012a94ae21e15be50a8b8044f0e` |

Archive 包含先新增但未接上 engine 的記錄工具／profiling module；當時的 workflow、engine、GPU backend 與 config 尚未修改。它不是原先發布的 2.0 sdist，因為還保留了後續已驗證的 raw-average 功能。

Frozen NPZ 放在 validation-results，不隨 wheel/sdist 打包。請把上述整個 baseline 資料夾一併傳上 server；不要在 server 重新生成它來取代原始 expected outputs。缺少或 hash 不符時 runner 會失敗。

## 新介面與量測語意

使用 `AlignmentConfig(profile_execution=True, ...)`，在 result 的 `metadata["performance"]` 取得量測。預設為 False；公開 workflow 函式簽名不變。`align_known_reference.py` 也新增 `--profile-execution`，會同時套用到 global 與 refine。

- `stages`：階層式呼叫次數、inclusive／exclusive wall time，以及 instrumented CUDA event spans。
- `iterations`：包含 half-set diagnostics 與 history 更新的完整 iteration wall time。原本 diagnostics 的 `seconds` 與 `gpu_total_seconds` 不改語意。
- `counters`：explicit H2D／D2H 呼叫與 payload bytes、native pose copies、polar angular FFT 次數、coarse／fine candidates 等。
- `gpu_memory`：本次呼叫中 CuPy allocation hook 追蹤的 pooled live allocation peak，另外保存 device／pool 的採樣峰值。

注意：parent／child 的 inclusive 時間會重疊，不可直接全部加總。Exclusive wall time 不是 CPU busy time；CUDA event span 也不等於 GPU utilization，可能包含 host submission 的間隔。Native pose 上傳時間包含在 transform span 內。資料搬移計數不涵蓋 driver/JIT/library 內部流量。

Allocation hook 涵蓋本次呼叫的 CuPy pool 配置與 pool reuse，不包含呼叫前已存在的 arrays，以及繞過該 pool 的 native CUDA／cuFFT 配置。Device usage 包含其他程序；採樣峰值只是峰值下界。本階段不將這些資料宣稱為 80% hard-limit 證明。

詳細 profiling 會增加 event、同步及 hook 開銷，因此只用來找工作量與等待來源。正式 throughput 必須使用另一組 `profile_execution=False` 的測量。

實作介面參考：[CuPy 14.2 MemoryHook](https://docs.cupy.dev/en/v14.2.0/reference/generated/cupy.cuda.MemoryHook.html)、[CuPy 14.2 Event](https://docs.cupy.dev/en/v14.2.0/reference/generated/cupy.cuda.Event.html)。

## Server 安裝

在 server 專案根目錄、已啟用的 `align-dev` 環境執行：

```bash
python -m pip install --force-reinstall --no-deps -e .

CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" \
python -m pip install -v --force-reinstall --no-build-isolation --no-deps \
  packages/alignimg-gpu/dist/alignimg_gpu-2.1.0.dev0.tar.gz
```

Performance runner 會檢查 core／GPU 模組、實際載入的 Python 原始碼 hashes，以及 native module 版本。CUDA 明確指定時不允許 fallback。即使 CUDA kernel 算式沒變，也需重建這次 native package，才能取得 dev0 的 binary version stamp。

## Server 驗證指令

依序執行，請勿把不同效能測試同時放在背景跑。維持 batch 512、memory fraction 0.8，不清空其他程序的 GPU 資源。若有其他運算佔用 GPU，先等該運算完成再收集正式時間。

```bash
python tools/server_validation.py \
  --suite quick --backend cuda --batch-size 512 \
  --output validation-results/performance/stage-0/dev0-cuda-quick.json

python tools/performance_validation.py \
  --suite smoke --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-0/dev0-cuda-smoke.json

python tools/performance_validation.py \
  --suite representative --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-0/dev0-cuda-representative.json

python tools/performance_validation.py \
  --suite smoke --backend cupy --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-0/dev0-cupy-smoke.json
```

Performance runner 每個案例執行一次 warm-up、三次 unprofiled 計時、一次 profiled run。輸入載入、hash 與 NPZ 輸出在 throughput 計時之外；三次結果與 profiling 結果會比較數值。CLI `--repeats` 可用於開發測試，但正式 server 驗收保留預設 3。

| Suite／case | 固定工作負載 |
|---|---|
| smoke：global | 8 張 24×24、K=2、一次 global iteration |
| smoke：adaptive | 同一固定 synthetic 輸入、一次 adaptive refinement，包含 final raw average |
| smoke：reference_free | 同一固定 synthetic 輸入、K=2、兩次 RF iterations；僅作執行／profiling conformance，不宣稱 RF 收斂 |
| representative：pose_fixed_mra | 原有 384-particle RELION benchmark、K=3、固定 class prior，由既有 RELION poses 開始一次 adaptive refinement |
| representative：local_global | `data/local/test_align.mrcs` 的前 256 張，對 `mu_aligned_mean.mrc` 做一次 global alignment |
| representative：local_refine | 相同前 256 張，讀取既有 `local-known-reference-raw-average/result.npz` 的 global poses 與 global reference，做一次 refinement 及 raw averaging |

Representative 沿用既有 prepared benchmark 與 local 資料，不重抽資料、不重新分類。Local refine 不重跑 initializer 來改變測試輸入。所有實際 images、references、priors、initial poses、index selection 及 config 會另存 `.inputs.npz` 並記錄 hash。

若缺少 local initializer，需補傳 `validation-results/local-known-reference-raw-average/result.npz`。缺檔會明確記錄 failed，不會改用臨時 reference 或略過後顯示全數通過。

可使用 `--only local_refine` 等選項單獨重測，但必須選擇新的 output 檔名；performance runner 不覆寫既有 report。重測時不要手動修改 frozen inputs 或 expected outputs。

## 帶回的檔案與判定

請帶回四份上述 JSON，以及 performance runner 產生的 `.inputs.npz`／`.result.npz`。Stage 0 不額外要求新的 MRCS：NPZ 已包含 soft references、class averages、poses、responsibilities、candidate arrays、reference history 與 half-set FRC。

- JSON 增量記錄各 case；初始化或 case 失敗會保留 traceback 並以非零 exit code 結束。
- GPU 記憶體 plans、OOM retries、來源與 native binary hash、library/compiler 環境會保存在報告。
- `performance_acceptance="baseline_only"` 表示這次只建立未優化基準，不宣稱加速。
- `scientific_review="pending"` 表示 CLI 數值檢查通過後，仍需檢視 server 結果；這不是正式 RF 科學驗收。
- 不在階段 0 執行 1000／5000-particle 完整 workflow，也不加入 mixed precision 參數。這些仍依主計畫在後續階段進行。

## 2026-09-07 Server 驗收結果與來源清理

- CUDA quick：18／18 passed；CUDA 與 CuPy smoke 各 3／3 passed。
- CUDA／CuPy smoke 的三份 result NPZ 均逐 byte 相同。
- Representative 的 local global、local refine 通過；pose-fixed MRA 最初只有一筆 float32 responsibility row sum 超過 `2e-6`，改用仍嚴格的 `5e-6` normalization tolerance 後通過。
- Frozen E／M-step fixture 的最大絕對誤差為 `2.38e-7`；profiled／unprofiled result 最大差異為零。
- Pose-fixed MRA 中位數為 20.924 秒（18.35 particle-iterations/s）；量測顯示主要瓶頸是 adaptive host controller 與大量小型 transfers，不是 VRAM。

FTP 上傳保留了 11 個本機已移除的舊檔，且 server 的 `THIRD_PARTY_NOTICES.md` 仍是舊版本。先以 binary mode 上傳目前本機的 `THIRD_PARTY_NOTICES.md` 與 cleanup script，然後在 server 專案根目錄執行：

```bash
bash tools/server_stage0_source_cleanup.sh --scan
bash tools/server_stage0_source_cleanup.sh --apply
```

`--apply` 只會把報告中已確認的路徑移到 timestamped `validation-results/drop/`，不會刪除。最後會驗證 notice、validator 與完整 source digest；任何其他差異都會回報失敗。清理後只需跑一次輕量 source check，不重跑完整 representative benchmark。
