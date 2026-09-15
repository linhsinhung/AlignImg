# AlignImg 2.1.0.dev3：階段 2 GPU workspace 與 VRAM 管理

依據 [分階段計畫](PERFORMANCE_OPTIMIZATION_2_1.zh-TW.md)。`2.1.0.dev2` 的 CUDA A/B 已通過，但 quick 找到 adaptive rescue subset 與 workspace source identity 的衝突；`2.1.0.dev3` 已修正為從完整常駐 scoring FFT 依 rescue indices 取值。Linux RTX 3090 的 CUDA／CuPy A/B、強制 host-streaming 與 1000-particle RF → feedback workflow 均已通過，Stage 2 已於 2026-09-08 封存；正式判定見 `validation-results/performance/stage-2/ACCEPTED.md`。

## 實作範圍

本階段不改公開 API、pose convention、candidate scoring、posterior、reference update 算式或輸出 dtype。CUDA 與 CuPy workflow 在單次 alignment engine 呼叫內建立一個私有 workspace，分別管理：

- 經標準化、供 candidate scoring 使用的 particle complex64 FFT；
- 未標準化、供 full／half-A／half-B Fourier M-step 使用的 raw particle complex64 FFT。

兩份資料各上傳一次，跨 iterations 與三次 M-step 重用。Workspace 不跨 workflow、不成為 public session，也不留下全域無上限 cache；engine 結束或例外時都會關閉。`apply_final_pose_to_raw=True` 的 raw average 位於 engine 返回之後，因此一定先釋放 workspace，再依既有 batch 設定套用 final pose，兩者不會同時占用這批常駐 FFT。

Workspace 在建立時取得一次可用／總 VRAM，沿用 `memory_fraction` 與至少 256 MiB reserve，兩種 FFT cache、固定陣列和 batch 暫存共用同一預算。正常路徑若容納不下 cache 加一個工作單位，該角色改用 host streaming。實際 allocation 或運算 OOM 時，依序採用：

1. 移除相關常駐 cache，重試同一 batch；
2. 若仍失敗，batch 對半縮小；
3. batch 1 仍失敗才回報 OOM。

縮批後的上限會覆蓋 workflow 初始 batch，不會被初始 `batch_size=512` 拉回。釋放是移除 workspace 的 device references、將 block 交回 CuPy memory pool，不在每個 workflow 強制清空全域 memory pool。

結果 metadata 新增 `gpu_workspace`，memory records 新增 `workflow_workspace` 的 create／evict／release 與兩種 `workspace_role`；profiling counters 新增各角色的 request、upload、hit、stream／OOM fallback／eviction，以及 workspace release bytes。`tools/performance_validation.py` 會把 workspace summary 寫入 JSON。

## 本機驗證

- 完整 pytest：190 passed、20 skipped；跳過項目包括本機不存在的 CUDA／CuPy 與選配資料。
- CPU quick：18／18 passed。
- 純 Python fake-device tests 覆蓋兩種 cache 的共享預算、一次上傳與重用、低 VRAM streaming、allocation OOM fallback、cache eviction、縮批覆蓋、完整釋放與連續 workflow 的生命週期。
- 已驗證 frozen dev1 source archive 與 native source；本階段 native CUDA kernel 未改，只有 `2.1.0.dev3` version stamp。

本機為 macOS arm64，不能代替 server CUDA 驗收。

## 2.1.0.dev2 Server 結果與修正

RTX 3090 上的 dev1→dev2 CUDA smoke／representative A/B 共九個 cases 均通過，36 個 NPZ artifact hashes 全部符合報告，完整 E-step 與最終結果最大數值差皆為零。H2D payload 在小型案例減少 73,728 或 221,184 bytes，在 384／local 案例減少 40,960,000 或 100,663,296 bytes；D2H payload 全部不變。兩種 cache 均維持 device policy、batch 512、無 eviction／streaming／OOM retry，且每個 workflow 關閉後完整釋放。

CUDA smoke median 相對 dev1 為 -5.5% 至 -20.2% 的改善，adaptive 改善 1.0%，adaptive-whitened 慢 1.8%，仍在 5% gate 內。Representative cases 改善 2.1% 至 15.1%。這證明主要 workspace 路徑有效，但 quick 僅 17／18：三輪 `adaptive_rescue` 的第二輪會建立一顆 particle 的 fallback subset，dev2 將這個合法 view 誤判為 scoring source 被替換。

dev3 不放寬 cache identity。Fallback 改為保存 rescue indices，以它們直接索引完整、原有的 device scoring FFT；host-streamed 情況仍使用 subset host data。修正不改 rescue candidate、posterior 或 M-step，且避免額外上傳 subset FFT。dev2 報告與 delivery 保留，不覆寫；正式 acceptance 以 dev3 重跑結果為準。

## dev1 → dev3 正式 A/B

`tools/stage2_validation.py` 會驗證並展開已封存的 `stage-1/dev1-delivery`，在相同 process environment、輸入、config、seed 與 dev3 native binary 下依序執行 frozen dev1 Python GPU engine 和目前 dev3。它會檢查：

- 所有完整 E-step arrays、poses、references、class averages、responsibilities、FRC 與 histories 維持既有 tolerance；
- scoring 與 update FFT 各只上傳一次，hit 次數分別為 `iterations - 1` 與 `3 × iterations - 1`（half-set diagnostics 開啟時）；
- workspace 關閉、兩份 cache bytes 完整釋放、峰值不超過 shared budget；
- 標準 RTX 3090 workload 無 streaming、eviction 或 OOM retry，所有 workspace plans 維持 requested batch 512；
- H2D calls 與 bytes 必須比 dev1 下降，D2H bytes 不變。

CUDA median 慢於 dev1 超過 5% 會標記 `repeat_required`；CuPy 是 portable fallback，仍記錄 timing，但只以數值、workspace 與傳輸契約作 gate。

## Server 部署與驗證

在既有 `align-dev` 環境、專案根目錄執行。需完整上傳 `src`、`packages/alignimg-gpu`、`tools`、`tests`、`docs`、root `pyproject.toml`，並保留 `validation-results/performance/stage-1/dev1-delivery/` 及 Stage 0 fixture。

```bash
python -m pip install --force-reinstall --no-deps -e .

CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" \
python -m pip install -v --force-reinstall --no-build-isolation --no-deps \
  packages/alignimg-gpu/dist/alignimg_gpu-2.1.0.dev3.tar.gz
```

請依序執行，並避免其他程序同時使用 GPU：

```bash
python tools/server_validation.py \
  --suite quick --backend cuda --batch-size 512 \
  --output validation-results/performance/stage-2/dev3-cuda-quick.json

python tools/stage2_validation.py \
  --suite smoke --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-2/dev3-cuda-smoke-ab.json

python tools/stage2_validation.py \
  --suite representative --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-2/dev3-cuda-representative-ab.json

python tools/stage2_validation.py \
  --suite smoke --backend cupy --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-2/dev3-cupy-smoke-ab.json

python tools/stage2_validation.py \
  --suite representative --backend cupy --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-2/dev3-cupy-representative-ab.json
```

上述測試通過後，再執行本階段的 1000-particle RF → feedback milestone，避免在基礎傳輸契約尚未確認前消耗長時間：

```bash
python tools/re2dc_70s_final_validation.py \
  --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-2/dev3-re2dc-70s-final-n1000.json
```

每個 A/B 命令會產生總報告、`.baseline.json`、`.current.json` 及每個 case 的 `.inputs.npz`／`.result.npz`。Stage 2 通過前不覆寫 Stage 1 檔案，也不開始 Stage 3 batching。

## 最終驗收結果

- 本機 pytest：190 passed、20 skipped；Server CUDA quick：18／18 passed，adaptive rescue regression：1／1 passed。
- CUDA 與 CuPy smoke／representative A/B 全數通過，完整 candidates、posterior 與輸出最大數值差為 0。CUDA representative 相對 dev1 為 0.04% 至 6.47% 改善／持平，無 case 退步超過 5%。
- 正常 3090 路徑維持 batch 512，scoring 與 raw-update FFT 各上傳一次後重用；D2H payload 不變，representative H2D payload 每 case 減少 40,960,000 或 100,663,296 bytes。
- 強制低記憶體測試取得零 workspace budget，scoring 與 update 全部走 host streaming、所有 memory plan 降為 batch 1，連續四次 workflow 成功；結果 NPZ 與 device-cache global 結果逐位元相同。
- 1000-particle K=10、15-round RF 與兩條 5-round feedback workflow 完成。相對 1.10.0 baseline，所有記錄的科學結果完全相同；RF 為 36.32 秒，fixed feedback 為 431.69 秒，corrective feedback 為 3699.60 秒。

Corrective feedback 的單次時間比舊 baseline 慢，但不是本階段的正式 A/B gate；其 workspace、batch、傳輸生命週期與數值均正常。這項逐 particle adaptive-posterior 熱點明確交由 Stage 3 的跨-particle batching 處理，不在 Stage 2 疊加特例。
