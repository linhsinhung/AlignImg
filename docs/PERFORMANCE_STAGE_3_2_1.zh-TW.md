# AlignImg 2.1.0.dev5：階段 3 Adaptive 跨 particles 批次化

依據 [分階段計畫](PERFORMANCE_OPTIMIZATION_2_1.zh-TW.md)。Stage 2 已確認 GPU workspace 能讓 scoring／raw-update FFT 在整個 workflow 內常駐與安全釋放；Stage 3 專注於 adaptive posterior search 中逐 particle 啟動 GPU scorer、逐次下載 score 所造成的同步與小批次負擔。本階段已於 2026-09-08 通過 CUDA／CuPy A/B、384-particle 等價性與 3050-particle known-reference 驗收並封存；正式判定見 `validation-results/performance/stage-3/ACCEPTED.md`。

## 實作範圍

本階段不改公開 workflow API、`AlignmentConfig`、pose convention、coarse／fine cell 定義、posterior-mass selection、stable tie ordering、rescue、M-step 或輸出 dtype。CPU 權威路徑仍逐 particle 評分，因此保留既有數學執行順序。

CUDA／CuPy adaptive 路徑改成：

1. CPU controller 以最多 32 particles 為一組建立 coarse cells；
2. 將該組 ragged cells 串成 flat candidate buffer，由既有 VRAM planner 依 `batch_size` 與 `memory_fraction` 切成 GPU candidate chunks；
3. 一次下載每個 chunk 的 NCC scores，再依原 offsets 拆回各 particle；
4. CPU 逐 particle 執行完全相同的 posterior、mass selection 與 fine-cell 建構；
5. fine cells 以相同方式跨 particles 評分，再交回既有 finalize／rescue／M-step 流程。

32 是內部 host-side ragged working-set 上限，不是新的 public tuning parameter。實際 GPU candidate chunk 仍由 `AlignmentConfig.batch_size` 與 Stage 2 的 VRAM planner 決定；OOM 時既有 cache eviction 與 batch 對半縮小規則不變。這個上限避免 K、搜尋範圍或 oversampling 增大時，在 host 同時建立無限制的 candidate arrays。

Whitened scorer 的 reference norms 改為每個 particle weight profile 計算一次、在 coarse／fine 間重用。Uniform Fourier NCC 仍共用單一 weight profile。新增 profiling counters：

- `adaptive_particle_batches`／`adaptive_batched_particles`；
- `adaptive_scoring_batches`／`adaptive_scored_candidates`；
- `adaptive_cross_particle_batches`。

`gpu_memory_plans` 另記錄 `adaptive_particle_batching`，區分 host particle group 與實際 GPU candidate batch。

## 數值與效能驗收

`tools/stage3_validation.py` 會驗證並展開 Stage 2 已封存的 `dev3-delivery`，用相同輸入、config、seed 與 dev4 native binary，比較 frozen dev3 與目前 dev4。Native CUDA kernel 本階段沒有改動，只有 version stamp；若 native source 或檔案集合不同，A/B 會拒絕執行。

每個 case 必須同時符合：

- 完整 E-step arrays、poses、references、class averages、responsibilities、FRC 與 history 維持既有 tolerance；
- coarse／fine candidate 總數完全相同；
- coarse／fine controller scorer calls、H2D calls 與 D2H calls 都低於 dev3；
- D2H score payload bytes 完全相同，證明沒有少算或少下載 candidate scores；
- 至少觀察到一個真正混合多 particles 的 GPU candidate chunk；
- workflow workspace 關閉、cache 峰值不超過 80% budget，標準 workload 不發生 OOM retry。

H2D bytes 允許小幅增加，因為 flat batch 需上傳每個 candidate 的 particle index；這個增量會明確記錄。Stage 5 才會把 particle index 直接整合進 native CUDA 熱點，現階段的目標是降低大量小呼叫與 host/device 同步。

CUDA median 若比 dev3 慢超過 5%，標記 `repeat_required`；CuPy 是 portable fallback，效能只記錄觀察值，但數值與 batching 契約仍是硬性門檻。

## 本機驗證

- 新增 controller contract test：4 particles 的 coarse／fine scorer calls 從 8 次降為 2 次，所有輸出欄位逐元素相同。
- 既有 adaptive posterior、calculation reuse 與 RELION pose benchmark 測試通過。
- macOS 本機沒有 CUDA／CuPy，GPU kernel、傳輸 counters 與 throughput 必須在 Linux server 驗收。

## dev4 Server preflight 與修正

dev4 CUDA quick 的 18 cases 中 13 個通過；所有 5 個 adaptive cases 在 scoring 前建立 `adaptive_particle_batching` memory record 時，因漏定義區域變數 `particle_count` 而停止。Smoke／representative current runs 因同一原因未產生可比較的數值或 timing；這不是 candidate scoring 或 posterior regression。

dev5 在 GPU adaptive entry 直接由 `len(particles.spatial)` 定義 particle count，並新增不需 CUDA 裝置的 entry regression test，驗證使用預設 `batch_size=None` 時也能建立 batching metadata 並抵達共享 adaptive controller。dev4 失敗 JSON 與 delivery snapshot保留，不覆寫。

## Server 部署與第一輪驗證

在既有 `align-dev` 環境與專案根目錄執行。需上傳完整 `src`、`packages/alignimg-gpu`、`tools`、`tests`、`docs`、root `pyproject.toml`，並保留 `validation-results/performance/stage-2/dev3-delivery/` 及 Stage 0 fixture。

```bash
python -m pip install --force-reinstall --no-deps -e .

CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" \
python -m pip install -v --force-reinstall --no-build-isolation --no-deps \
  packages/alignimg-gpu/dist/alignimg_gpu-2.1.0.dev5.tar.gz
```

先避免其他程序共用 GPU，依序執行：

```bash
python tools/server_validation.py \
  --suite quick --backend cuda --batch-size 512 \
  --output validation-results/performance/stage-3/dev5-cuda-quick.json

python tools/stage3_validation.py \
  --suite smoke --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-3/dev5-cuda-smoke-ab.json

python tools/stage3_validation.py \
  --suite representative --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-3/dev5-cuda-representative-ab.json

python tools/stage3_validation.py \
  --suite smoke --backend cupy --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-3/dev5-cupy-smoke-ab.json

python tools/stage3_validation.py \
  --suite representative --backend cupy --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-3/dev5-cupy-representative-ab.json
```

其中 smoke 只執行 `adaptive`、`adaptive_whitened`；representative 只執行 384-particle `pose_fixed_mra` 與 256-particle local `local_refine`。A/B 每個 case 會另產生 baseline／current JSON 及 inputs／result NPZ；帶回分析時至少保留三份 JSON，NPZ 由總報告的 SHA-256 驗證。

第一輪 A/B 通過後，再執行完整 3050-particle K=1 known-reference milestone；不在基本 batching 契約未確認前先消耗長時間：

```bash
python tools/align_known_reference.py \
  --particles data/local/test_align.mrcs \
  --reference data/local/mu_aligned_mean.mrc \
  --output-directory validation-results/performance/stage-3/dev5-local-known-reference \
  --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --global-iterations 1 --refine-iterations 3 \
  --angle-samples 256 --proposal-angles 8 --translation-range 6 \
  --apply-final-pose-to-raw --profile-execution
```

Stage 3 只有在 adaptive quick、CUDA／CuPy dev3→dev4 A/B、384-particle fixed-MRA、K=1 local refinement 與完整 3050-particle run 都完成檢視後才封存。此階段不加入新限制、auto-stop、分類規則或 Stage 4 的 full／half-set 共用累加。

## 最終 Server 驗收

正式版本為 `2.1.0.dev5`。CUDA quick 18／18 通過；CUDA smoke 的 adaptive／
adaptive-whitened 相對 dev3 改善 20.47%／18.05%，clean-GPU representative
重跑的 local refine／fixed-MRA 改善 18.34%／17.01%。CuPy 四個對應 case
亦無 regression。所有 A/B scientific arrays 最大誤差為 `0`，candidate 數量與
D2H payload 不變，H2D／D2H calls 均下降。

384-particle benchmark 中，independent K=1 與 joint fixed-MRA 的 angle、shift、
responsibility 最大差異均為 `0`，reference correlation 為 `1`；96.88% particles
在 11.25° 內、95.57% 在 2 px 內。3050-particle profiled K=1 workflow 的所有
JSON／NPZ／MRC 值 finite，raw class average 與 NPZ 完全一致，batch 512 無 OOM；
488 MB workspace cache 完整釋放。完整數值、checksums 與 Stage 4 邊界記錄於
正式 acceptance record。
