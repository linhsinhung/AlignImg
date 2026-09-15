# AlignImg 2.1 效能優化：Stage 6 混合精度選配實驗

紀錄日期：2026-09-11（Asia/Taipei）  
開發版本：`alignimg 2.1.0.dev11`、`alignimg-gpu 2.1.0.dev11`  
狀態：Server 驗證完成；數值合格但無效能收益，不採用並已從主線撤回

## 範圍與假設

本階段只改 GPU Fourier-domain M-step 的加權累加精度，不改 alignment
數學、candidate search、score、posterior、class prior、robust weighting、中心定義、
half-set 流程或公開 workflow API。`gpu_accumulation_precision="float64"` 保持預設，
也是科學結果的權威路徑。

選配的 `"mixed"` 路徑在每個 VRAM-bounded candidate batch 中建立 K 組 FP32 real／
imaginary 部分和；該 batch 完成後立刻將它們加至 workflow-scoped FP64 sums。
component weight denominator 從輸入 weight 到 atomic accumulation 都維持 FP64。
最後 reference normalization 與 IFFT 沿用既有路徑。

這是最小、可撤回的設計。沒有加入 FP16、TF32、持久化 FP32 reference、補償求和、
新的公開 accumulator object，或另一套 inference engine。CPU、spatial M-step 與
`apply_final_pose_to_raw=True` 的 final raw-average 都不受此選項影響。

## 實作與可觀測性

- `AlignmentConfig.gpu_accumulation_precision` 接受 `"float64"`／`"mixed"`。
- Native CUDA fused kernel 依模式寫入 FP64 sums 或 FP32 batch partials。
- CuPy fallback 使用相同的 FP32-partial → FP64-merge 數學契約。
- VRAM planner 將 mixed partial buffers 納入 fixed allocation；既有 80% hard limit、
  batch 512 及 OOM 縮批規則不變。
- metadata 分別記錄 requested 與 executed precision。
- profiler 記錄 mixed batches、partial bytes，以及 native mixed kernel calls／candidates。

Quick suite 新增單輪壓力案例，同時涵蓋：

- 約 `1e5` 的 source scale span；
- 大值相消；
- `1e-6` 低權重；
- 單一 batch 與 chunk size 5 的 batch-partition 比較；
- FP64 total-weight 完全不變。

`tools/stage6_validation.py` 會用相同 dev11 source、fixture、seed、backend 與 batch，
先跑 FP64，再跑 mixed。兩邊的輸入除 precision 欄位外必須逐值相同，並保存完整
candidate／posterior 與 final result NPZ 供數值比較。

## 驗收規則

1. 既有測試門檻不放寬；mixed 與 FP64 使用 `performance_validation.compare_arrays`
   的既定 pose／reference／posterior tolerance。
2. FP64 run 不得出現 mixed counter；mixed run 必須實際走 mixed path。CUDA 明確要求
   native mixed kernel，不能 fallback。
3. GPU workspace 必須關閉，不能出現 OOM retry；batch 512 與 `memory_fraction=0.8`
   維持。
4. Smoke timing 只作觀察，不因微型固定成本否決實作。
5. Representative case 若 end-to-end 慢超過 5%，要求 clean-GPU 重測；仍成立則撤回。
6. 小於 5% 不宣稱加速。只有 end-to-end 或 shared M-step 穩定快至少 5%，才進入
   1000／3050-particle 驗證；若代表性案例沒有實質收益，就不發布 mixed 選項。
7. 1000 與 3050 都通過後，mixed 仍只作 opt-in；正式預設保持 FP64。

## Server 安裝

在既有 `align-dev` 環境、repository root 執行：

```bash
python -m pip install --force-reinstall --no-deps -e .

CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" \
python -m pip install -v \
  --force-reinstall \
  --no-build-isolation \
  --no-deps \
  packages/alignimg-gpu/dist/alignimg_gpu-2.1.0.dev11.tar.gz
```

安裝後確認 core、Python GPU package 與 native extension 都是 dev11：

```bash
python - <<'PY'
import alignimg
import alignimg_gpu
from alignimg_gpu.backend import _native_module

native = _native_module()
print("AlignImg:", alignimg.__version__, alignimg.__file__)
print("AlignImg GPU:", alignimg_gpu.__version__, alignimg_gpu.__file__)
print("Native CUDA:", None if native is None else native.__version__)
print("Default precision:", alignimg.AlignmentConfig().gpu_accumulation_precision)
PY
```

## 第一輪 Server 驗證

請先確認 GPU 沒有其他 compute workload，再依序執行：

```bash
mkdir -p validation-results/performance/stage-6

python tools/server_validation.py \
  --suite quick --backend cuda --batch-size 512 \
  --output validation-results/performance/stage-6/dev11-cuda-quick.json

python tools/server_validation.py \
  --suite quick --backend cupy --batch-size 512 \
  --output validation-results/performance/stage-6/dev11-cupy-quick.json

python tools/stage6_validation.py \
  --suite smoke --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-6/dev11-cuda-mixed-smoke-ab.json

python tools/stage6_validation.py \
  --suite representative --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-6/dev11-cuda-mixed-representative-ab.json

python tools/stage6_validation.py \
  --suite smoke --backend cupy --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-6/dev11-cupy-mixed-smoke-ab.json
```

每支 Stage 6 A/B runner 另外產生 `.float64.json` 與 `.mixed.json`。第一輪請帶回兩份
quick JSON、三份主 A/B JSON，以及對應的六份 precision JSON。大型 result／inputs NPZ
先留在 server；只有報告發現 hash 或數值異常時再取回指定 artifact。

第一輪結果通過後，才設計 1000-particle RF → feedback 與 3050-particle K=1
known-reference mixed／FP64 A/B。本階段不直接執行 5000 particles 或三 seeds；那是
Stage 7 release freeze 的工作。

## Server 驗收結果：不採用

dev11 在 Linux RTX 3090 上完成 CUDA／CuPy quick（各 21／21）、CUDA smoke／
representative A/B 與 CuPy smoke A/B。Source SHA-256 為
`8a66964ecc25bcefbe0e4d5aa654143bf613e2546c646e4ba5ca9d18268011d7`，兩條 CUDA
A/B 使用相同 native binary
`c2ab04f26350e9bf5ddd611d23db5ad530a7dada67be52feec0d00aff69606d0`。
所有 workspace 均正常關閉，沒有 OOM retry 或 backend fallback。

數值壓力案例涵蓋 `1e5` source scale span、訊號抵消、`1e-6` weight 與兩種 batch
partition。CUDA／CuPy 的 FP64 weight error 都是 0；maximum average absolute error
分別為 `8.89e-5`／`1.22e-4`。完整 workflow 的最大 absolute error 為：adaptive
`2.56e-7`、local refine `2.95e-6`、pose-fixed MRA `6.66e-7`，均通過既有 tolerance。

效能結果如下：

| 案例 | mixed / FP64 end-to-end | mixed / FP64 shared M-step | 判定 |
|---|---:|---:|---|
| CUDA adaptive smoke | 0.9841 | 1.1226 | 小案例整體差異不宣稱收益；M-step 較慢 |
| CUDA local refine | 1.0027 | 1.5572 | 整體無差；M-step 慢 55.7% |
| CUDA pose-fixed MRA | 0.9962 | 1.2947 | 整體無差；M-step 慢 29.5% |
| CuPy adaptive smoke | 1.0139 | 1.0484 | 無收益 |

Representative 三次 end-to-end measurements 分布集中，不符合偶發 GPU contention
特徵。Mixed 路徑在 local refine 的 364 個 batches 額外清零／合併 58.24 MB partials；
pose-fixed MRA 的 935 個 batches 額外處理 735.31 MB。FP32 atomic 的節省不足以抵銷
這些固定工作，且 particle 數增加時該成本會隨 batch 數持續累積。

因此 Stage 6 依預定規則判定 `no_material_benefit`。不再消耗時間執行 1000／3050
particles；`gpu_accumulation_precision`、FP32 partial kernel、routing 與公開文件已從主線
撤回，計算引擎恢復為逐檔比對相同的 Stage 5 dev10 FP64 implementation。Dev11 source、
build 與 JSON 保留為負實驗證據，避免未來重複走同一設計。
