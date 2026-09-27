# AlignImg 2.3 Fast-Hard：Stage 2 CPU-authoritative `polar_hard`

狀態：**ACCEPTED**（2026-09-18）；本機與 clean server CPU gate 均通過。GPU
package 未修改，CuPy/native CUDA `polar_hard` 明確延至 Stage 3。

Stage 1 已依 `validation-results/fast-hard/stage-1/REJECTED.md` 正式拒絕。Hard-4 的
homogeneous angle median、reference correlation 與 candidate-inference speedup 未通過
frozen gate；唯一允許的 hard-2/hard-4 單因子實驗也未修復品質。

## 範圍與成功條件

- 新增 CPU-only `search_strategy="polar_hard"`；不新增 workflow function 或公開
  `fast_hard` preset。
- 合法組合固定為 `candidate_scoring="polar"`、
  `score_model="polar_ring_ccf"`、`top_l=1`。
- 第一輪 translation center 為 `(0, 0)`；後續輪次以先前 hard top-1 center 累積局部
  grid，但每輪 angle 仍完整搜尋 360 度。
- Responsibilities 必須保留完整 `(N, K)` one-hot array；assignments 等於 argmax。
- Fourier/spatial reference updater、centering、half-set、empty-component、final raw
  average 均沿用既有 engine。
- Stage 2 以 CPU 數學與品質為 gate；速度與 GPU parity 留到 Stage 3。

## 實作

### `src/alignimg/_polar_hard.py`

- 以 AlignImg integer origin 為唯一中心，spatial polar rings 使用 OpenCV bilinear
  interpolation。
- Particle/reference rings 使用相同 angular-mean centering、sqrt-radius weights 與
  L2 normalization。
- 每個 reference 每輪只建立一次 polar FFT；每個 particle/translation center 建立一次
  particle polar FFT，再對 positive-prior references 做完整 angular IFFT correlation。
- Safe translation centers 固定 y-major ordering；超出 polar support/image boundary 的
  center 不建立候選。
- Deterministic flat candidate ordering 為
  `reference -> mirror -> shift_y -> shift_x -> angle`。
- 離散全域 angle peak 使用 periodic 左/中/右三點 quadratic fit；non-finite、flat、
  convex 或 `|delta| > 0.5 bin` 時保留離散 peak。
- Class objective 為 `Cpolar / temperature + log(prior)`；zero prior reference 不建立
  候選，最高 objective 唯一 posterior=1。
- Polar sampling center 會轉換為既有 mirror/CCW rotate/post-rotation shift pose convention。

### Config 與 engine

- `AlignmentConfig.normalized()` 加入 `polar_hard`、`polar` 與 `polar_ring_ccf`。
- `polar_hard` 對 proposal count、adaptive cells、非預設 oversampling 與 rescue 設定
  明確報錯，不靜默忽略。
- `reference_update` 仍允許 `spatial` 與 `fourier`；mirror 預設保持 `False`。
- Metadata engine 為 `alignimg-polar-hard-cpu`，search scope 固定記錄
  `global angle exploration; accumulated local translation centers`。
- Stage 3 前任何帶 GPU candidate-inference 的 `polar_hard` 呼叫都明確拋出
  `NotImplementedError`，不得 fallback 到舊 proposal solver。

## CPU-authoritative tests

`tests/test_polar_hard.py` 覆蓋：

- ring identity、90 度與任意角度 peak；
- periodic angular wrap；
- translation sign、x/y ordering、integer origin 與 boundary clipping；
- later-iteration translation-center accumulation且 angle 仍完整 global；
- mirror convention；
- quadratic accepted、flat/convex、non-finite、out-of-range fallback；
- zero prior 排除、uniform tie ordering、corrective prior；
- hard top-1 one-hot contract；
- fixed K=3 與獨立 K=1 等價；
- spatial/Fourier M-step contract；
- empty-component 行為只記錄；
- Stage 3 前 GPU path 明確拒絕。

## 本機 gate 結果

- 完整測試：296 passed、22 skipped。
- Stage 1/2 runner 與 polar tests：21 passed（新增 empty-component test 後為 22）。
- Ruff 與 Black 通過。
- Synthetic 所有 cases finite、one-hot、exact deterministic；profile on/off parity exact。
- Noiseless K=3：
  - angle median/p95 `0.3828 / 1.0174` degrees；
  - 每顆 particle 的 gauge-adjusted angle 最大誤差 `1.2007` degrees，低於一個
    `360/256 = 1.40625` degree angular bin；
  - shift median/p95 `0.3706 / 0.8265` pixels；
  - assignments 與 balanced baseline 完全相同。
- K=1 mirror off/on 的 angle median 約 `0.018–0.021` degrees、mirror accuracy 1.0。
- Homogeneous fixed K=3：
  - assignments/occupancy `[128, 128, 128]` 不變；
  - hard-4 -> polar_hard angle median `4.21875 -> 2.16852`；
  - hard-4 -> polar_hard angle p95 `138.79688 -> 74.20960`；
  - hard-4 reference correlations
    `[0.78268, 0.79910, 0.75583] -> [0.84067, 0.81616, 0.82510]`；
  - 相對 balanced，polar_hard 的 angle median/p95 與 shift median較佳；shift p95
    `14.4997` 高於 balanced `12.8503`，列為後續 precision ladder 風險；
  - class 1 reference correlation 相對 balanced 仍低約 `0.0246`，Stage 3/4 不得忽略。

SNR 0.2 依 frozen plan 只作 observation。SNR 0.5 hard path 仍有 assignment/reference
品質風險；Stage 2 的 gate 是 noiseless correctness 與改善 config-only failure，不宣告
最終 low-SNR biological acceptance。

## Server 執行

在 Stage 2 handoff 解開並核對 delivery README 的 archive/source SHA 後執行：

```bash
python -m pytest -q

python tools/fast_hard_validation.py \
  --suite synthetic --backend cpu \
  --batch-size 16 --memory-fraction 0.8 \
  --deterministic-repeats 2 --profile-execution --compare-polar-hard \
  --inputs validation-results/fast-hard/stage-0/frozen-inputs/synthetic.inputs.npz \
  --output validation-results/fast-hard/stage-2/dev0-server-cpu-synthetic.json

python tools/fast_hard_validation.py \
  --suite homogeneous --backend cpu --only fixed_k3 \
  --batch-size 16 --memory-fraction 0.8 \
  --deterministic-repeats 2 --profile-execution --compare-polar-hard \
  --inputs validation-results/fast-hard/stage-0/frozen-inputs/homogeneous.inputs.npz \
  --output validation-results/fast-hard/stage-2/dev0-server-cpu-homogeneous-fixed-k3.json
```

只帶回兩份 JSON；大型 result NPZ 留在 server。Stage 2 CPU acceptance 通過後才開始
Stage 3 CuPy/native CUDA 實作。

## Server acceptance 結果

- Source manifest：
  `1de647593a117906e2915fe96e6e607f976d2c50de91d216ba608b8e6e5ad8b2`
  （140 files）。
- 完整測試：306 passed、12 skipped。
- Synthetic 與 homogeneous reports 均 finite、hard top-1、exact deterministic，且
  profile on/off parity exact。
- Noiseless K=3 angle median/p95：`0.38341 / 1.01890` degrees；低於一個
  `1.40625` degree angular bin。
- Homogeneous fixed K=3 angle median/p95：`2.08897 / 74.16567` degrees；shift
  median/p95：`1.47044 / 14.56444` pixels；occupancy 保持 `[128, 128, 128]`。
- Homogeneous reference correlations：`[0.84073, 0.81781, 0.82532]`；Stage 1
  hard-4 的 failed quality indicators 全部改善。
- 正式決議、report SHA-256、限制與 Stage 3 frozen gates 記錄於
  `validation-results/fast-hard/stage-2/ACCEPTED.md`。
