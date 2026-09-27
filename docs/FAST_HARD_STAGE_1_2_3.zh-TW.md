# AlignImg 2.3 Fast-Hard：Stage 1 config-only hard proposal

狀態：**REJECTED**。Config-only hard proposal 未通過已凍結的 accuracy 與 speed gate；
Stage 2 CPU-authoritative `polar_hard` 已獲准開始。

Stage 0 已依 `validation-results/fast-hard/stage-0/ACCEPTED.md` 正式接受。本階段只
驗證既有 proposal solver 的最小 hard 設定，不修改 core solver、GPU package、版本號
或公開 preset。

## 假設、範圍與成功條件

- Baseline 與 candidate 使用相同 frozen input、三個 iterations、frequency band、mask、
  reference update、seed、backend、batch 與 diagnostics。
- Candidate 相對 `global_balanced` baseline **只**改：
  `proposal_angles_per_reference: 6 -> 4`、`top_l: 8 -> 1`。
- 每個 variant 在同一 process 依序完成一次 warm-up、三次 unprofiled measured 與三次
  profiled measured；candidate-inference gate 使用三次 profiled wall time 的 median。
- `top_l=1` 必須產生真正 one-hot responsibilities，並沿用既有 M-step，不能新增或修改
  engine 分支。
- 所有 Stage 0 provisional gates 已凍結，不得依本階段結果放寬。

## 實作

`tools/fast_hard_validation.py` 新增：

- 私有 `fast_hard_config()`；沒有加入 `AlignmentConfig.preset()`。
- `--compare-fast-hard`，且必須搭配 `--profile-execution`。
- `alignimg.fast-hard-validation.v2` report；每個 case 包含 `baseline`、`fast_hard` 與
  `comparison`。
- 三次 profiled candidate-inference raw times 與 median。
- wall/candidate-inference speedup、assignment、pose delta、reference correlation 與
  search-count ratios。
- Hard contract runtime assertion；任何非 one-hot responsibility 或非 1.0 retained
  posterior 直接使 report 失敗。
- Homogeneous 完整執行時，另比較 fixed K=3 與三個 K=1 candidate poses、assignments
  與 references。

`tests/test_fast_hard.py` 證明：

- 每列 responsibilities 恰有一個 1，assignment 等於 argmax。
- Spatial hard M-step 等於以相同 pose、assignment、inlier weight 直接分組累加。
- Fixed one-hot joint K=3 與三個獨立 K=1 等價。
- Uniform priors 可選擇全部 references。
- 同 config/input 重跑 result hashes 相同。
- Mirror off/on convention 不變。
- `apply_final_pose_to_raw` 與 half-set diagnostics 不改變 inference/full update。

## 不可放寬的 acceptance gate

- 所有結果 finite；responsibility row-sum 誤差 `<= 2e-6`。
- 相同 backend/config/input deterministic repeats 完全相同。
- Homogeneous fixed K=3 的 angle median、shift median、angle p95、shift p95 各不得高於
  同 process balanced baseline 的 `1.25x`。
- 每類 reference correlation 相對 baseline 的降幅不得超過 `0.01`。
- Fixed K=3 assignments 不變；candidate 的 joint/per-class K=1 pose angle/shift差
  `<= 1e-3`，mirror 完全相同。
- CUDA 不得 fallback；requested/actual batch 512，所有 plan 與 workspace 遵守 80%
  VRAM hard limit。
- Homogeneous 384 與 mra1000 candidate-inference median speedup 均須 `>= 1.5x`；
  小於 5% 不宣稱改善。
- Synthetic SNR 0.2 只作 observation，不要求 hard path 勝過 soft baseline。

## 本機結果

- 完整測試：283 passed、22 skipped。
- Stage 1 runner contract tests：14 passed（包含實際 v2 A/B report smoke）。
- CPU synthetic 五案全部 finite、one-hot、exact deterministic；candidate-inference
  speedup 約 `1.94x–3.57x`。
- CPU synthetic SNR 0.5 的 candidate reference correlations 相對 baseline 下降約
  `0.023–0.026`；記錄為 moderate-SNR 警訊，正式決策仍以指定 homogeneous gate
  與 clean CUDA 結果為準。
- CPU homogeneous fixed K=3：
  - candidate-inference speedup `3.82x`；total wall speedup `4.10x`；
  - assignments、hard contract、shift median/p95 通過；
  - angle median `2.8125 -> 4.21875`（`1.50x`，失敗）；
  - angle p95 `106.6641 -> 138.7969`（約 `1.30x`，失敗）；
  - reference correlation 降幅 `0.0417–0.0708`（失敗）。

本機結果表示 config-only candidate 很可能是「速度通過、accuracy 失敗」。依總計畫仍須
完成 clean CUDA A/B 才能正式 reject。若 CUDA 同樣失敗，下一步只允許一次
`proposal_angles_per_reference=2` versus `4` 的單因子實驗；不得同時調整其他參數。

## Clean-GPU 結果與決議

- Server 完整測試：293 passed、12 skipped。
- CUDA/CuPy quick 均為 21/21 passed；native numerical、indexed Fourier 與 fused
  M-step parity 通過。
- CUDA/CuPy synthetic 的 baseline 與 candidate result hashes 逐案相同；所有 repeats、
  profile parity、hard contract、batch 512、80% VRAM plan 與無 fallback/OOM gate 通過。
- CUDA homogeneous fixed K=3：
  - assignments 不變；joint K=3 與三個 K=1 pose/shift 完全相同；
  - angle median `2.8125 -> 4.21875`，為 baseline 的 `1.50x`，失敗；
  - reference correlation 降幅為 `0.07991 / 0.04756 / 0.04835`，全部超過
    `0.01`，失敗；
  - candidate-inference speedup `1.383x`，低於 `1.5x`，失敗；
  - angle p95、shift median/p95 仍在 `1.25x` 內。
- CuPy fixed K=3 與 CUDA 的 baseline/candidate result SHA-256 完全相同；其
  candidate-inference speedup 為 `1.263x`，也未通過。

依 frozen gate，`proposal_angles_per_reference=4, top_l=1` 同時未通過 accuracy 與
speed，Stage 1 正式拒絕。Homogeneous 是 mandatory gate；因此不再執行無法改變本決議
的 mra1000 performance run。

唯一允許的本機 `2` versus `4` 單因子實驗只改 proposal count。Hard-2 相對 hard-4
candidate-inference 快 `1.895x`，但 angle median 由 `4.21875` 惡化為 `5.625`，三類
reference correlation 亦全部下降。這證明縮減 proposal 不能修復品質，因此不再送
server，直接依計畫進入 Stage 2。

## Server 執行紀錄

沿用既有專案路徑、conda/CUDA 環境與 Stage 0 frozen inputs。Stage 1 handoff 沒有刪除
既有 source files，因此可在已驗證乾淨的 Stage 0 tree 上直接解開；`data/` 與既有
`validation-results/` 不需移動。

先驗證 handoff/source SHA（以 delivery README 的值為準），再執行：

```bash
python -m pytest -q

python tools/server_validation.py \
  --suite quick --backend cuda --batch-size 512 \
  --output validation-results/fast-hard/stage-1/dev0-cuda-quick.json

python tools/server_validation.py \
  --suite quick --backend cupy --batch-size 512 \
  --output validation-results/fast-hard/stage-1/dev0-cupy-quick.json

python tools/fast_hard_validation.py \
  --suite synthetic --backend cuda \
  --batch-size 512 --memory-fraction 0.8 \
  --deterministic-repeats 2 --profile-execution --compare-fast-hard \
  --inputs validation-results/fast-hard/stage-0/frozen-inputs/synthetic.inputs.npz \
  --output validation-results/fast-hard/stage-1/dev0-cuda-synthetic.json

python tools/fast_hard_validation.py \
  --suite synthetic --backend cupy \
  --batch-size 512 --memory-fraction 0.8 \
  --deterministic-repeats 2 --profile-execution --compare-fast-hard \
  --inputs validation-results/fast-hard/stage-0/frozen-inputs/synthetic.inputs.npz \
  --output validation-results/fast-hard/stage-1/dev0-cupy-synthetic.json

python tools/fast_hard_validation.py \
  --suite homogeneous --backend cupy --only fixed_k3 \
  --batch-size 512 --memory-fraction 0.8 \
  --deterministic-repeats 2 --profile-execution --compare-fast-hard \
  --inputs validation-results/fast-hard/stage-0/frozen-inputs/homogeneous.inputs.npz \
  --output validation-results/fast-hard/stage-1/dev0-cupy-homogeneous-fixed-k3.json

python tools/fast_hard_validation.py \
  --suite homogeneous --backend cuda \
  --batch-size 512 --memory-fraction 0.8 \
  --deterministic-repeats 2 --profile-execution --compare-fast-hard \
  --inputs validation-results/fast-hard/stage-0/frozen-inputs/homogeneous.inputs.npz \
  --output validation-results/fast-hard/stage-1/dev0-cuda-homogeneous.json

python tools/fast_hard_validation.py \
  --suite mra1000 --backend cuda \
  --batch-size 512 --memory-fraction 0.8 \
  --deterministic-repeats 2 --profile-execution --compare-fast-hard \
  --inputs validation-results/fast-hard/stage-0/frozen-inputs/mra1000.inputs.npz \
  --output validation-results/fast-hard/stage-1/dev0-cuda-mra1000.json
```

實際在 homogeneous mandatory gate 確認失敗後 early-stop，未執行最後的 mra1000
命令。大型 result NPZ 留在 server；本機 acceptance/rejection 以 JSON 中的 result
hashes 稽核。
