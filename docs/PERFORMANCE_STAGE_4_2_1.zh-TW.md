# AlignImg 2.1.0.dev6：階段 4 Full／half-set 共用 M-step 累加

依據 [分階段計畫](PERFORMANCE_OPTIMIZATION_2_1.zh-TW.md)。Stage 3 已將
adaptive candidate scoring 跨 particles 批次化；Stage 4 只處理同一輪 posterior
在 full reference、half-A 與 half-B M-step 中重複執行 particle transform 的問題。

## 數學與處理順序

令 particle `i`、candidate `g`、component `k` 的未正規化權重為
`w_igk`，固定 half-set membership 為 `h(i)`。新路徑只建立兩份未正規化統計：

```text
S_hk = sum_{i:h(i)=h,g} w_igk T_g(I_i)
W_hk = sum_{i:h(i)=h,g} w_igk
```

每個 candidate 只 transform 一次，並累加到其所屬的 half。Full statistics 在任何
正規化、mask、lowpass 或置中之前由下式取得：

```text
S_full,k = S_A,k + S_B,k
W_full,k = W_A,k + W_B,k
```

這不是合併兩張已完成的 half averages。既有順序保持為：

1. 由 full 未正規化統計建立 full reference；
2. 執行 full mask／lowpass／center correction，取得整數 center shift；
3. 將相同 full center shift 套到兩個 half 的未正規化結果；
4. half-A／half-B 各自執行 mask／lowpass／center correction；
5. 由兩張完成的 half references 計算 legacy FRC 與 stable FRC。

因此 candidate pose 的 full center correction、FRC frame 與舊路徑一致。空 component
仍由既有 full effective-weight 規則 reseed；不強制合併或重分配 half components。

## 實作範圍

- CPU spatial 與 Fourier updater 都有 shared unnormalized accumulation；舊三次 updater
  保留為測試 oracle 與明確傳入 custom updater 時的相容路徑。
- CUDA／CuPy 使用兩組 FP64 real／imag accumulators。Full complex sum 在 GPU 上由
  兩個 half sums 相加，full 與兩個 half 仍各只輸出一張 reference image。
- Workflow metadata 新增 `halfset_update_policy`；正式路徑為
  `shared_unnormalized_accumulation`。
- Profiling counter `shared_mstep_candidate_transforms` 記錄實際只做一次的 M-step
  candidate transforms。
- VRAM planner 對兩個 half accumulators、derived full sum 與 IFFT temporaries採保守
  fixed-memory 預算；`memory_fraction=0.8`、cache eviction、host streaming 與 batch
  對半縮小規則不變。
- `halfset_diagnostics=False` 時仍走原本單一 full updater，不增加記憶體或工作量。

本階段不改 candidate scoring、posterior、class／robust weights、reference 正規化、
center convention、FRC threshold、low-occupancy masking、raw-average、輸出 dtype 或
公開 workflow API。Native CUDA kernels 未修改，只有版本字串更新。

## 本機驗證

- Shared spatial／Fourier updater 對三次 legacy updater oracle，涵蓋非零 center shift、
  mask、lowpass、mirror 與空 component。
- Proposal 與 adaptive ragged posterior workflow 的 references、weights、poses、FRC、
  stable cutoff 與 histories 維持既有 tolerance。
- CPU quick：18／18 passed。
- 本機完整 pytest：210 passed、20 skipped。GPU kernels、傳輸量與正式吞吐仍需在
  Linux RTX 3090 server 驗收。

## Server 部署與 A/B 驗證

在既有 `align-dev` 環境與專案根目錄執行。必須保留
`validation-results/performance/stage-3/dev5-delivery/`，Stage 4 runner 會驗證其
source／archive checksum，並以相同 installed native binary 比較 frozen dev5 與
dev6 Python GPU engine。

```bash
python -m pip install --force-reinstall --no-deps -e .

CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" \
python -m pip install -v --force-reinstall --no-build-isolation --no-deps \
  packages/alignimg-gpu/dist/alignimg_gpu-2.1.0.dev6.tar.gz
```

先避免其他程序共用 GPU，再依序執行：

```bash
python tools/server_validation.py \
  --suite quick --backend cuda --batch-size 512 \
  --output validation-results/performance/stage-4/dev6-cuda-quick.json

python tools/stage4_validation.py \
  --suite smoke --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-4/dev6-cuda-smoke-ab.json

python tools/stage4_validation.py \
  --suite representative --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-4/dev6-cuda-representative-ab.json

python tools/stage4_validation.py \
  --suite smoke --backend cupy --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-4/dev6-cupy-smoke-ab.json

python tools/stage4_validation.py \
  --suite representative --backend cupy --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-4/dev6-cupy-representative-ab.json
```

Smoke 比較 proposal global 與 adaptive refine；representative 比較 384-particle
fixed-MRA 與 256-particle local refine。每個 case 都必須：

- 完整 E-step／M-step arrays、effective weights、poses、references、class averages、
  responsibilities、legacy/stable FRC 與 histories 通過既有 tolerance；
- M-step transform calls、update-cache requests、H2D／D2H calls 與 payload 不增加；
- 使用 shared policy，workspace 關閉，無標準 workload OOM retry；
- CUDA median 慢超過 5% 時 clean-GPU 重跑；CuPy timing 只作 fallback observation。

A/B 通過後執行第二輪 1000-particle RF → feedback workflow：

```bash
python tools/re2dc_70s_final_validation.py \
  --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-4/dev6-re2dc-70s-final-n1000.json
```

四份 final JSON、RF／feedback NPZ 與 MRCS 都需 finite；fixed assignment 必須不變，
FRC reliability 與 class occupancy 只檢查 technical contract，不以 ARI 作 Stage 4
通過門檻。完成檢視後才封存 Stage 4 並進入 Stage 5 native CUDA 熱點整合。
