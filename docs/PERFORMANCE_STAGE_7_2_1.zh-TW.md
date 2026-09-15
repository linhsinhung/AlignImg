# AlignImg 2.1 效能優化：Stage 7 整體驗收與 release freeze

紀錄日期：2026-09-11（Asia/Taipei）  
Release candidate：`alignimg 2.1.0rc1`、`alignimg-gpu 2.1.0rc1`  
正式版本：`alignimg 2.1.0`、`alignimg-gpu 2.1.0`  
狀態：RC1 server gate 已接受；正式版本等待最後 package/native identity check

## 範圍與成功條件

Stage 7 不再修改 alignment 數學或 GPU 計算引擎。Release candidate 的 compute
source 必須逐檔等同 Stage 5 已接受的 dev10 snapshot；dev11 mixed accumulation 保留為
負實驗證據，不進入公開設定或 release。

Release freeze 需要：

1. core、Python GPU package 與 native CUDA extension 版本一致；
2. 安裝的 Python source 與 server checkout 一致；
3. dev10 compute files 的 SHA-256 全部相同；
4. CUDA／CuPy quick 全數通過，CUDA 不得 fallback；
5. representative performance suite 完成，batch 512、VRAM fraction 0.8；
6. 5000-particle、K=10、20 iterations、seeds 0／1／2 全數產生 finite artifacts，
   responsibilities 正規化且所有 GPU memory plans 保留 batch 512；
7. 與 frozen 1.10 baseline 的 runtime、partition 與 reference diagnostics 完成檢討。

ARI、NMI、RELION label agreement、FRC 及跨版本 assignment ARI 是科學觀察，不是
biological classification 的自動 pass/fail threshold。Runner 通過後仍將
`release_decision` 設為 `pending_review`；帶回結果人工檢討後，才把版本改為正式
`2.1.0`。

## Release candidate 安裝

在 `align-dev` 環境和 repository root 執行：

```bash
python -m pip install --force-reinstall --no-deps -e .

CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" \
python -m pip install -v \
  --force-reinstall \
  --no-build-isolation \
  --no-deps \
  packages/alignimg-gpu/dist/alignimg_gpu-2.1.0rc1.tar.gz
```

先執行不運算 5000 particles 的 release preflight：

```bash
mkdir -p validation-results/performance/stage-7

python tools/stage7_validation.py \
  --check-only \
  --backend cuda \
  --output validation-results/performance/stage-7/rc1-cuda-preflight.json
```

Preflight 會直接拒絕 stale editable install、stale GPU package、沒有重新編譯的 native
extension、改變過的 dev10 compute source、非 production defaults，或仍存在的
`gpu_accumulation_precision` 公開選項。

## Quick 與 representative gates

GPU 沒有其他 compute workload 時依序執行：

```bash
python tools/server_validation.py \
  --suite quick --backend cuda --batch-size 512 \
  --output validation-results/performance/stage-7/rc1-cuda-quick.json

python tools/server_validation.py \
  --suite quick --backend cupy --batch-size 512 \
  --output validation-results/performance/stage-7/rc1-cupy-quick.json

python tools/performance_validation.py \
  --suite representative --backend cuda \
  --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-7/rc1-cuda-representative.json
```

## 正式 5000-particle gate

前三項通過後執行：

```bash
python tools/stage7_validation.py \
  --backend cuda \
  --batch-size 512 \
  --memory-fraction 0.8 \
  --output validation-results/performance/stage-7/rc1-final.json
```

此 runner 固定 production path，不提供調參入口：

- `candidate_scoring="fourier"`
- `score_model="fourier_ncc"`
- `reference_update="fourier"`
- 5000 CTF-corrected 70S particles
- K=10、20 iterations、10 rounds anneal + 10 rounds hold
- seeds 0、1、2

它會依序產生：

- `rc1-final.json`：release master report；
- `rc1-final.rf.json`：三 seeds 完整 RF report；
- `rc1-final.relion.json`：RELION partition diagnostics；
- 每個 seed 的 `.result.npz`、`.references.mrcs` 與 GUI `.report.json`。

Master report 也會讀取 frozen 1.10 baseline 及其三組 artifacts，記錄相同 seed 的 runtime
ratio、assignment ARI、occupancy／responsibility 差異與 radial-profile reference matching。
Radial profile 會丟失角向資訊，只能作診斷，不能單獨證明 reference 相同。

第一輪請帶回下列 JSON；大型 NPZ／MRCS 先留在 server，只有數值或 hash 顯示異常時
再取回指定 artifacts：

1. `rc1-cuda-preflight.json`
2. `rc1-cuda-quick.json`
3. `rc1-cupy-quick.json`
4. `rc1-cuda-representative.json`
5. `rc1-final.json`
6. `rc1-final.rf.json`
7. `rc1-final.relion.json`

## RC1 Server 驗收結果

RC1 已在 Linux、RTX 3090、batch 512、VRAM fraction 0.8 完成全部 gate：

- CUDA／CuPy quick 各 20／20；兩邊各 111 個 memory plans 全部維持 batch 512，
  無 OOM、縮批或 backend fallback；
- representative 4／4；local-refine median 為 8.2524 秒，相對已接受 dev10 的
  8.1953 秒只差 +0.70%，在 5% 重測門檻內；
- 5000-particle 三 seeds runtime 為 222.531、219.715、220.544 秒，median 220.544
  秒；frozen 1.10 baseline median 為 429.322 秒，因此 candidate／baseline 為
  0.5137，約 1.95× throughput；
- 每個 seed 的 assignment ARI 對 baseline 都是 1.0；occupancy entropy、mean maximum
  responsibility 與 RELION ARI／NMI／agreement 均完全相同；
- FRC 的最大 absolute difference 約 `3e-7`；所有 component 的 minimum half-set
  effective weight 都大於 138，沒有不可靠 component 或 reseed；
- 每個 seed 的 scoring／update DFT cache 都只上傳一次並命中 19 次；workflow 結束時
  1.31 GB cache 全部釋放回 CuPy pool。

因此 RC1 技術與科學 gate 接受。這些結果認可 dev10 FP64 compute engine；沒有重新納入
dev11 mixed precision。

## 正式 2.1.0 的最後 identity check

RC1 通過後已進行純版本 freeze：core、GPU Python package 與 native extension 從
`2.1.0rc1` 同步改為 `2.1.0`，並重建 wheel／sdist。這一步沒有修改 compute source。

Server 安裝正式 GPU sdist 後，只需重跑 package/native identity 與 CUDA quick；不重跑
已由相同 compute source 完成的 5000-particle gate：

```bash
python -m pip install --force-reinstall --no-deps -e .

CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" \
python -m pip install -v \
  --force-reinstall \
  --no-build-isolation \
  --no-deps \
  packages/alignimg-gpu/dist/alignimg_gpu-2.1.0.tar.gz

python tools/stage7_validation.py \
  --check-only --backend cuda \
  --output validation-results/performance/stage-7/final-cuda-preflight.json

python tools/server_validation.py \
  --suite quick --backend cuda --batch-size 512 \
  --output validation-results/performance/stage-7/final-cuda-quick.json
```
