# AlignImg 2.1：階段 5 Native CUDA 熱點整合

## dev7：Native indexed particle FFT

依據 [分階段計畫](PERFORMANCE_OPTIMIZATION_2_1.zh-TW.md)。Stage 5 包含數個可獨立
驗證的 native CUDA 熱點；本次 dev7 只實作第一項，不同時加入 angle reuse、fused
weighted reduction 或 raw-average GPU accumulation。

## 本次改動

dev6 的 CUDA Fourier transform 在 workflow cache 已存在時，仍先執行：

```text
cached particle FFT -> particle_fft[indices] device gather -> native transform
```

dev7 改為：

```text
cached particle FFT + device particle indices -> native indexed transform
```

Native kernel 直接以 candidate 的 particle index 定位來源 FFT，不再建立一份與
candidate batch 同尺寸的 gathered complex stack。這條路徑同時用於 proposal／adaptive
candidate scoring、legacy M-step updater 與 shared full／half-set updater。

若 VRAM planner 選擇 host streaming，仍使用原本先在 host 選取 batch 再上傳的路徑；
CuPy fallback 也維持既有 gather 實作，作為 portable conformance oracle。

本次不改 scoring、posterior、pose prior、candidate ordering、FP64 M-step accumulation、
center correction、FRC、reference/class-average estimator、公開 API 或 memory-fraction
規則。Batch size 的估計仍沿用 dev6 保守預算，不先把理論省下的 gather 空間換成更大
batch。

## 本機驗證

- Native indexed routing、pointer contract、host-streaming fallback：3 passed。
- Stage 5 acceptance、server runner、歷史 baseline 與封存相關測試：27 passed。
- 完整 pytest：221 passed、20 skipped。
- CPU quick：19／19 passed；`native_indexed_fourier` 在 CPU 明確記錄為不適用。

本機為 macOS arm64，不能編譯或執行 CUDA kernel；native build 與逐值 conformance 必須
由 Linux RTX 3090 server 驗收。

## 新增量測

- `native_indexed_fourier_transform_calls`
- `native_indexed_fourier_transform_candidates`
- `device_gather_bytes_avoided`

`device_gather_bytes_avoided` 必須等於 indexed candidate 數乘以
`H * W * sizeof(complex64)`。另以 allocation hook 比較 dev6/dev7 的 tracked CuPy live
peak；driver、native session 與 cuFFT 內部配置仍不包含在此數字內。

## Server 前置檢查

Stage 5 A/B 直接使用已驗收的 dev6 performance report 與其 result NPZ。安裝 dev7 前，
先確認以下檔案仍在 server：

```bash
ls validation-results/performance/stage-4/dev6-cuda-smoke-ab.current.*.result.npz
ls validation-results/performance/stage-4/dev6-cuda-representative-ab.current.*.result.npz
```

若檔案存在，再安裝 dev7：

```bash
python -m pip install --force-reinstall --no-deps -e .

CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" \
python -m pip install -v --force-reinstall --no-build-isolation --no-deps \
  packages/alignimg-gpu/dist/alignimg_gpu-2.1.0.dev7.tar.gz
```

## 第一輪驗證

在沒有其他 GPU 工作共用裝置時執行：

```bash
python tools/server_validation.py \
  --suite quick --backend cuda --batch-size 512 \
  --output validation-results/performance/stage-5/dev7-cuda-quick.json

python tools/server_validation.py \
  --suite quick --backend cupy --batch-size 512 \
  --output validation-results/performance/stage-5/dev7-cupy-quick.json

python tools/stage5_validation.py \
  --suite smoke --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-5/dev7-cuda-indexed-smoke-ab.json

python tools/stage5_validation.py \
  --suite representative --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-5/dev7-cuda-indexed-representative-ab.json
```

Quick 中的 `native_indexed_fourier` 會用同一個 native build 比較 gathered 與 indexed
輸出，要求逐值相同。A/B runner 另外要求完整 candidate／posterior／reference／FRC／
raw-average arrays 通過既有 tolerance、H2D/D2H 不增加、tracked allocation peak 不增加、
workspace 正常關閉且標準 workload 無 OOM retry。

CUDA median 比 dev6 慢超過 5% 時先 clean-GPU 重跑，不直接合併下一個 Stage 5 增量。
dev7 通過後才評估 angle/mirror rotation reuse。

## dev7 驗收結論

dev7 已於 Linux RTX 3090 通過 CUDA／CuPy quick 與 dev6-to-dev7 smoke／
representative A/B。所有 captured arrays 相對 dev6 的數值誤差皆為零；代表性案例
時間持平，tracked live peak 分別減少約 39 MiB 與 64 MiB，累積避免約 30 GiB 與
128 GiB 的 candidate-sized device gather traffic。正式紀錄位於
`validation-results/performance/stage-5/dev7-delivery/ACCEPTED.md`。

## dev8：重用相同 angle／mirror rotation

dev8 只加入第二個 Stage 5 增量。Adaptive coarse/fine cells 與 soft M-step candidates
原本已將相同 particle、angle、mirror 的 translations 連續排列，因此使用 contiguous
rotation groups，不建立跨 batch 或跨 iteration 的長期 cache，也不重排 candidates。

Native grouped kernel 對每個 `particle + angle + mirror + Fourier pixel` 執行一次
bilinear rotation interpolation，再於 kernel 內依序套用該 group 的 translation phase，
直接寫回原 candidate 順序。只有平均每個 group 至少三個 candidates 時啟用；重用率
不足時沿用已驗收的 dev7 indexed kernel。

此設計不新增 candidate-sized gather、rotated-image cache 或 output reorder。Group angles
與 offsets 合併成一次小型 native upload；candidate particle index buffer 仍與 scorer 共用。
不改 candidate set、score、posterior、FP64 M-step、reference estimator、FRC、公開 API、
CuPy fallback 或 host-streaming。

新增 profiler counters：

- `native_grouped_fourier_transform_calls`
- `native_grouped_fourier_transform_candidates`
- `native_grouped_fourier_rotation_groups`
- `fourier_rotation_interpolations_avoided`
- `fourier_rotation_reuse_fallback_batches`

## dev8 Server 驗證

先保留 dev7 A/B 產生的 result NPZ：

```bash
ls validation-results/performance/stage-5/dev7-cuda-indexed-smoke-ab.current.*.result.npz
ls validation-results/performance/stage-5/dev7-cuda-indexed-representative-ab.current.*.result.npz
```

安裝 dev8：

```bash
python -m pip install --force-reinstall --no-deps -e .

CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" \
python -m pip install -v --force-reinstall --no-build-isolation --no-deps \
  packages/alignimg-gpu/dist/alignimg_gpu-2.1.0.dev8.tar.gz
```

在 GPU 無其他計算工作的情況執行：

```bash
python tools/server_validation.py \
  --suite quick --backend cuda --batch-size 512 \
  --output validation-results/performance/stage-5/dev8-cuda-quick.json

python tools/server_validation.py \
  --suite quick --backend cupy --batch-size 512 \
  --output validation-results/performance/stage-5/dev8-cupy-quick.json

python tools/stage5_validation.py \
  --suite smoke --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-5/dev8-cuda-rotation-smoke-ab.json

python tools/stage5_validation.py \
  --suite representative --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-5/dev8-cuda-rotation-representative-ab.json
```

`native_rotation_reuse` 會比較 dev7 indexed 與 dev8 grouped output。A/B runner 要求
完整科學輸出維持既有 tolerance、group 數小於 transformed candidate 數、interpolation
省略量計數一致、H2D／D2H 不增加、tracked CuPy live peak 不增加、workspace 關閉且無
OOM retry。任一案例慢於 dev7 超過 5% 時先 clean-GPU 重跑；若仍無效益，保留 dev7
路徑而不繼續堆疊這個增量。

## dev8 驗收結論：不採用

dev8 在 Linux RTX 3090 上通過 CUDA／CuPy quick（各 20／20）、smoke A/B、數值、
VRAM 與傳輸量檢查。Captured scientific arrays 相對 dev7 全部逐值相同，grouped kernel
亦避免了約 74–79% 的 rotation interpolations。

然而代表性案例第一次量測比 dev7 慢 9.37%（local refine）與 7.33%（fixed MRA）；
clean-GPU 五次重測的 median 仍慢 8.66% 與 6.50%。因此這不是可忽略的量測抖動，超過
既定 5% 退步門檻。Grouped kernel 與 routing 已由主執行路徑撤回，Stage 5 繼續以已驗收
的 dev7 native indexed FFT 作為基準。完整負結果保留於
`validation-results/performance/stage-5/dev8-delivery/REJECTED.md`，不以減少理論運算量
取代實際端到端 throughput 驗收。

## dev9：Native fused Fourier M-step accumulation

dev9 從已驗收的 dev7 indexed path 重新開始，不包含 dev8 grouped rotation reuse。本次只整合
Fourier reference update 的兩個既有步驟：

```text
dev7: native indexed transform -> candidate-sized complex64 stack
      -> CuPy FP64 weighted scatter

dev9: native indexed transform + FP64 atomic weighted accumulation
      -> K-sized Fourier sums and weights
```

每個 CUDA thread 仍處理一個 candidate／Fourier pixel；rotation interpolation、translation
phase、candidate 順序、posterior 與權重完全沿用 dev7。Kernel 直接將 weighted real／imaginary
coefficient 累加到 FP64 component accumulators，並累加 FP64 total weights，因此不再建立
candidate-sized transformed output，也不再執行獨立的 CuPy weighted scatter。

這條 fused path 僅在 native CUDA、Fourier M-step 且 particle FFT 已由 workflow workspace
保留於 device 時啟用。CuPy backend 與 VRAM planner 選擇的 host-streaming 模式維持 dev7
實作，作為 portable fallback。Batch 與 80% VRAM hard limit 不變；本增量不把省下的暫存
空間換成更大的自動 batch。

本次不改 scoring、candidate set、posterior、class priors、robust weights、center correction、
half-set 合併規則、FRC、raw average、公開 API 或輸出 estimator。因平行 FP64 atomic addition
的加總順序可與 CuPy reduction 不同，驗收使用既有科學數值 tolerance，不要求 bitwise
identity。

新增 profiler counters：

- `native_fused_fourier_accumulation_calls`
- `native_fused_fourier_accumulation_candidates`
- `fourier_transformed_batch_bytes_avoided`
- `fourier_fp64_scatter_input_bytes_avoided`

Stage 5 validator 同時要求 fused candidate 數等於 shared M-step transform 數、dev7 的獨立
CuPy weighted accumulation stage 在 dev9 消失、H2D／D2H 不增加、tracked CuPy live peak
不增加、workspace 正常關閉且沒有 OOM retry。

## dev9 Server 驗證

安裝同步的 core 與 native CUDA build：

```bash
python -m pip install --force-reinstall --no-deps -e .

CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" \
python -m pip install -v --force-reinstall --no-build-isolation --no-deps \
  packages/alignimg-gpu/dist/alignimg_gpu-2.1.0.dev9.tar.gz
```

確認 server 上仍保留 dev7 A/B 的 current report 與 result NPZ 後，在 GPU 無其他計算工作
時執行：

```bash
python tools/server_validation.py \
  --suite quick --backend cuda --batch-size 512 \
  --output validation-results/performance/stage-5/dev9-cuda-quick.json

python tools/server_validation.py \
  --suite quick --backend cupy --batch-size 512 \
  --output validation-results/performance/stage-5/dev9-cupy-quick.json

python tools/stage5_validation.py \
  --suite smoke --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-5/dev9-cuda-fused-smoke-ab.json

python tools/stage5_validation.py \
  --suite representative --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-5/dev9-cuda-fused-representative-ab.json
```

`native_fused_fourier_accumulation` quick case 直接比較 dev7 transform＋CuPy scatter 與 dev9
fused FP64 sums／weights。A/B 的每個案例先暖機，再以三次 unprofiled median 判定；超過
dev7 5% 時先 clean-GPU 重測，仍退步便撤回此增量。通過 server 驗收前，dev7 仍是 Stage 5
正式 accepted baseline。

## dev9 驗收結論：接受

dev9 已於 Linux RTX 3090 通過 CUDA／CuPy quick（各 20／20）、native fused primitive、
數值、VRAM、傳輸與 clean-GPU representative A/B。Clean-repeat 的 source SHA-256 與
封存 source 完全一致；所有 captured scientific arrays 相對 dev7 的最大誤差為零，native
FP64 sums 的最大差異為 `1.11e-16`。

五次 clean-GPU median 中，local refine 為 dev7 的 0.9897 倍，pose-fixed MRA 為
0.9722 倍，均通過代表性案例不得慢超過 5% 的門檻。兩者的 shared reference update
分別由 0.3562 s 降至 0.1017 s，以及由 1.2329 s 降至 0.3193 s，目標 M-step 約減少
71.5% 與 74.1%。

8-particle global smoke 由 12.68 ms 增至 13.86 ms，因此原始 JSON 仍保留
`repeat_required`；不修改或隱藏此負面結果。該案例的 reference update 本身仍快 13.3%，
而絕對 end-to-end 差異只有約 1.18 ms，故不以此 micro-case 否決已通過正式門檻的代表性
workloads。完整判定見
`validation-results/performance/stage-5/dev9-delivery/ACCEPTED.md`。

dev9 成為下一個 Stage 5 增量的 accepted baseline；本結論尚不包含 GPU final raw-average
accumulation。

## dev10：GPU final raw-average accumulation

dev10 只修改 `AlignmentConfig(apply_final_pose_to_raw=True)` 在 inference 全部完成後的輸出
重建，不改 pose、assignment、responsibility、soft reference 或任何 E／M-step 數學。資料流由：

```text
dev9: in-memory raw batch -> GPU final-pose transform -> N-image D2H
      -> CPU FP64 hard-class weighted accumulation

dev10: in-memory raw batch -> GPU final-pose transform
       -> GPU FP64 hard-class weighted accumulation -> K-image D2H
```

實作刻意先採簡單版本：CUDA backend 沿用 persistent native spatial transform；CuPy backend
沿用既有 runtime-compiled transform，兩者後方都使用 CuPy FP64 device accumulation。沒有新增
另一顆 fused spatial kernel。若 scale benchmark 顯示累加本身才是瓶頸，再另立增量評估融合，
不在尚未證明有效前增加維護成本。

「raw」仍指 workflow 呼叫者傳入、已在 host RAM 的 `images`。本次不會重開 MRCS、不新增
out-of-core reader，也無法消除每個 batch 的 H2D；省下的是 N 張 aligned images 的 D2H 與
CPU accumulation。VRAM planner、`memory_fraction=0.8` hard limit、batch 512 與 OOM 縮批
規則均保留。空 component 繼續保留原 soft reference，不被零影像覆寫。

Profiler 新增：

- `final_raw_gpu_accumulation_batches`
- `final_raw_gpu_accumulation_particles`
- `final_raw_aligned_d2h_bytes_avoided`
- `final_raw_output_d2h_bytes`

`class_average_gpu_accumulation="cupy_fp64_device"`、實際 transform backend、batch、component
數、空 component 與獨立 VRAM plan／OOM events 會寫入 metadata。

### dev10 Server 驗證

先安裝同步的 core 與 native CUDA build：

```bash
python -m pip install --force-reinstall --no-deps -e .

CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" \
python -m pip install -v --force-reinstall --no-build-isolation --no-deps \
  packages/alignimg-gpu/dist/alignimg_gpu-2.1.0.dev10.tar.gz
```

dev10 workflow A/B 預設讀取已接受 dev9 clean-repeat 的 `.current.json` 與對應 result NPZ；
這些 baseline artifacts 必須留在 server 原路徑。GPU 沒有其他工作時執行：

```bash
python tools/server_validation.py \
  --suite quick --backend cuda --batch-size 512 \
  --output validation-results/performance/stage-5/dev10-cuda-quick.json

python tools/server_validation.py \
  --suite quick --backend cupy --batch-size 512 \
  --output validation-results/performance/stage-5/dev10-cupy-quick.json

python tools/stage5_validation.py \
  --suite smoke --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-5/dev10-cuda-raw-average-smoke-ab.json

python tools/stage5_validation.py \
  --suite representative --backend cuda --batch-size 512 --memory-fraction 0.8 \
  --output validation-results/performance/stage-5/dev10-cuda-raw-average-representative-ab.json

python tools/stage5_raw_average_validation.py \
  --backend cuda --counts 1000,3000,5000 --size 128 --components 10 \
  --batch-size 512 --memory-fraction 0.8 --repeats 3 \
  --output validation-results/performance/stage-5/dev10-cuda-raw-average-scale.json
```

Quick 檢查 CUDA／CuPy 數學與完整 public workflow；workflow A/B 要求 captured scientific
arrays 維持既有 tolerance、final raw D2H 由 N 張降為 K 張，且 representative end-to-end
不得比 dev9 慢超過 5%。最後一支 runner 隔離 final raw-average stage，使用相同的 synthetic
128×128 RAM stack 同時重現 dev9 host accumulation 與 dev10 device accumulation；它不計磁碟
I/O 或 alignment inference。1000／3000／5000 中最大案例至少快 5% 才將此增量認列為
`material_benefit`。若最大案例無實質收益便撤回；若慢超過 5%，先在 clean GPU 重跑一次，
確認後再撤回。

## dev10 驗收結論：接受，Stage 5 完成

dev10 已於 Linux RTX 3090 通過 CUDA／CuPy quick（各 20／20）、dev9→dev10 smoke／
representative workflow A/B，以及 1000／3000／5000 particles、128×128、K=10 的隔離
raw-average scale A/B。Server source SHA-256 為
`a18a1bac05a7a610be1bfb4bbb77204c055edb81f3913041d239543aa32812f3`，與封存來源一致；
native CUDA binary SHA-256 為
`8f9cfea92de9c738cf6119b184a9b5e9906d0896c5514cfab869e809395b5abd`。

Workflow captured scientific arrays 的最大誤差為零。Adaptive smoke end-to-end 為 dev9 的
0.9949 倍；其 final raw micro-stage 因固定 GPU 成本由 0.716 ms 增至 1.976 ms，負面結果
保留。Local refine end-to-end 為 dev9 的 1.0083 倍，低於 5% 退步門檻；其 final raw stage
則由 9.935 ms 降至 4.037 ms，約快 59.4%。兩者 tracked CuPy live peak 未增加，workspace
正常關閉，沒有 OOM retry。

隔離 scale 結果：

| particles | dev9 host accumulation | dev10 GPU accumulation | dev10 / dev9 | 加速 |
|---:|---:|---:|---:|---:|
| 1000 | 0.09948 s | 0.01725 s | 0.1734 | 5.77× |
| 3000 | 0.24081 s | 0.04529 s | 0.1881 | 5.32× |
| 5000 | 0.39632 s | 0.07392 s | 0.1865 | 5.36× |

三個 scale outputs 的最大誤差及 relative L2 error 均為零。5000 particles 的 aligned-image
D2H 由 327.68 MB／10 calls 降為 0.655 MB／1 call，總顯式傳輸省下約 326.24 MB；batch
維持 512，沒有 OOM retry。這個結果不包含磁碟 I/O 或 alignment inference，因此只對
final raw-average 路徑宣稱收益。

dev10 通過「最大案例至少快 5%」的預定門檻，成為 Stage 5 最終 accepted baseline。
Stage 5 至此完成；完整正式判定見
`validation-results/performance/stage-5/ACCEPTED.md` 與
`validation-results/performance/stage-5/dev10-delivery/ACCEPTED.md`。
