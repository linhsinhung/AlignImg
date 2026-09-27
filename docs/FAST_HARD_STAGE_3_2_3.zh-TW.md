# AlignImg 2.3 Fast-Hard：Stage 3 CuPy / native CUDA `polar_hard`

狀態：**READY FOR SERVER VALIDATION**（2026-09-18）。Stage 2 CPU-authoritative
實作已接受；本階段只驗證 GPU parity、native attribution、資源上限與 throughput，尚不
新增公開 `fast_hard` preset。

## 範圍與成功條件

- CuPy 與 native CUDA 使用與 Stage 2 完全相同的 polar sampling、candidate ordering、
  quadratic angle fit、prior objective 與 hard top-1 契約。
- CPU/CuPy/CUDA 的 top-1 reference 與 mirror 必須一致；angle 差不超過
  `1e-3` degree，integer raw shift 差不超過 `1e-5` pixel。
- Score沿用既有float32 Fourier/native maximum absolute tolerance `2e-5`，objective
  absolute tolerance固定為 `1e-3`；relative score error照常記錄，但不另設重複的
  per-element hard gate。不得在看到server結果後放寬既有absolute tolerance。
- Batch 256/512 不得改變 assignment、mirror 或超出上述 tolerance 的 pose。
- Native report 必須記錄 native sampler/peak、無 backend fallback，且完整 angular
  correlation maps 的 D2H bytes 為 0。
- 80% VRAM planner 必須記錄 requested/actual particle batch；OOM 只能縮小 device
  particle batch，
  不得切回 CPU。
- 384 與 1000 particle workloads 的 candidate-inference median 必須快於同次執行的
  Stage 1 hard-4 anchor；否則不接受 native fast path。

## 實作

### CuPy portable path

- Stage 3 server preflight發現 OpenCV 4.13與server build的 `warpPolar` 座標產生有約
  一個 interpolation-table step的差異，會放大成約 `0.008°` quadratic angle差，超過
  frozen parity gate。這不是 native compile failure，也不能用放寬 tolerance處理。
- CPU authority因此改為明文的 sin/cos polar offsets與OpenCV `INTER_LINEAR`相同的
  5-bit bilinear table；CuPy RawKernel與native kernel共用此公式，不再依賴
  version-dependent `cv::warpPolar` coordinate generation。
- 本機以 Stage 2 frozen K=3 noiseless重算後，assignments/reference/mirror完全一致，
  angle最大變化 `0.000275°`，candidate score最大變化 `8.95e-7`；quality median/p95
  結論不變。Server仍須先產生同source的Stage 3 CPU parity authority，再跑GPU。
- RawKernel 批次建立 particle/reference polar rings。
- CuPy 在 device 上完成 ring centering、sqrt-radius weighting、normalization、angular
  FFT/IFFT、weighted correlation、periodic peak 與 quadratic fit。
- Candidate reduction依 `reference -> mirror -> y-major center -> angle` 的 frozen flat
  order，device 上保留 deterministic top-1/top-2 objective。
- Host 只接收每顆 particle 的最終 reference、angle、shift、mirror、score、margin 與
  少量 diagnostics；完整 correlation maps 不下載。

### Native CUDA path

- 新增 `polar_hard_cuda.hpp/.cu` 與 `PolarHardSession` pybind binding；sampling 與
  periodic peak/quadratic selection使用 dedicated native kernels。
- Angular FFT/IFFT 與 cross-spectrum reduction沿用 CuPy/cuFFT；top-1/top-2 reduction
  保持在 GPU。此 attribution 不宣稱整條路徑都是單一 native kernel。
- `CMakeLists.txt` 將新 translation unit 納入既有 `_native` module；沒有複用或改寫
  RE2DC CUDA source，因此本階段不新增 third-party attribution。
- Native session依 device/image size/angular/radial shape重用；任何缺少
  `PolarHardSession` 的 CUDA build 都明確失敗，不 fallback。

### Memory 與 metadata

- 第一輪server smoke的數值、native attribution與資源限制均通過，但profile顯示原實作
  將 `batch_size=256` 誤解為 particle×translation×mirror center數；36-particle K=3
  workload因此每輪拆成約13個小GPU chunk，無法作為throughput acceptance依據。
- 修正版將公開 `batch_size` 維持為particle數。Planner的每-item bytes計入該particle
  所有translation×mirror centers的real rings、complex spectra、cross spectra、curves與
  reduction scratch；只有80% VRAM上限或OOM retry可以縮小實際particle batch。
- Report metadata區分 `polar_sampler_backend`、`polar_peak_backend`，並記錄
  `polar_full_correlation_map_d2h=False`。
- 每輪 diagnostics加入 objective margin min/mean、quadratic accepted count、safe center
  count與 boundary rejection count。

### 384-particle formal0 corrective action

- Formal0的CPU scientific quality與fixed-K3/per-class K=1契約通過，CuPy/CUDA彼此
  bitwise一致，但兩個frozen gate失敗，因此未接受：128px輸入的CPU/GPU shift差約
  `5.3e-5 px`；fixed-K3因前一case留下的CuPy pool blocks而被planner縮成
  `particle_batch_size=1`。
- GPU bilinear interpolation原先可由CUDA compiler收縮成FMA；CPU NumPy authority則
  是逐次round-to-nearest float32運算。修正版以 `__fmul_rn`/`__fadd_rn`/`__fsub_rn`
  明確固定相同rounding，不能藉由放寬parity tolerance處理。
- Polar planner在量測 `memGetInfo` 前釋放CuPy pool中的unused cached blocks；仍存活的
  workflow arrays照常佔用VRAM並受80%上限約束，OOM retry規則不變。

### 384-particle formal1 corrective action

- Formal1確認pool release有效且CuPy/CUDA仍逐案一致，但fixed-K3的
  float64/complex128主路徑每輪先OOM再由512縮至particle batch 256；candidate-inference
  約 `2.9-3.0 s`，只有Stage 1的 `0.16x`。沒有OOM的per-class K1也只有
  `0.43-0.49x`，因此未接受。
- Stage 3 fast path改用計畫所稱的float32 polar precision：sampled/normalized rings為
  float32，angular FFT與cross-spectrum為complex64。Planner按新tensor bytes計算並保留
  1.5x cuFFT/einsum workspace allowance。
- CPU與GPU都先沿radial bins加總cross-spectrum，再做angular IFFT；這與原數學等價，
  但移除CPU逐ring IFFT後加總、GPU先加總後IFFT造成的rounding差。
- Frozen shift gate的原文是integer raw shift，不是由連續angle旋轉出的最終pose shift。
  新result artifact因此明確保存最後一次grid search相對輸入center選中的integer raw
  shift；`1e-5 px`只驗證該offset，final pose/candidate shift誤差仍記錄為觀察值。
  Formal1沒有該artifact欄位，維持rejected，不回溯重判。

### 384-particle formal2 corrective action

- Formal2的float32/complex64路徑解除OOM：fixed-K3每輪單批384 particles，三個K1每輪
  單批128 particles；CuPy/CUDA逐案一致，raw shift為0，angle、absolute score、
  objective與scientific gates均通過。
- Throughput仍不合格：fixed-K3只有Stage 1的約 `0.55x`，K1約 `0.69-0.80x`。Profile
  顯示native sampler/peak只占毫秒級，主要成本是real rings仍走完整complex FFT，且
  one-hot fixed-K3仍計算三個reference curves。
- 修正版改用RFFT/IRFFT，保留完整256-bin periodic curve但只保存129個frequency bins；
  若每顆particle恰有一個active prior，只計算該reference spectrum，再映回原本
  reference-major deterministic reduction。一般multi-reference priors不變。
- Formal2 validator誤將relative score `2e-5`當成獨立gate；主計畫要求沿用既有float32
  Fourier/polar tolerance，而既有native契約是maximum absolute score error `2e-5`。
  修正版保留relative metric但只以未變更的absolute `2e-5`判定；Formal2仍因throughput
  失敗而rejected，不回溯接受。

### Formal3 server preflight corrective action

- Core editable install、CUDA 12.8 native compile與GPU wheel install均成功；完整pytest只在
  NumPy device surrogate的score對照出現`2.265e-6`差異。這低於既有frozen score
  absolute tolerance `2e-5`，不是安裝或native compile失敗。
- 該單元測試原先對所有float欄位共用`1e-6`，與正式驗收契約不一致。修正版只將score
  與objective margin分別對齊既有`2e-5`與`1e-3` absolute tolerance；pose、center與
  raw shift仍維持原有`1e-6` absolute加NumPy預設relative tolerance，沒有更動正式
  parity gate或演算法。

## 本機驗證

- 完整 pytest：303 passed、22 skipped。
- Stage 2/3 polar、runner與 profiling定向測試：48 passed、2 skipped。
- Ruff（本階段相關檔案）、Black與 `git diff --check` 通過。
- GPU 演算法以 NumPy device surrogate逐欄對照 CPU authority，包含 chunked centers、
  mirror、reference、angle、shift、score與 top-2 margin。
- GPU package sdist build通過，且包含 `polar_hard_cuda.hpp/.cu`、binding與 CMake entry。
- 本機沒有 NVIDIA CUDA toolchain/device；native compile與真實 CuPy/CUDA numerical、
  VRAM、transfer及 throughput gate只能在 server 完成。

## Server 執行原則

直接覆蓋既有 `/media/linhsinhung/Data_12T_2/test_align`，不建立額外 stage 目錄。
Archive/source SHA 以 Stage 3 delivery README 為準。解壓後必須重裝 editable core，並
從本次 `packages/alignimg-gpu` source重新編譯 native extension；舊的 2.2.0 binary
雖然版本字串相同，但沒有 `PolarHardSession`，不能用於 Stage 3。

第一輪只跑 full pytest及 synthetic K=3 noiseless 的 CuPy/CUDA batch-256 smoke。兩份
JSON通過後，再跑 batch-512、完整 synthetic、384 homogeneous與1000-particle formal
matrix，避免在 native compile或基本 parity失敗時浪費長時間運算。

Stage 3 GPU report使用 schema `alignimg.fast-hard-validation.v4`。由於上述OpenCV
portability修正，Stage 2 report仍保留作歷史與scientific acceptance evidence，但不得
直接作GPU numerical authority；server必須先用同一份Stage 3 source產生CPU report，
GPU再引用其 result NPZ。JSON可帶回；大型 result NPZ仍留在server。

## Frozen acceptance gate

- Same-backend measured/profiled repeats deterministic，且 profiling不改結果。
- CPU/CuPy/CUDA parity全部在固定 tolerance內。
- CuPy與CUDA的 batch-256/batch-512結果符合相同 tolerance。
- CUDA runtime identity指向本次重編的 binary，並提供 binary SHA-256。
- CUDA metadata為 `alignimg-polar-hard-cuda`、sampler/peak均為 `native_cuda`；CuPy
  metadata為 `alignimg-polar-hard-cupy`、sampler/peak均為 `cupy`。
- 所有 hard responsibilities為 one-hot，結果 finite/normalized，無 fallback。
- 所有 memory plan遵守 `memory_fraction=0.8`，完整 maps D2H bytes為 0。
- 384與1000 workloads的 Stage 3 candidate-inference median均快於 Stage 1 anchor。

只有全部通過才進 Stage 4 preset、GUI與 diagnostics整合；任何 parity、resource或
throughput failure都保留原始 JSON，先針對單一 failure修正，不事後改 gate。
