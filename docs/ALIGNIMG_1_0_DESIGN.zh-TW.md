# AlignImg 1.x 實作與科學契約

這份文件記錄 1.0 實際採用的模型、資料流與限制。較完整的統一數學背景見
`UNIFIED_ALIGNMENT_FRAMEWORK.zh-TW.md`。

## 三個 workflow，一個 inference engine

- `reference_free_align`：沒有 reference 或 pose；使用者必須提供 K。
- `align_to_references`：有一張或多張 reference，沒有 pose prior。
- `refine_alignment`：有 reference 與既有 pose；預設加入 local pose prior 和
  robust inlier weighting。

K=1 且沒有 pose prior 是 single-reference global alignment；K=1 且有 pose
prior 是 robust refinement。K>1 時，輸出的 assignment 表示 alignment
component，而不是最終 biological class。

## 每輪資料流

1. 對輸入做背景平均移除、soft circular mask 與 normalization。
2. 建立 FFT、frequency mask 與 polar magnitude descriptor。
3. polar correlation 為每個 particle/reference 提出少量 rotation candidates。
4. 在 Cartesian Fourier domain 搜尋 translation 並計算 band-limited NCC。
5. 加入 class prior、pose prior 與退火溫度，保留 top-L posterior。
6. 同一 reference 的 candidate posterior 加總成 `(N,K)` responsibilities。
7. top-L candidates 以 soft weights 對 real-space particles 做 M-step average。
8. 記錄 occupancy、reference change、posterior concentration、inlier weight
   與 half-set FRC。

CPU 權威實作會將 shortlisted rotation 以 canonical raster transform 產生後再
FFT，以避免不正確的 Fourier-grid phase/interpolation。這表示 particle 的初始
FFT/polar descriptor會被快取，但第一版 CPU 並非嚴格的「整段只做一次
FFT」。GPU backend 會將這些 shortlisted transforms 和 batched FFT 留在裝置
上。後續若加入經過 conformance test 的 Fourier rotation kernel，可以替換此
內部步驟而不更動公開 API。

## Reference-Free bootstrap

RF 使用 polar angular harmonics 的 magnitude 作 rotation-invariant descriptor，
以固定 seed 的 k-means++ 選點和 medoid 更新得到 K 個 components。每群成員先
對 medoid 做 polar rotation alignment，再平均並以 Gaussian low-pass 建立初始
reference。之後直接進入相同的 multi-reference soft engine。

K 不自動估計。低 occupancy component 會以 posterior entropy 最高的 particle
deterministically reseed，並記錄在 diagnostics。

## Priors 與 classification feedback

`class_priors` 是 `(N,K)` 非負矩陣，每列自動正規化：

- 未提供：uniform prior。
- one-hot：particle 只能留在既有 class，適合 fixed-class refinement。
- soft prior：大部分信任 classification，但允許 corrective MRA。
- 0：明確禁止該 particle/reference 組合。

Pose prior 以 circular angle distance 及 x/y Gaussian shift penalty 計算。global
和 RF 的所有 iterations 都維持 global pose exploration；只有 `refine_alignment`
第一輪使用傳入 pose，後續輪再以上一輪 MAP pose 作 local prior。

`make_class_priors` 將外部 classification feedback 轉成 `(N,K)` prior：hard
assignments 配合 `trust=1.0` 會產生 one-hot fixed-class prior；`trust<1.0`
會混合 uniform prior，允許 corrective MRA。Soft responsibilities 會先逐列正規化，
再依相同 trust 規則混合。這些 priors 直接傳入 `refine_alignment`，不需要另一套
feedback engine。Corrective refinement 建議優先使用 soft responsibilities，保留
外部分類原本的不確定性；若只有 hard labels 仍可使用 assignments，但高 trust 的
one-hot prior 屬於保守 fallback，可能完全不產生 hard reassignment。改派結果仍只是
alignment component，不代表 biological class。

`temperature_anneal_iterations` 可讓 temperature 在前段退火後固定於
`temperature_end`；`None` 保留在全部 iterations 中退火的舊行為。

## Transform convention

`PoseSet` 將輸入 particle 轉至 reference coordinates，順序固定為：

1. 以離散整數 x 原點做 periodic left-right mirror
2. 以 `(H//2, H//2)` 為中心做 CCW rotation
3. shift y/x
4. periodic/wrap boundary

Legacy array 雖可透過 adaptor 轉成 `PoseSet`，但舊引擎仍採用 zero boundary
與不同的數值流程，所以 adaptor 只轉換欄位，不能保證 transformed pixels
相同。1.4 geometric-center `PoseSet` 必須透過
`convert_v1_4_poses_to_integer_center()` 顯式轉換。

## Bias control

- mirror 欄位永遠存在，但 search 預設關閉。
- RF 使用固定 seed 建立 half-set membership。
- 主 reference 仍由全資料更新；half references/FRC 只作診斷，不影響 scoring。
- refine 預設使用 sigmoid robust inlier weights。
- component occupancy 太低時 deterministic reseed。
- 所有 iteration 都記錄 reference history、temperature、FRC 與 effective weight。
- FRC 同時記錄 legacy last-crossing cutoff，以及三點平滑、連續三個 rings
  低於 0.143 的 stable cutoff。

## CPU/GPU 邊界

`alignimg` 提供完整 CPU engine。`alignimg-gpu` 是獨立 distribution：

- CPU：資料驗證、polar descriptor/bootstrap、workflow controller、diagnostics。
- GPU：shortlisted rotation、Cartesian Fourier NCC、canonical transform、soft
  reference accumulation。

GPU 明確要求但 CuPy/CUDA 不可用時會報錯，不會 silent fallback。兩個 backend
共用 candidate、posterior、pose 與 result contract；浮點結果只要求在科學
tolerance 內一致，不要求 bitwise 相同。

## 已知限制

- 1.5 新引擎只接受偶數尺寸的 square images。
- CTF correction 與 denoising 必須在 alignment 外完成。
- 只支援 2-D in-plane pose 與單 NVIDIA GPU。
- half-set 是 bias diagnostic，不是 cross-half refinement。
- `top_l=1` 可明確要求 hard MAP，但預設為 8。
- GPU 數值與大規模效能仍必須在 Linux/NVIDIA 驗證主機上完成 conformance。
