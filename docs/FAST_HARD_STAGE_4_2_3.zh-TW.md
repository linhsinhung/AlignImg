# AlignImg 2.3 Fast-Hard：Stage 4 preset、diagnostics 與 precision ladder

狀態：**READY FOR SERVER VALIDATION**（2026-09-23）。Stage 3 已由
`validation-results/fast-hard/stage-3/STAGE3_ACCEPTED.md` 接受；本階段公開已通過的
fast path、補齊review evidence，並建立Stage 5 end-to-end runner。尚未宣稱
fast→precise科學與整體效能gate通過。

## 凍結假設

1. `fast_hard`只由caller或GUI明確選擇；既有`global_balanced`、
   `global_accurate`、`reference_free`與`refine` preset及workflow default不變。
2. Fast preset使用已接受的`polar_hard`、polar ring CCF、Fourier M-step、top-1
   hard posterior，且預設`halfset_diagnostics=False`。
3. Precise stage仍使用Fourier `quadratic_refine`、fixed one-hot class priors及
   `halfset_diagnostics=True`。
4. Diagnostics只提供複查證據，不依margin、occupancy、reassignment、pose delta或
   reference change自動切模式、增加iteration或重播。
5. Balanced-only、fast-only與fast→precise比較使用相同總iteration數。

## 公開preset與GUI

`AlignmentConfig.preset("fast_hard")`固定：

```text
search_strategy = polar_hard
candidate_scoring = polar
score_model = polar_ring_ccf
reference_update = fourier
top_l = 1
proposal_angles_per_reference = None
temperature_start/end = 0.08 / 0.05
halfset_diagnostics = False
```

其餘欄位沿用`AlignmentConfig`既有預設。GUI的global search selector新增明確的
**Fast hard**；選取時鎖定上述相依欄位，離開時恢復該workflow原有preset。GUI初始值
仍為proposal。若同時啟用refine，refine config明確回到Fourier/quadratic並開啟
half-set diagnostics，不會把polar scorer誤帶入local refine。

## Diagnostics與timing

每輪共同記錄effective search/candidate/score model、每particle active reference數及：

- candidate-inference、reference-update、centering與FRC wall time；
- effective component weights、hard occupancy與reseeded components；
- 完整angular bin數、每particle translation-center數；
- best/second objective margin；
- quadratic angle accepted/fallback與boundary-hit count；
- 前後iteration hard reassignment fraction、K×K transition matrix；
- angle/shift pose delta與mirror-change fraction；
- reference relative change。

Final raw-average time另記於result metadata及最後一輪diagnostic。GPU actual batch、
workspace、VRAM plan、explicit transfer counters與native attribution沿用Stage 3既有
metadata/profile，不另建第二套planner。Half-set開關的full reference update不變契約由
既有hard-inference測試持續保護。

## Precision ladder runner

新增`tools/fast_to_precise_validation.py`，只讀Stage 0 frozen inputs，不重新抽樣。
每個case比較：

1. balanced soft跑`global_iterations + precise_iterations`；
2. fast-only跑相同總iterations；
3. fast global跑`global_iterations`，再用其hard assignments建立one-hot priors，跑
   `precise_iterations`的quadratic refine。

三者共享initial references、images、mask/frequency defaults、seed、backend與memory
policy。Runner對每個cell執行warm-up一次及至少三次正式量測，保存final NPZ、hash、
quality、occupancy、row-sum、stage timings、resource metadata與same-backend determinism。
`representative`預設選擇synthetic K3 noiseless/SNR 0.2、homogeneous fixed K3與
mra1000 open K10。這只是固定Stage 5介面，不在Stage 4 smoke後提前接受science gate。

## 本機驗證

- 完整pytest：`306 passed, 22 skipped`。
- Stage 4 preset/GUI/diagnostic/runner定向測試：`28 passed`。
- 真實artifact smoke：CPU synthetic K1執行三種schedule，各為相同2 iterations，
  measured repeats exact deterministic，fixed assignment保持，產出1份JSON與3份NPZ。
- Ruff check、Ruff format check、compileall與`git diff --check`通過。
- 本機無NVIDIA環境；server只需確認Stage 4 overlay與既有Stage 3 native module共同工作，
  不重跑Stage 3 384/1000 throughput gate。

## Server gate

- source manifest與handoff README完全一致；archive沒有PAX xattr、AppleDouble、cache、
  build或dist member。
- 完整pytest預期`316 passed, 12 skipped`。
- core/GPU/native仍為暫存的2.2.0 identity；`fast_hard` preset可normalize，既有default
  仍為proposal/quadratic。
- CUDA synthetic K3 precision-ladder smoke三個schedule皆完成、總iterations相同、
  repeats deterministic、fast responsibilities one-hot、fixed precise assignments不變。
- CUDA不得fallback；polar sampler/peak均為`native_cuda`，memory plans遵守0.8上限。

上述server JSON帶回並審查後，Stage 4才接受並進入Stage 5 representative science與
schedule throughput驗收。
