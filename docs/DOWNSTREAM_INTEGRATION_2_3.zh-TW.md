# AlignImg 2.3 下游整合指引

日期：2026-09-26。適用於RE2DC及其他使用AlignImg的影像分析系統。

AlignImg負責RF、single/multi-reference alignment與local refinement。下游repository
負責資料轉換、particle IDs/order、pose composition、外部分群及pipeline orchestration。
原計畫Stage 6的adapter由RE2DC開發端實作與驗證；本文件是AlignImg提供的handoff。

## 輸入、模式與輸出

- Images：NumPy float32 `(N, H, H)`，even-square；reference stack為`(K, H, H)`。
- RF：`reference_free_align(images, n_components=K, config=..., backend=...)`。
- 有reference：`align_to_references(images, references, class_priors=..., ...)`。
- Local refine：`refine_alignment(images, references, initial_poses, ...)`。
- `preset("fast3")`供頻繁探索；`global_balanced`或既有RF預設供checkpoint及最後收斂。
- `AlignmentResult`回傳完整`poses`、assignments、responsibilities、candidates、
  references、class averages、inlier weights、diagnostics及metadata。

Reference indices是zero-based；K1與K>1走同一套API。MRA assignments是alignment
components，外部classification labels需由下游決定。`make_class_priors()`可將labels或
probabilities轉為priors；zero排除reference，one-hot固定class，uniform允許reassignment。

```python
from dataclasses import replace
import alignimg as ai

config = replace(ai.AlignmentConfig.preset("fast3"), batch_size=512)
global_result = ai.align_to_references(
    images, references, config=config, backend="cuda"
)
# 下游檢視class averages並執行自己的分群，再選擇下一輪模式。
priors = ai.make_class_priors(assignments=labels, n_components=len(references))
checkpoint = ai.align_to_references(
    images, references, class_priors=priors,
    config=ai.AlignmentConfig.preset("global_balanced"), backend="cuda",
)
```

此範例的`images`、`references`及`labels`由下游提供。References是可更新的initial
models，不是不可變known target。Fast 3會重建final raw class averages；raw指本次呼叫的
`images`，不會自動讀另一份原始檔案。若對齊使用denoised images，下游需另保存raw
stack，再以`transform_images(raw_images, result.poses)`套回同順序的particles。

## Pose與frame契約

`PoseSet`是完整input-to-reference transform：先periodic left-right mirror，再以
`(H//2, H//2)`旋轉，最後平移`(shift_y_px, shift_x_px)`；boundary使用wrap。Positive
shifts往row/column index增加方向移動。Angles/shifts為float32，可含小數。

- 下游需明確轉換geometric-center、angle sign、axis order及mirror convention。
- 偶數box的integer/geometric center差異及mirror offset不能以一個固定半pixel平移取代。
- Returned poses是完整transform；refine回傳值不可直接加上initial poses。
- 若第二次呼叫輸入已經被第一輪transform，下游須以affine transform合成兩輪pose；
  不可只加angles/shifts。使用原particles及returned complete poses可避免此類composition。
- References與initial poses必須共享frame；reference centering應由單一owner負責。
  RE2DC若保留自己的centerer，可明確設定`center_references=False`，並驗證frame更新。
- `poses_from_legacy_params()`只轉換storage columns；它不保證外部transform convention
  相同。`convert_v1_4_poses_to_integer_center()`只針對已定義的AlignImg 1.4 contract。

## RE2DC開發端應驗證的項目

1. Identity、90度、任意角、fractional shift、mirror及two-pass composition。
2. Particle IDs/order從PrePro到RF/MRA及raw reconstruction保持一致。
3. `radius/search/iterations/seed`到AlignImg config的顯式映射。初版不支援的
   `auto_stop`或asymmetric x/y search需明確拒絕。
4. PrePro兩段RF套回raw時無half-pixel drift；DRMRA可完成outer/inner rounds。
5. 相同inputs比較舊backend與AlignImg的pose、reference及runtime；允許quality取捨
   需由RE2DC自己的驗收條件定義。
6. `backend="cuda"`要求native；`cupy`要求portable GPU；`auto`可自動選擇。
   保存實際backend、package/native versions、batch與memory metadata供診斷。

不要求RE2DC使用固定Fast→precise組合。已測兩輪fixed quadratic不保證恢復Balanced
external accuracy；應依外部分群穩定性選擇Fast 3、Balanced或local refinement。
未來其他下游系統亦可依相同通用contract整合，無須AlignImg內置專用adapter。
