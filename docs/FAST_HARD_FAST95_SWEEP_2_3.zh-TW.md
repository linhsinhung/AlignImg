# AlignImg 2.3 Fast-hard practical 0.95 iteration sweep

狀態：**COMPLETED — NO FAST-95 OPERATING POINT**（2026-09-24）。

## 問題

Stage 5證明fast-only可節省時間，但不符合原本accuracy-first release gate。這個補充實驗
改回答較窄的實用問題：若使用者接受final raw class-average對supplied known reference
的correlation至少`0.95`，最少需要多少次fast-only iterations？

這不是重新接受Stage 5，也不改跑5000或Stage 6。`0.95`是使用者指定的practical
threshold，不等同biological accuracy百分比。

## Frozen design

- Workload：相同3050-particle、100×100、K=1 known-reference frozen inputs。
- Iterations：固定`3、5、8、10、12`，不因中途結果增加其他點。
- 每個點皆使用`fast_hard`、`apply_final_pose_to_raw=True`及相同final raw estimator。
- 每個點warm-up一次、三次正式repeats；不跑profile或precise。
- 第12點config必須與`known3050-accuracy0`的fast-only 12完全相同。
- Primary practical threshold：raw correlation `>=0.950000`。
- Secondary frozen release floor：accepted 2.2 `0.9543693464 - 0.005`，即
  `>=0.9493693464`。
- 報告相鄰點raw-correlation delta、median wall time，以及相對frozen balanced 12的
  wall speedup。Correlation不是線性accuracy百分比，不以correlation ratio宣稱準確率。
- 所有assignments必須為0；正式repeats須exact deterministic，CUDA/resource契約不變。

若沒有任何點達`0.95`，結論必須是本workload未建立Fast-95 operating point；不得四捨
五入fast結果或以soft-reference/internal NCC替代final raw metric。

## Result

Server report為
`validation-results/fast-hard/stage-5/fast95-sweep0-server-cuda-known3050.json`，
SHA-256為
`80895526c5923d9c29070189f3e0a9ba2fc909d24928a72b02701ee029370c15`。Frozen
inputs、accepted baseline與comparison report hashes均符合；五個點的三次repeats均
exact deterministic，native CUDA/resource合約正常。

| Fast iterations | Raw correlation | Median wall | Balanced 12 speedup |
|---:|---:|---:|---:|
| 3 | **0.948872** | **7.514 s** | **2.485x** |
| 5 | 0.948079 | 9.063 s | 2.060x |
| 8 | 0.947500 | 11.246 s | 1.661x |
| 10 | 0.947143 | 12.808 s | 1.458x |
| 12 | 0.946738 | 14.176 s | 1.317x |

所有點的`minimum_iterations_meeting_practical_0_95`與
`minimum_iterations_meeting_accepted_2_2_floor`皆為`null`。最佳點fast 3比0.95低
`0.001128`，比accepted floor低`0.000498`。由3增加到12 iterations時，raw
correlation單調降低`0.002133`，耗時增加為`1.887x`。

因此本workload沒有建立嚴格Fast-95 operating point；也不支持「10次內必然達
0.95」。若只追求本資料的近似0.95與最低時間，3 iterations是此次測得的
Pareto點，但仍必須標示為低於預設門檻的explicit opt-in。完整稽核見
`validation-results/fast-hard/stage-5/FAST95_SWEEP0_REJECTED.md`。

後續產品決定另行接受這個微小差距，將測得的Fast 3點定位為反覆分群／
對齊檢視的opt-in exploration preset；Balanced維持checkpoint與最後收旂用途。
這個limited practical acceptance不會將本實驗的strict threshold結果改寫為pass。
