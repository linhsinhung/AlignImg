# AlignImg 2.3 Fast-Hard：Stage 5 corrective 10+2 validation

狀態：**REJECTED — MRA1000 PERFORMANCE ONLY PASS；3050 K1 ACCURACY FAIL**
（2026-09-24）。

原先凍結的fast 3 + fixed quadratic 2在1000-particle open K10上只有balanced 5的
`0.6805x`，已正式拒絕並停止3050/5000延伸；該結論不因本實驗改寫。本文件依使用者
授權，另行登記preset-default長early stage的補充實驗。

## 問題與固定比較

使用同一份Stage 0 frozen inputs、initial references、backend與memory policy，比較：

1. balanced soft 12 iterations；
2. fast-only 12 iterations；
3. fast-hard 10 iterations，再以fixed assignments執行quadratic refine 2 iterations。

三者總數皆為12 inference iterations，不以減少iteration數宣稱加速。10+2只回答
「反覆early fast stage之後，最後兩輪precise的固定成本能否被攤平」，不是3+2的
重判或調參取勝。

## 預先固定的判定

- Primary performance gate：MRA1000的10+2完整wall median相對balanced 12至少
  `1.05x`；低於此值不得宣稱end-to-end加速。
- 三次measured repeats、一次warm-up；另跑profile以定位成本。
- fixed class assignments在precise stage不得改變；responsibilities須finite且row-sum
  誤差不超過`2e-6`。
- CUDA不得fallback；batch 512、memory fraction 0.8，polar sampler/peak須為
  `native_cuda`，不得將完整correlation maps下載至host。
- Determinism沿用既有規則；若只有FP64 reduction尾差，必須揭露array與最大差值，
  不得標成exact。
- 無論pass/fail，本輪仍不自動解禁3050/5000；先審查本輪證據再決定。

## Precise accuracy增益與SNR observation

MRA1000沒有pose/class ground truth，因此只可報告throughput、determinism、occupancy與
stage間輸出變化，不能稱為正確率。正確率改用frozen synthetic K3的noiseless、SNR
0.5與SNR 0.2量測，並使用完全相同的12-iteration三種schedule。

新版runner在每個stage保留truth-based quality，因此可直接比較fast第10輪與其後
precise第2輪：

- class assignment accuracy；
- median/p95 angle absolute error；
- median/p95 shift magnitude error；
- per-class reference correlation；
- per-class aligned raw-average correlation。

同時保留balanced 12與fast-only 12作對照。這能區分「兩次precise相對fast第10輪的
增益」與「10+2相對12次fast-only的最終差異」。Synthetic SNR sweep是預先聲明的
科學觀察，不新增事後pass gate；特別揭露fast-only是否隨SNR下降，以及precise是否
恢復該損失。

## Server輸出

預定只帶回兩份JSON；result NPZ先留在server：

```text
validation-results/fast-hard/stage-5/corrective0-server-cuda-mra1000-10plus2.json
validation-results/fast-hard/stage-5/corrective0-server-cuda-synthetic-k3-snr-10plus2.json
```

## 3050-particle K1 accuracy investigation

Corrective0審查後，使用者只授權恢復3050-particle K1 known-reference，用來隔離K>1
classification error。5000-particle K10仍停止。本輪不把既有user stack描述為具有
pose ground truth；accuracy使用兩個已凍結的外部標尺：

1. final raw class-average對supplied known reference的correlation；
2. pose、soft reference與raw average相對已接受AlignImg 2.2 quadratic結果的差異。

三個schedule仍為balanced 12、fast-only 12與fast 10 + precise 2。Primary accuracy
gate沿用2.2已接受raw correlation `0.9543693464`與最多`0.005`下降，即10+2不得低於
`0.9493693464`。Pose delta因沒有truth只作difference，不可稱absolute error。

本輪執行warm-up一次、三次正式repeats，不執行profile；wall time只記錄，不作新增
performance gate。所有assignments必須為K1 index 0、repeats須exact deterministic、
responsibilities與GPU resource契約不變。輸入及accepted baseline均以既有SHA-256
固定，不重新產生資料。

## 最終結果與決策

- MRA1000的10+2相對balanced 12為`1.350x`，通過corrective performance gate；
  fast-only為`5.457x`。
- Synthetic assignment accuracy隨SNR由100%降至94.4%及83.3%；fixed precise不改
  assignments。SNR 0.2 median angle惡化，SNR 0.5 reference/p95亦未完全恢復balanced。
- 3050 K1已排除classification：balanced、fast-only、10+2的final raw correlation分別
  為`0.965926`、`0.946738`、`0.944912`。10+2相對accepted 2.2下降`0.009457`，未達
  最低`0.949369`的frozen accuracy gate。
- 3050三個schedule皆exact deterministic、assignments全為0、native/resource正常；
  failure不是classification、fallback或nondeterminism。
- 3050 fast-only仍有`1.332x`wall收益；10+2只有`0.242x`，且accuracy未恢復。

因此Stage 5拒絕，5000-particle K10與Stage 6 RE2DC adapter不執行。`fast_hard`只能定位
為explicit opt-in的低成本early global模式；不得宣稱與balanced等精度，亦不得宣稱
最後兩輪fixed quadratic必然恢復accuracy。正式證據見
`validation-results/fast-hard/stage-5/KNOWN3050_ACCURACY0_REJECTED.md`。

## Fast-only practical threshold follow-up

後續預先固定fast-only `3、5、8、10、12` iterations，在同一份3050-particle
K1 workload檢驗strict raw correlation `>=0.95`。結果沒有任何點達門檻；最高為
3 iterations的`0.948872`，比0.95低`0.001128`，但以`7.514 s`相對
balanced 12達`2.485x`。由3增加到12 iterations時，correlation單調降至
`0.946738`。

因此不接受「10 iterations內已建立Fast-95」的強結論。對本資料而言，若使用者
願意明確接受約`0.9489`而非嚴格`0.95`，3 iterations是此次測得的最佳
speed/quality點；不得將此單一K=1 observation推廣到K>1或不同SNR。完整
結果見`validation-results/fast-hard/stage-5/FAST95_SWEEP0_REJECTED.md`。

## Product-scope decision

使用者接受Fast 3在此K=1 workload比strict 0.95低`0.001128`的實用取捨，
並將兩種模式定位為：

1. `fast3`：反覆分群、對齊、檢視迴圈，固定3次polar-hard並輸出final raw
   class averages；
2. balanced workflow：checkpoint、最後收旂與accuracy-sensitive正式輸出。

此決定建立新的limited practical acceptance，不回溯更改frozen Stage 5 gate，
也不解禁5000-particle K10或Stage 6。新增公開`fast3` preset，而原有
`fast_hard`維持可自訂iteration的相容行為。GUI的Fast 3 exploration不會自動附加
已被拒絕的fixed precise stage。正式紀錄見
`validation-results/fast-hard/stage-5/FAST3_LIMITED_ACCEPTANCE.md`。
