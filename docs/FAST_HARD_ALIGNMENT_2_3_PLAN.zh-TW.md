# AlignImg 2.3 release scope 與收尾計畫

更新日期：2026-09-27。狀態：**功能及原sealed artifact server安裝驗證已接受；
整理後repository的交付與server回歸待完成**。
版本：`alignimg 2.3.0`、`alignimg-gpu 2.3.0`、Workbench `0.7.0`。

本文件承接2026-09-15的原始fast-hard proposal與Stage 0–5實驗紀錄，記錄使用者
審查結果後確認的產品範圍。原始fast→fixed-precise的accuracy gate及rejection繼續
保留為科學證據；本次正式版本採用已接受的Fast 3 exploration用途。

## 產品範圍

AlignImg是一套通用2D image alignment library，輸入遵守even-square image contract。
它透過既有`reference_free_align()`、`align_to_references()`與`refine_alignment()`
提供RF、K=1 alignment與K>1 MRA；K=1及K>1共享計算路徑、pose與result契約。
主要validation datasets來自cryo-EM，尚不宣稱其他image domains具有相同accuracy。

- `fast3`：3次polar-hard，top-1 hard inference，final raw reconstruction；適合
  反覆分群、對齊、檢視的探索迴圈。
- balanced workflow：保留既有soft global/RF預設；用於checkpoint、最後收斂及
  accuracy-sensitive輸出。Local quadratic refinement由caller按需要選擇。
- `fast_hard`：保留原有可自訂iteration及output policy的底層preset。
- 模式由caller明確選擇；review diagnostics供判斷，沒有automatic switching或
  自動增加iterations。

## 已完成與停止項目

| 階段 | 結果 |
|---|---|
| Stage 0 | 2.2 baseline、inputs與hashes凍結完成 |
| Stage 1 | Config-only調查完成；結果支持進入true-polar實作 |
| Stage 2 | CPU-authoritative polar-hard完成 |
| Stage 3 | CuPy/native CUDA parity、performance、resource接受 |
| Stage 4 | Public preset、GUI、diagnostics、runner接受 |
| 原始Stage 5 | Fixed fast→precise未通過frozen gates；rejection保留 |
| Fast 3補充 | K1 sweep完成；使用者接受quality取捨，limited acceptance及server tests完成 |
| 5000 K10 | 依原始gate停止；不是2.3修訂範圍的必要項目 |
| 原Stage 6 | 改為下游integration文件；實際adapter由下游repository開發 |

3050-particle K1的Fast 3 raw correlation為`0.948872`，median wall為`7.514 s`，
相對balanced 12的`18.674 s`為`2.485x`。這是不同iteration數的實用schedule比較，
不是相同iteration的solver speedup。Strict `0.95`門檻仍未通過；使用者明確接受
`0.001128`的差距。低SNR及K>1不能套用相同quality保證。

## 正式封版步驟

1. Core/GPU/native版本升為2.3.0，GUI升為0.7.0；保持已接受preset及數值行為。
2. 完成本文件、release freeze及下游integration guide。
3. 完整pytest、相關lint、clean build與artifact內容/metadata/checksum audit。
4. FTP上傳乾淨handoff，在server原位置覆蓋、安裝正式artifacts。
5. 驗證source manifest、core/GPU/native/GUI identities與一次CUDA quick。
6. Identity及quick JSON已帶回並於2026-09-27接受：156-file source manifest、
   core/GPU/native 2.3.0、GUI 0.7.0、CUDA quick 22 passed / 0 failed。

後續測試布局與README/ignore整理不改變compute/native程式；原sealed artifact不改寫。
發佈整理後repository前，另交付新manifest、備份替換server舊tests，再跑252個目前
產品案例。cleanup-r2已於2026-09-27通過server驗收：252 passed、0 failed、0 errors、
0 skipped，且確認使用`site-packages`內已安裝的GPU package與native CUDA module。
不需再次重編CUDA或重跑quick/大型實驗。

不需重新執行3050 A/B、SNR sweep或5000 K10。下游RE2DC整合契約記錄於
`DOWNSTREAM_INTEGRATION_2_3.zh-TW.md`；AlignImg release不以其adapter完成為條件。
