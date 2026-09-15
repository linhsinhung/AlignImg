# 統一的 2-D cryo-EM Image Alignment 潛變量框架

## 文件目的

這份文件建立一套共同的數學與概念語言，用來理解下列看似不同的
2-D cryo-EM alignment 方法：

- reference-free alignment（RF）
- 單一 reference global alignment
- one-reference robust MAP-EM refinement
- multi-reference alignment（MRA）
- soft/Bayesian multi-reference alignment
- 已知 pose 或 class average 下的 warm-start refinement

核心觀點是：這些方法不需要被理解為互不相干的演算法。它們可以被視為同一個
latent-variable alignment framework 在不同初始化、模型數量、推論精度、更新規則
與運算 backend 下的具體設定。

這份文件主要回答「這些方法在數學上如何彼此關聯」。它不指定最終 public API，
也不宣稱目前 AlignImg、RE2DC RF/MRA 或 RELION 已完整實作本文所有模型。

## 閱讀導覽

- 第 1–4 節：問題、符號與統一生成模型。
- 第 5–6 節：hard MAP、top-\(L\)、soft posterior 與 EM update。
- 第 7 節：RF、single-reference 與 MRA 的精確關係。
- 第 8–11 節：初始化、bias、低 SNR 與 classification。
- 第 12–13 節：可伸縮的 CPU/GPU 計算分層。
- 第 14–15 節：典型工作流程與 diagnostics。
- 第 16–18 節：常見混淆、快速複習與核心結論。

---

## 1. Alignment 問題的本質

給定一組 noisy 2-D particle images：

\[
\mathcal{I}=\{I_1,I_2,\ldots,I_N\},
\]

我們希望估計每張 particle image 的 in-plane pose，使屬於相同結構的訊號能在共同
座標系中疊加。最基本的 pose 包含：

\[
g_i=(\theta_i,t_{x,i},t_{y,i},m_i),
\]

其中：

- \(\theta_i\) 是 in-plane rotation。
- \(t_{x,i},t_{y,i}\) 是平移。
- \(m_i\) 是是否 mirror 的離散狀態。

如果資料可能包含多種結構或多個 2-D view/class，我們還需要估計每張 particle
屬於哪一個 reference：

\[
k_i\in\{1,2,\ldots,K\}.
\]

實際資料也可能包含 contaminants、damaged particles 或無法被任何 reference
合理解釋的影像，因此再加入 inlier/outlier 狀態：

\[
c_i\in\{\text{inlier},\text{outlier}\}.
\]

把這些未知量放在一起，可將每張 particle 的隱變量寫成：

\[
z_i=(k_i,\theta_i,t_{x,i},t_{y,i},m_i,c_i).
\]

Alignment 的工作就是：根據影像 \(I_i\)、references、noise/CTF 資訊與已有的
prior knowledge，推論 \(z_i\)，再利用推論結果更新 references。

---

## 2. 真正需要統一的五個維度

一套 alignment 方法可以由五個近乎獨立的選擇描述：

\[
\boxed{
\text{初始化方式}
\times
\text{reference 數 }K
\times
\text{pose/class inference 強度}
\times
\text{reference update 規則}
\times
\text{運算 backend}
}
\]

### 2.1 初始化方式

References 可以來自：

- 使用者提供的外部 model。
- 全體 particles 的初始平均。
- 隨機 subset 的 sum/average。
- denoised particles 的平均。
- 多個 bootstrap starts。
- 前一輪 classification 產生的 class averages。

Pose 也可能完全未知，或來自上一輪 alignment/classification。

### 2.2 Reference 數量

- \(K=1\)：所有 inlier particles 由同一個 model 解釋。
- \(K>1\)：資料可由多個 classes/references 解釋。

### 2.3 Inference 強度

- hard deterministic：只保留最佳候選。
- hard MAP：只保留最佳候選，但加入 prior。
- top-\(L\)：只保留少量高分候選，再做較精確推論。
- soft posterior：保留 pose/class 的機率分布。
- full marginalization：對完整 pose/class 空間積分或加總。

### 2.4 Reference update 規則

- 等權平均。
- robust weighted average。
- soft class responsibilities。
- even/odd FSC-driven filtering。
- CTF/noise-weighted Fourier update。
- 帶 regularization 的 Bayesian update。

### 2.5 運算 backend

- scalar CPU
- multicore CPU
- batched CPU FFT
- CUDA polar search
- 未來的 CTF-aware CUDA scorer

Backend 決定怎麼算，不應暗中改變模型的 pose convention、score 意義或更新規則。

---

## 3. 符號表

| 符號 | 意義 |
| --- | --- |
| \(N\) | particle 數量 |
| \(K\) | reference/class 數量 |
| \(I_i\) | 第 \(i\) 張觀察影像 |
| \(R_k\) | 第 \(k\) 張 reference |
| \(g\) | pose，通常為 \((\theta,t_x,t_y,m)\) |
| \(T_g\) | 對 reference 或 particle 施加 pose 的 transform operator |
| \(k_i\) | particle 的 class/reference index |
| \(c_i\) | inlier/outlier 狀態 |
| \(z_i\) | particle 的完整隱變量 |
| \(CTF_i\) | 第 \(i\) 張 particle 的 CTF operator |
| \(\sigma_i\) | noise scale；也可依 Fourier frequency 改變 |
| \(\pi_k\) | class prior probability |
| \(q_i(z)\) | 對 particle \(i\) 的近似 posterior/responsibility |
| \(w_i\) | particle 的 robust inlier weight |
| \(T\) | posterior temperature，不是 transform operator |

---

## 4. 統一的生成模型

### 4.1 一個直覺性的生成故事

可把每張 particle 的產生過程想像成：

1. 以機率 \(p(c_i)\) 決定它是 inlier 或 outlier。
2. 若為 inlier，依 class prior \(p(k_i)=\pi_{k_i}\) 選擇 reference。
3. 依 pose prior 選擇 \(g_i=(\theta_i,t_{x,i},t_{y,i},m_i)\)。
4. 對 \(R_{k_i}\) 施加 pose transform。
5. 經過 particle-specific CTF。
6. 加入 noise，得到觀察影像 \(I_i\)。

對 inlier particle，可寫成：

\[
I_i = CTF_i\,T_{g_i}(R_{k_i})+\varepsilon_i,
\]

其中 \(\varepsilon_i\) 是 noise。

若暫時忽略 CTF，可簡化為：

\[
I_i = T_{g_i}(R_{k_i})+\varepsilon_i.
\]

這個簡化形式可以對應 real-space NCC、polar CCF 與一般 template matching；完整
形式則更接近 RELION 類型的 Fourier-domain likelihood。

### 4.2 Signal likelihood

若假設 Fourier-domain noise 近似 Gaussian，則一個典型的 likelihood 為：

\[
p(I_i\mid k,g,c_i=\text{inlier},R)
\propto
\exp\left[
-\frac{1}{2}
\sum_{\mathbf{q}}
\frac{
\left|
X_i(\mathbf{q})-
CTF_i(\mathbf{q})\,T_gR_k(\mathbf{q})
\right|^2
}{\sigma_i^2(\mathbf{q})}
\right],
\]

其中：

- \(X_i(\mathbf{q})\) 是 particle 的 Fourier coefficient。
- \(\mathbf{q}\) 是 spatial frequency。
- \(\sigma_i^2(\mathbf{q})\) 描述不同頻率下的 noise variance。

Real-space NCC 或 polar-ring CCF 可以被理解為這個 likelihood 在不同 normalization、
mask、noise 假設與離散搜尋策略下的近似 score。它們可能排序相似，但數值本身通常
不等價。

### 4.3 Priors

完整模型還包含：

\[
p(k)=\pi_k,
\]

\[
p(g\mid g_i^{\text{prior}}),
\]

以及：

\[
p(c_i).
\]

Pose prior 可以限制 angle/shift 不要偏離上一輪估計太遠。例如 Gaussian shift prior：

\[
\log p(t_x,t_y)
=
-\frac{1}{2}
\left[
\left(\frac{t_x-\mu_x}{\sigma_x}\right)^2+
\left(\frac{t_y-\mu_y}{\sigma_y}\right)^2
\right]+C.
\]

若有 warm-start pose，\(\mu_x,\mu_y\) 可以來自上一輪 pose；若希望把 particle
拉回影像中心，也可令 \(\mu_x=\mu_y=0\)。這兩種 prior 的科學意義不同，不應混為
一談。

Angle prior 必須尊重角度的週期性。例如：

\[
\Delta\theta=
\operatorname{wrap}(\theta-\theta^{\text{prior}}),
\]

再以 \(\Delta\theta\) 建立 circular Gaussian 或近似 Gaussian penalty。

### 4.4 Outlier model

若 \(c_i=\text{outlier}\)，可使用一個寬廣的 background model：

\[
p(I_i\mid c_i=\text{outlier})=p_{\text{out}}(I_i).
\]

於是完整 joint probability 可概念性寫成：

\[
p(I_i,z_i\mid R)
=
p(c_i)
\,p(k_i\mid c_i)
\,p(g_i\mid k_i,c_i,g_i^{\text{prior}})
\,p(I_i\mid k_i,g_i,c_i,R).
\]

在較簡單的實作中，不一定真的建立 \(p_{\text{out}}\)。以 score quantile、sigmoid
weight 或最低 class confidence 產生 \(w_i\)，可視為對 outlier posterior 的近似。

---

## 5. Posterior：不同 alignment 方法真正分歧的地方

給定 references，particle 的 posterior 為：

\[
p(z_i\mid I_i,R)
=
\frac{
p(I_i\mid z_i,R)p(z_i)
}{
\sum_{z_i'}p(I_i\mid z_i',R)p(z_i')
}.
\]

angle 和 translation 是連續變量時，分母中的加總在嚴格意義上包含積分。實際
程式通常將 pose space 離散化，或先取得少量候選再局部 refine。

不同方法的主要差異，不是有沒有 posterior，而是如何近似它。

### 5.1 Hard deterministic inference

只保留 image score 最高的候選：

\[
z_i^*=\arg\max_z S(I_i,z,R),
\]

\[
q_i(z)=\delta(z-z_i^*).
\]

這種方法速度快、行為清楚。當 SNR 高、最佳 peak 明顯時，hard 與 soft 方法通常會
得到非常接近的結果。

### 5.2 Hard MAP inference

加入 priors 後再取最佳候選：

\[
z_i^*
=
\arg\max_z
\left[
\log p(I_i\mid z,R)+\log p(z)
\right].
\]

這仍然是 hard assignment，但不再是單純 deterministic correlation maximum。
現行 AlignImg 的 robust MAP-EM 概念較接近這個形式。

### 5.3 Top-\(L\) approximation

先由快速搜尋取得：

\[
\mathcal{C}_i=\{z_{i1},z_{i2},\ldots,z_{iL}\},
\]

再只在 \(\mathcal{C}_i\) 中計算較精確的 likelihood 與 prior：

\[
q_i(z)
\approx
\frac{
\exp(\ell_i(z))
}{
\sum_{z'\in\mathcal{C}_i}\exp(\ell_i(z'))
},
\quad z\in\mathcal{C}_i.
\]

這是 fast deterministic proposal 與 Bayesian re-ranking 之間的重要折衷。

### 5.4 Soft posterior

保留多個 class/pose hypotheses：

\[
q_i(z)\approx p(z\mid I_i,R).
\]

Reference update 不只使用最佳 pose，而是讓每個候選依 posterior probability 貢獻。
低 SNR 下，當多個 pose score 接近時，這能避免過早把 particle 鎖定在 noise peak。

### 5.5 Tempered posterior 與 annealing

可加入 temperature：

\[
q_i(z;T)
\propto
\exp\left(\frac{\ell_i(z)}{T}\right).
\]

- \(T>1\)：posterior 較平，探索較多候選。
- \(T=1\)：原始 posterior。
- \(T\rightarrow0\)：逐漸接近 hard-MAP。

一個常見策略是初期使用較高 temperature，reference 逐漸穩定後再降溫。這不是保證
避開所有 local optimum，但能延後過早的 hard commitment。

---

## 6. EM 與 reference update

統一框架可以用 EM 的語言理解。

### 6.1 E-step

在 references 固定時，估計：

\[
q_i(z)=p(z_i=z\mid I_i,R).
\]

Hard alignment 將 \(q_i\) 近似為單一 delta；soft alignment 則保留分布。

### 6.2 M-step

在 responsibilities 固定時更新 references：

\[
R_k^{\text{new}}
=
\arg\max_{R_k}
\sum_i\sum_z
q_i(z)
\log p(I_i\mid z,R_k).
\]

忽略 CTF、frequency-dependent noise 與 regularization 時，它可簡化成 aligned images
的 weighted average：

\[
R_k^{\text{new}}
=
\frac{
\sum_i\sum_g
q_i(k,g,c=\text{inlier})\,T_g^{-1}(I_i)
}{
\sum_i\sum_g q_i(k,g,c=\text{inlier})
}.
\]

若沒有顯式建立 outlier variable，可用
\(q_i(k,g)w_i\) 近似 \(q_i(k,g,c=\text{inlier})\)。若採 hard pose/class assignment，
公式就退化為每類 aligned particles 的平均。

在 CTF-aware Fourier model 下，更新會更接近：

\[
R_k^{\text{new}}(\mathbf{q})
=
\frac{
\sum_{i,z:k(z)=k}
q_i(z)
\,CTF_i^*(\mathbf{q})
\,T_{g(z)}^{-1}X_i(\mathbf{q})
/\sigma_i^2(\mathbf{q})
}{
\sum_{i,z:k(z)=k}
q_i(z)
\,|CTF_i(\mathbf{q})|^2
/\sigma_i^2(\mathbf{q})
+\lambda_k(\mathbf{q})
}.
\]

\(\lambda_k\) 代表 frequency-dependent regularization。這說明 RELION 類方法的效果
不只是「使用 soft posterior」，也來自 CTF、noise model 與 regularized update。

### 6.3 為什麼很多實作仍叫 MAP-EM

實務上常使用：

- hard pose estimate
- robust particle weight
- alternating reference update

這並不是對完整 pose posterior 做精確 marginalization，但仍保留「估 pose／更新 model」
的 alternating EM 結構。因此更精確的稱呼可能是 hard-MAP EM 或 generalized EM。

---

## 7. RF、單 reference 與 MRA 如何由同一模型導出

| 模式 | \(K\) | Reference 初始化 | Pose/Class inference | Reference update |
| --- | ---: | --- | --- | --- |
| Homogeneous RF | 1 | particles bootstrap | hard、top-\(L\) 或 soft pose | 單一共識 average |
| 單 reference global alignment | 1 | 使用者提供 | global pose，通常無 warm prior | 單一 reference update |
| Robust refinement | 1 | class average 或可信 reference | pose prior + robust inlier inference | robust weighted average |
| 傳統 MRA | \(>1\) | 多張 class averages | hard class + hard pose | 每類 hard average |
| Bayesian MRA | \(>1\) | 多張 references | soft class + pose marginalization | soft responsibility update |
| Warm-start class refinement | \(>1\) | classification averages | pose/class priors | class-wise robust update |

### 7.1 Reference-free 不是一種特定 search kernel

「Reference-free」首先描述的是沒有外部 initial reference，而不是 polar search、NCC、
hard assignment 或 soft inference 中的任何一種。

因此下面都可以是 reference-free：

- \(K=1\) 的 homogeneous consensus alignment。
- \(K>1\) 的 multi-start 或 ab-initio multi-class alignment。
- hard deterministic RF。
- soft Bayesian RF。

在「假設全部資料 homogeneous，朝同一個 model 對齊」的使用情境中，RF 通常設定
為 \(K=1\)，但這是建模假設，不是 reference-free 的定義。

### 7.2 K=1 的 MRA 是否等價於 one-reference MAP-EM？

在抽象 latent-variable model 中，令 \(K=1\) 後，class index \(k_i\) 變成常數，
MRA 的 class-selection 部分確實退化為 single-reference alignment。

但是，\(K=1\) 不會自動統一：

- likelihood 或 similarity score
- Cartesian 與 polar representation
- angle/translation candidate generation
- mirror policy
- pose prior
- hard 或 soft inference
- outlier weighting
- reference filtering
- centering 與 transform interpolation

因此「K=1 MRA」與「現行 robust MAP-EM」可以共享同一個 framework，但目前不是數值
等價的演算法。要達到真正等價，還必須讓 scorer、priors、candidate set、posterior
reduction 與 M-step 全部一致。

---

## 8. 初始化：Reference-Free 最困難的部分

沒有 initial reference 時，必須從 particles 建立初始 model。常見策略如下。

### 8.1 全體平均

\[
R^{(0)}=\frac{1}{N}\sum_i I_i.
\]

最簡單，但如果 particles 的旋轉與平移非常分散，平均會高度模糊。

### 8.2 隨機 subset sum/average

以部分 particles 建立初始概括。速度快，但結果依 random seed 改變，也可能放大特定
orientation、class 或 noise pattern。

### 8.3 Denoised consensus

先以 2SDR 或其他 denoiser 提高結構可見度，再建立 reference 或估計初始 poses。
較安全的分工是：

\[
\text{denoised data 提出 pose candidates}
\rightarrow
\text{raw data 做最終 scoring/update}.
\]

這能利用 denoising 的穩定性，又降低 denoiser hallucination 直接進入最終 reference
的風險。

### 8.4 Multi-start bootstrap

建立多個 \(R_b^{(0)}\)，各自短跑數輪，再依 half-set FSC、likelihood 或 stability
選擇：

\[
b^*=\arg\max_b Q(R_b).
\]

它不能完全消除 reference bias，但能降低單次隨機初始化決定整個結果的風險。

### 8.5 Class-average warm start

經過 classification 後，可使用 class average 作為更高 SNR 的 reference，並使用已有
pose/class assignment 作為 prior。這通常是 refinement 最有利的情境。

### 8.6 無 reference 問題的不可識別性

純 RF alignment 通常只能決定相對 pose，無法由資料自行決定唯一的全域角度、平移
原點或 handedness。若把所有 particles 與 reference 一起旋轉同一角度，資料 likelihood
可能不變。

因此 RF controller 需要選擇一個 gauge，例如：

- 每輪將 reference 置中。
- 固定 reference 的全域 orientation convention。
- 明確決定 mirror 是否允許。

這些操作不是額外估出真實的絕對 pose，而是在多個等價解之間選擇一致表示。

---

## 9. Reference update 與 bias control

### 9.1 等權平均

\[
R^{\text{new}}=\frac{1}{N}\sum_i I_i^{\text{aligned}}.
\]

簡單，但 contaminants 與錯誤 pose 會直接進入 reference。

### 9.2 Robust weighted average

\[
R^{\text{new}}
=
\frac{\sum_i w_i I_i^{\text{aligned}}}{\sum_i w_i}.
\]

\(w_i\) 可以來自：

- sigmoid score weight
- score quantile
- inlier posterior \(p(c_i=\text{inlier}\mid I_i)\)
- class/pose posterior confidence

需要注意：若 weight 直接由當前 reference correlation 產生，可能把真正但較少見的
heterogeneity 當成 outlier。這也是何時該使用 \(K>1\) 而不是只加強 robust rejection
的關鍵。

### 9.3 Even/odd FSC-driven update

將 particles 固定分成 even/odd halves，分別建立：

\[
R_{\text{even}},R_{\text{odd}}.
\]

兩者的 Fourier shell correlation 可用來估計哪些 frequency 仍有可重現訊號，再決定
reference filter。相較固定 Gaussian low-pass，這是 data-adaptive regularization。

### 9.4 Half-set 不等於完全沒有 bias

如果同一 particle 參與 reference 建立，又對齊到包含自己的 reference，noise 仍可能
產生 self-alignment bias。更嚴格的方法會讓 particle 使用 opposite-half reference，
或至少分別維持兩套獨立 references。

### 9.5 Class-wise update

MRA 對每個 class 分別更新：

\[
R_k^{\text{new}}
=
\frac{
\sum_i r_{ik} I_{ik}^{\text{aligned}}
}{
\sum_i r_{ik}
},
\]

其中 \(r_{ik}\) 是：

- hard MRA：0 或 1。
- soft MRA：\(p(k_i=k\mid I_i)\)。

小 class 需要明確 policy，例如保留舊 reference、reseed、merge 或刪除。不同 policy
會改變最終 class 數量與 reproducibility。

---

## 10. Deterministic 與 Bayesian：不是二選一

### 10.1 高 SNR

當正確 pose 的 likelihood 遠高於其他候選：

\[
p(z_i^*\mid I_i,R)\approx1,
\]

soft posterior 自然接近 hard assignment。此時 deterministic 與 Bayesian 方法可能只在
subpixel interpolation、filtering 或 transform convention 上有小差異。

### 10.2 低 SNR

當多個 noise peaks 的 score 接近：

\[
p(z_{i1}\mid I_i,R)
\approx
p(z_{i2}\mid I_i,R)
\approx\cdots,
\]

hard search 仍必須挑一個 winner。錯誤 pose 隨後被平均進 reference，reference 又會強化
下一輪相同的錯誤，形成 feedback loop。

Soft posterior、temperature、pose prior 與 frequency regularization 可以降低這種過早
鎖定，但不能單獨保證找到 global optimum。若 likelihood、noise model 或 reference
本身有偏差，Bayesian inference 仍可能很有信心地得到錯誤結果。

### 10.3 RELION 類方法為何有效

低 SNR 下的效果通常來自整套設計，而不只是「使用 MAP」：

- pose/class marginalization
- CTF-aware Fourier likelihood
- frequency-dependent noise model
- regularized reference update
- resolution schedule
- priors
- half-set validation

因此，把 NCC 最大值加上一個 prior penalty 是有用的 MAP 近似，但尚不等於完整的
RELION inference。

---

## 11. Alignment 與 classification 的關係

Alignment 與 classification 互相依賴：

- pose 不準確會模糊 class features。
- class 分錯會產生錯誤 reference。
- 錯誤 reference 又會把 pose 拉向錯誤結構。

MRA 可被理解為同時估計 class 與 pose 的 hard-EM：

\[
(k_i^*,g_i^*)
=
\arg\max_{k,g}S(I_i,R_k,g).
\]

Soft MRA 則保留：

\[
r_{ikg}=p(k_i=k,g_i=g\mid I_i,R).
\]

如果上層 classification 使用 2SDR、gamma-SUP 或其他 feature model，可有兩種分工：

1. Alignment engine 自己估 class responsibilities。
2. Classification engine 提供 class prior/assignment，alignment 只 refine poses。

統一 framework 應允許第二種情境注入：

\[
p(k_i)=\pi_{ik}^{\text{external}},
\]

而不是強迫所有 classification 都由 image correlation 決定。

---

## 12. 可伸縮的計算架構

數學模型與計算方法應分層。推薦將一次 iteration 拆成五個角色。

### 12.1 Candidate proposer

快速找出可能的 pose/class 候選，例如：

- Cartesian coarse angle scan + translation FFT
- polar-ring FFT
- local search around prior pose
- GPU multi-reference/mirror search

### 12.2 Scorer

對候選計算可比較的 score：

- masked NCC
- polar CCF
- whitened Fourier NCC
- CTF/noise-aware log likelihood

### 12.3 Posterior reducer

將候選轉為：

- top-1 hard pose
- top-\(L\) normalized weights
- soft class/pose posterior
- robust inlier probability

### 12.4 Reference updater

依 responsibilities、inlier weights、CTF 與 filter policy 更新 references。

### 12.5 Controller

管理：

- iteration schedule
- initialization
- centering
- half sets/FSC
- auto-stop
- class reseeding
- diagnostics

這個分層的重要性在於：同一個 GPU proposer 可以搭配不同 scorer/reducer；同一套
Bayesian controller 也能使用 CPU 或 CUDA backend。

---

## 13. GPU polar search 在統一框架中的位置

高效率 polar CUDA kernel 最適合被定位為 candidate proposer，或在特定 score 定義下
同時擔任 proposer/scorer。

一個可伸縮的 hybrid 流程是：

```text
Particles + References
          │
          ▼
Fast polar GPU search
          │
          └── top-L (class, mirror, angle, shift, CCF)
                         │
                         ▼
MAP/Bayesian re-scoring
  ├── pose prior
  ├── class prior
  ├── outlier probability
  ├── CTF/noise likelihood
  └── temperature
                         │
                         ▼
Posterior / hard assignments
                         │
                         ▼
Robust + FSC-aware reference update
```

這種架構保留 polar search 的速度，也避免將整套科學模型永久綁定在 raw polar CCF
與 hard argmax 上。

值得沿用的 GPU 工程原則包括：

- reference transform/FFT 預先計算
- batched particles
- 固定 FFT microbatch
- fused argmax 與 angle interpolation
- 依 free VRAM 設定 budget
- OOM 時縮小 batch
- lazy runtime initialization
- 明確失敗，不靜默更換科學 backend
- 完整記錄 H2D、kernel、D2H 與 memory diagnostics

---

## 14. 四個典型工作流程

### 14.1 無 reference、無 pose、假設 homogeneous

設定：

\[
K=1,\quad R^{(0)}=\operatorname{bootstrap}(I),\quad p(g)=\text{broad}.
\]

建議流程：

1. 由 denoised 或 multi-start bootstrap 建立初始 reference。
2. 做 broad/global candidate search。
3. 初期使用 top-\(L\) 或較高 temperature。
4. 以 half-set/FSC 更新與過濾 reference。
5. 後期降低 temperature、縮小 pose search range。
6. 回到 raw data refinement。

### 14.2 有可信 reference、無 pose

設定：

\[
K=1,\quad R^{(0)}=R_{\text{provided}},\quad p(g)=\text{broad}.
\]

主要需求是 global search。若 reference 可靠、SNR 尚可，可使用快速 hard/top-\(L\)
搜尋；低 SNR 時則使用 frequency schedule 與 soft re-ranking。

### 14.3 有 reference 與 prior pose

設定：

\[
K=1,\quad R^{(0)}=R_{\text{class}},\quad p(g\mid g^{\text{prior}})=\text{local}.
\]

此時不必重新做完整 global search。可在 prior pose 周圍做 local proposal，使用 robust
inlier weights 防止 class contaminants 破壞 average。

### 14.4 多 references、class/pose 皆不確定

設定：

\[
K>1,\quad p(k)=\pi_k,\quad p(g)=\text{broad or warm}.
\]

Hard MRA 只保留最佳 \((k,g)\)；soft MRA 保留多個 class/pose hypotheses。低 SNR 或
classes 相似時，class prior、occupancy regularization 與 posterior entropy 會非常重要。

---

## 15. 應該記錄哪些 diagnostics

一套可研究、可除錯的 alignment framework 不應只輸出 final poses。至少應考慮：

### Pose diagnostics

- angle/shift distribution
- prior-to-posterior pose change
- top-1/top-2 score margin
- posterior entropy
- mirror fraction
- boundary-hit fraction

### Particle diagnostics

- image likelihood 或 normalized score
- inlier probability/robust weight
- assigned class 與 class probability
- alignment improvement on raw data

### Reference diagnostics

- reference history
- even/odd FSC
- frequency cutoff/falloff
- center shift
- effective particle count
- per-class occupancy
- small/empty-class action

### Backend diagnostics

- candidate evaluations
- batch size
- CPU/GPU time breakdown
- H2D/kernel/D2H timing
- peak VRAM
- OOM reductions
- numerical backend/revision

這些資訊能協助區分「搜尋沒有找到正確 candidate」、「likelihood 排序錯誤」、
「posterior 過早變尖」與「reference update 放大錯誤」等不同失敗來源。

---

## 16. 常見混淆

### 混淆一：Reference-free 就是把 particle mean 當 initial reference

Particle mean 是一種 RF initializer，但 RF 還需要處理 centering、filtering、bias、
multi-start 與後續 inference。

### 混淆二：K=1 MRA 就一定等於現行 single-reference MAP-EM

K=1 只移除 class competition；score、search、prior、mirror、outlier 與 M-step 仍可能
完全不同。

### 混淆三：MAP 就等於 soft Bayesian marginalization

MAP 只取 posterior mode。Soft Bayesian inference 保留多個 hypotheses。兩者都可放在
EM controller 中，但 reference update 的資訊量不同。

### 混淆四：Soft inference 一定能避開 local optimum

Soft posterior 能延後 hard commitment，但錯誤 likelihood、reference bias 或不適當 prior
仍會導向錯誤解。

### 混淆五：GPU backend 只是 CPU 程式跑得更快

如果 GPU 使用不同 ring resolution、candidate grid、interpolation 或 reduction order，
結果可能只在科學 tolerance 內相容，而非 bitwise 相同。Backend contract 必須明確規定
允許差異。

### 混淆六：Denoising 後 alignment 變好，代表最終 reference 也應由 denoised data 更新

Denoised data 很適合 proposal/initialization，但可能含有 model bias。最終驗證與更新回到
raw、CTF-aware data 通常較安全。

---

## 17. 最精簡的記憶方式

感到混亂時，可以回到下面五個問題：

1. **Reference 從哪裡來？**
   外部 model、particle bootstrap、denoised consensus，還是 class averages？

2. **模型假設有幾類？**
   \(K=1\) homogeneous，還是 \(K>1\) heterogeneous？

3. **每張 particle 保留多少可能解？**
   top-1、top-\(L\)，還是完整 soft posterior？

4. **哪些 particles、poses 與 frequencies 能進入 reference？**
   等權、robust、soft class、FSC-filtered，還是 CTF/noise weighted？

5. **Backend 只改變速度，還是也改變科學行為？**
   Candidate set、score、transform convention 與 numerical tolerance 是否一致？

只要這五個問題回答清楚，RF、single-reference MAP-EM 與 MRA 就能放回同一張概念地圖。

---

## 18. 核心結論

RF、單一 reference alignment 與 MRA 的共同核心是：

\[
\boxed{
\text{推論 particle 的 class/pose/outlier 隱變量}
\longleftrightarrow
\text{利用 responsibilities 更新 references}
}
\]

它們的差異主要來自：

- reference 是否已知
- \(K\) 是 1 還是大於 1
- pose/class posterior 採 hard、top-\(L\) 或 soft 近似
- 是否使用 pose/class/outlier priors
- reference 如何 regularize、filter 與 center
- candidate search 與 likelihood 由哪個 backend 計算

因此，一套可伸縮的 Image Alignment 工具不必選邊站在 deterministic 或 Bayesian、RF
或 MRA、CPU 或 GPU。比較合理的設計是保留同一個明確的 latent-variable model，再依
資料 SNR、reference quality、是否有 prior pose、class heterogeneity 與可用算力，選擇
不同精度與成本的 inference strategy。
