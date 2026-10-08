# AlignImg 測試分層

整理日期：2026-09-26。目標是讓AlignImg作為通用image alignment library，日常回歸
專注RF、MRA、pose與數值/resource契約；不把歷史研究流程當作現在的產品需求。

## 測試範圍

| 位置 | 用途 | 預設執行 |
| --- | --- | --- |
| `tests/core/` | Public API、Balanced／Fast 3／refine、RF／MRA、priors、geometry、Fourier、raw average與profiling | CPU案例是；`gpu`案例否 |
| `tests/gpu/` | Backend memory planner、workspace、device-pointer wrappers、polar arithmetic的CPU emulation | 是，不需要CUDA |
| `tests/gui/` | Workbench設定、模式切換、artifact round-trip與offscreen UI | 否 |
| `tests/validation/` | 現行validation/report工具、frozen-input管理、source manifest與archive安全性 | 否 |
| `tests/historical/` | 2.1 development stage gates、2.2/2.3特定A/B及RE2DC/RELION benchmark研究流程 | 否 |
| `tests/integration/` | 需要本機真實資料的可選public API smoke | 否 |

`gpu` marker表示**真正需要GPU硬體**，不是檔名包含GPU。Core檔案中的CPU/GPU
parity案例有此marker；`tests/gpu/`目前的23個案例使用fake arrays、NumPy或mock，
所以仍在日常測試。新增真正需要GPU的案例時必須加上`@pytest.mark.gpu`。

測試啟動時會優先使用環境中已安裝的`alignimg-gpu`，使native `_native` extension
可被找到；只有未安裝GPU package時才把repository內的Python fallback加入path。
因此日常CPU測試不要求安裝GPU package，而server硬體測試會檢查實際安裝版本。

## 執行方式

### 2.3.1 GitHub source 補充

Polar batch/recovery/streaming 的 CPU regression 使用
`tests/fixtures/polar_231/frozen_t2_backend.py`，不再依賴未納入 Git 的
`validation-results/`。這份 historical T2 solver 與已驗證 fixture 逐 byte 相同，
SHA-256 為 `cd0524a4f7d702ba0fc97ff2cc8270519789781da4ad6137a4d8d1d657ff412a`；
loader 仍會核對此雜湊。它只用於 regression 對照，不是 runtime backend。

這是封存後的測試可攜性修正，不改 alignment runtime、數值門檻、版本字串或既有
freeze archives。乾淨 clone 可以執行 CPU 產品與工具測試；server GPU gates 與
需要真實 benchmark inputs 的歷史實驗，仍須對應硬體及資料。

專案root下的日常回歸：

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Pytest預設只搜尋`tests/core`與`tests/gpu`，並套用`-m 'not gpu'`。這是181個
不需要GUI、真實GPU或私有baseline的案例，不是完整suite。

GPU host上明確執行硬體parity/resource回歸：

```bash
python -m pytest -q -m gpu
```

CLI的`-m gpu`會覆蓋預設marker filter，選出20個CUDA/CuPy案例。缺少backend時
仍依原有規則skip；全skip不代表GPU驗證成功。正式native安裝驗證另外使用
`tools/server_validation.py --suite quick --backend cuda`，不允許默默fallback。

目前產品與工具的完整回歸（不含歷史實驗）：

```bash
python -m pytest -q tests/core tests/gpu tests/gui tests/validation -m ""
```

這會收集252個案例，並包含硬體案例。GUI artifact測試需要`mrcfile`；offscreen UI
另外需要Workbench dependencies（PyQt6、pyqtgraph）。可依需要單跑：

```bash
python -m pytest -q tests/gui
python -m pytest -q tests/validation
python -m pytest -q tests/historical
RUN_ALIGNIMG_INTEGRATION=1 python -m pytest -q tests/integration -m integration
```

完整檢查，包含歷史與可選案例：

```bash
python -m pytest -q tests -m ""
```

空的marker expression清除預設filter。整理前後皆收集336個案例，所有test function
與parameter IDs的出現次數相同。Optional案例仍依原規則skip；私有資料未提供時，
完整suite不等於所有硬體與歷史baseline均已驗證。

## 這次整理的取捨

- 不刪除任何既有案例、不放寬assertion，也不改運算程式、驗收threshold或frozen records。
- 將現行`test_v1_workflows.py`改名為`core/test_workflows.py`；名稱中的v1不代表內容已過時。
- 將RE2DC pose benchmark中的兩個generic fixed-MRA/K1等價性測試移至
  `core/test_fixed_mra.py`，保留Balanced/adaptive契約；polar/quadratic同類契約仍各自保留，
  因為驗證的是不同search strategies，不能當重複案例刪掉。
- Profiling的engine觀測性留在core；report/source/failure handling移至validation；依賴
  pre-instrumentation baseline的三個案例移至historical。
- Server validation中的CPU A/B仍測report runner和backend歸屬，與engine單元測試目的
  不同；不刪斷言，但不再每次日常pytest重複執行。
- Stage 1仍要求`2.1.0.dev1`、Stage 7仍描述frozen dev10/5000-particle規格。這些只供
  歷史重現與工具維護，不要求2.3沿用舊版本freeze或恢復已停止實驗。
- RE2DC/RELION的CTF、STAR parsing、benchmark conversion及frozen實驗保留在historical，
  不因此新增RE2DC adapter或把它列入AlignImg release gate。
- Public legacy pose conversion、integer-origin/mirror與舊GUI artifact讀取是仍提供的
  相容性契約，繼續測試，不因名稱帶legacy就刪除。

## 後續新增與FTP注意事項

新增測試依上表歸類；避免新增Stage編號來組織通用回歸，或在core依賴私有資料。
測試工具的修改需跑validation；GUI修改需跑gui；backend/kernel修改需在server跑gpu
及explicit-backend quick。發行前跑目前產品完整回歸，歷史suite在維護歷史工具時另跑。

這次搬移會改變source manifest。已封存的`alignimg-2.3.0-server-handoff.tar.gz`沒有
改寫，仍使用整理前的測試布局與156-file manifest。請勿把新tests/config與舊handoff的
hash混用。下次交付整理後的source時，需要重新打包及產生manifest；單純FTP覆蓋也
不會移除原`tests/test_*.py`，需先備份舊tests再替換，否則完整收集會重複。
