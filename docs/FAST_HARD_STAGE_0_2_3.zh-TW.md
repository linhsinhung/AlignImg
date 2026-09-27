# AlignImg 2.3 Fast-Hard：Stage 0 baseline 封存

狀態：**ACCEPTED**；正式記錄見
`validation-results/fast-hard/stage-0/ACCEPTED.md`。

本階段只建立 2.2 baseline 證據，不修改 alignment core、GPU package、公開 preset
或版本號。Stage 1 的 `top_l=1` config-only 候選必須等本文件列出的 CPU、CuPy、
native CUDA 報告與 frozen inputs 完成複查後才可加入。

## 假設與成功條件

- 基準為目前 `alignimg==2.2.0` 與同版 `alignimg-gpu`／native extension。
- A/B 共用三個 iterations、frequency band、mask、reference update、輸入與 seed。
- throughput 測量在同一 process 先 warm-up 一次，再正式執行至少三次並報告 median
  及全部 raw times。
- 每個 suite 的 `frozen-inputs/*.inputs.npz` 只建立一次；後續 stage 必須傳入並
  驗證相同 SHA-256，不可重新抽樣。
- 所有輸出 finite、responsibility row-sum 誤差不超過 `2e-6`、requested backend
  不得 fallback，且同 backend 重跑的 result hash 必須一致。

## 新增 runner

`tools/fast_hard_validation.py` 提供固定介面：

```bash
python tools/fast_hard_validation.py \
  --suite synthetic --backend cuda \
  --batch-size 512 --memory-fraction 0.8 \
  --deterministic-repeats 2 --profile-execution \
  --inputs validation-results/fast-hard/stage-0/frozen-inputs/synthetic.inputs.npz \
  --output validation-results/fast-hard/dev0-cuda-synthetic.json

python tools/fast_hard_validation.py \
  --suite homogeneous --backend cuda \
  --batch-size 512 --memory-fraction 0.8 \
  --deterministic-repeats 2 --profile-execution \
  --inputs validation-results/fast-hard/stage-0/frozen-inputs/homogeneous.inputs.npz \
  --output validation-results/fast-hard/dev0-cuda-homogeneous.json

python tools/fast_hard_validation.py \
  --suite mra1000 --backend cuda \
  --batch-size 512 --memory-fraction 0.8 \
  --deterministic-repeats 2 --profile-execution \
  --inputs validation-results/fast-hard/stage-0/frozen-inputs/mra1000.inputs.npz \
  --output validation-results/fast-hard/dev0-cuda-mra1000.json
```

第一次執行會建立：

```text
validation-results/fast-hard/stage-0/frozen-inputs/synthetic.inputs.npz
validation-results/fast-hard/stage-0/frozen-inputs/homogeneous.inputs.npz
validation-results/fast-hard/stage-0/frozen-inputs/mra1000.inputs.npz
```

也可先以 `--freeze-only` 建立 inputs。Runner 拒絕覆寫既有 report 或 result，並在
JSON 內記錄 frozen-input、原始資料、source、preset、config 與 result hashes。
Code-only FTP handoff 沒有 `.git/` 時，報告明確記錄 `git.available=false`；這不是失敗，
正式 provenance 改由已驗證的完整 source SHA-256 與 bundle SHA-256 提供。

## Workloads

- `synthetic`：非對稱 K=1 identity／90 度／任意角／整數與小數 shift，mirror
  off/on；K=3 noiseless、SNR 0.5、SNR 0.2。
- `homogeneous`：既有 384-particle RELION benchmark 的三組 K=1 與 joint fixed K=3。
- `mra1000`：既有 1000-particle prepared 70S 與凍結的 K=10 references；只記錄
  技術行為，不作 biological gate。
- 3050-particle known-reference 與 5000-particle 三 seeds 留到 Stage 5，不在 Stage 0–3
  重跑。

## 報告欄位

每個 cell 包含 wall time、iteration time、候選數估計、responsibility、assignments、
occupancy、pose/reference/raw-average quality、deterministic hashes、profiling tree、
H2D/D2H counters、GPU memory samples、VRAM plans、workspace 與 actual batch。CPU 報告
用於數學與 runner smoke；正式 throughput 只以 clean NVIDIA server 的 native CUDA
報告判定。

## Stage 0 acceptance checklist

1. 本機完整 `python -m pytest -q` 通過。
2. 建置並檢查 core/GPU 同版 dev artifact；server 安裝來源與 checkout hash 一致。
3. CPU、CuPy、CUDA 各執行 quick、synthetic 與 fixed-MRA conformance；CUDA 另執行
   `homogeneous`、`mra1000`。
4. 帶回所有 JSON，確認 frozen input SHA-256 相同、CUDA 無 fallback、batch 512 遵守
   80% VRAM hard limit、deterministic hashes 與品質欄位完整。
5. 分開記錄 numerical、scientific、performance、resource 與 known limitations；本階段
   不比較 fast/balanced，也不宣告速度改善。

完成以上項目並寫入 `validation-results/fast-hard/stage-0/ACCEPTED.md` 前，不新增
`fast_hard` preset，也不進入 `polar_hard`。

## FTP server handoff

FTP 的「覆蓋資料夾」通常只會更新同名檔，不會移除新版已刪除的舊檔。正式 Stage 0
不得在舊 code tree 上做 merge overlay。固定沿用既有 server 路徑時，先把 code directories
移入 `validation-results` 下的可復原備份，再於原位置解開 code-only handoff archive；既有
conda 環境、`data/` 與 `validation-results/` 不需重建。

Handoff archive 必須以 portable tar 建立，不含 macOS AppleDouble `._*`、xattr 或 ACL。
在 server 原專案根目錄執行：

```bash
cd /media/linhsinhung/Data_12T_2/test_align
mkdir validation-results/fast-hard/stage-0/server-code-backup
mv src packages tests tools docs examples third_party legacy \
  validation-results/fast-hard/stage-0/server-code-backup/
tar -xzf alignimg-2.3-stage0-server-handoff.tar.gz -C .
```

若仍使用逐資料夾上傳，也必須先移走上述 code directories，不可 merge overlay，並保留
相對路徑。Handoff 內容至少包含：

```text
tools/fast_hard_validation.py
tools/performance_fixtures.py
tests/test_fast_hard_validation.py
tests/test_calculation_reuse.py
tests/test_execution_profiling.py
docs/FAST_HARD_STAGE_0_2_3.zh-TW.md
packages/alignimg-gpu/dist/alignimg_gpu-2.2.0.tar.gz
validation-results/performance/stage-0/baseline-2.0.0/adaptive.npz
validation-results/performance/stage-0/baseline-2.0.0/global.npz
validation-results/performance/stage-0/baseline-2.0.0/manifest.json
validation-results/performance/stage-0/baseline-2.0.0/source.tar.gz
validation-results/fast-hard/stage-0/frozen-inputs/synthetic.inputs.npz
validation-results/fast-hard/stage-0/frozen-inputs/homogeneous.inputs.npz
validation-results/fast-hard/stage-0/frozen-inputs/mra1000.inputs.npz
```

本階段沒有修改 `src/alignimg` 或 `packages/alignimg-gpu/src`。如果只是診斷既有 overlay
目錄，可先把不屬於目前 2.2 tree 的舊 mixed-precision 測試移到可復原的隔離目錄；
正式 acceptance 則以上述完整替換為準：

```bash
mkdir -p validation-results/fast-hard/stage-0/server-stale-files/tests
if [ -f tests/test_mixed_precision.py ]; then
  mv tests/test_mixed_precision.py \
    validation-results/fast-hard/stage-0/server-stale-files/tests/
fi
```

確認關鍵檔案 SHA-256。下列值全部相同才可繼續；版本字串相同但 source hash 不同仍視為
混合安裝：

```bash
sha256sum \
  tools/performance_fixtures.py \
  tests/test_calculation_reuse.py \
  tests/test_execution_profiling.py \
  validation-results/performance/stage-0/baseline-2.0.0/adaptive.npz \
  validation-results/performance/stage-0/baseline-2.0.0/global.npz \
  validation-results/performance/stage-0/baseline-2.0.0/manifest.json \
  validation-results/performance/stage-0/baseline-2.0.0/source.tar.gz \
  packages/alignimg-gpu/dist/alignimg_gpu-2.2.0.tar.gz
```

預期依序為：

```text
f655ac4879317a1a3e64645d3cd90229e3017a5e6d85d589bb7b4d558c23452a
a218fe1a89060b93a9d56449a49b11318fae70adb71ff6664d6a81e4db71ee48
f7fb0265500ba32042c2ba57f3490a936142d8d62551bb66ca8220c829cefd71
380085981d01e4362e44a8339d049f6f720b7012a94ae21e15be50a8b8044f0e
8af856937ce59b45c339056c0bb78b29ff11a7e83c6a481182f9620c5f539708
f4e6b3e1574933a0cdc2c2e1b1b129ba1e649f80050d7fcd79bf1dada8e36f9a
4e37ec8c3fc22fe653841db9ccf151997332a9ab886046b78e89cce7dda9b212
a9a6c42627ec0f4ea0cc28da27bbb249f29610bbc52f1ccaa1792018c192188d
```

再重新安裝 core 與 native GPU package：

```bash
python -m pip install \
  --force-reinstall \
  --no-deps \
  -e .

CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" \
python -m pip install -v \
  --force-reinstall \
  --no-build-isolation \
  --no-deps \
  packages/alignimg-gpu/dist/alignimg_gpu-2.2.0.tar.gz
```

安裝後另驗證實際 import 的 Python GPU backend，不只檢查版本 stamp：

```bash
python - <<'PY'
from hashlib import sha256
from pathlib import Path
import alignimg_gpu.backend as backend
import alignimg_gpu._workspace as workspace

expected = {
    Path(backend.__file__): "2087c5f8e7261f4b30b4776e239cd9bee6a5a8034a5f33d0c72dd0fb71338214",
    Path(workspace.__file__): "b0f69b9c548b08d69f1876467fcb5af5c4855acdecbd793e682b15268402d296",
}
for path, wanted in expected.items():
    actual = sha256(path.read_bytes()).hexdigest()
    print(actual, path)
    assert actual == wanted, f"stale installed module: {path}"
PY
```

驗證正式版本 identity 與既有 preset 未變：

```bash
python - <<'PY'
from importlib.metadata import version
import alignimg
import alignimg_gpu
from alignimg_gpu.backend import _native_module

native = _native_module()
values = {
    "AlignImg module": alignimg.__version__,
    "AlignImg metadata": version("alignimg"),
    "AlignImg GPU": alignimg_gpu.__version__,
    "Native CUDA": getattr(native, "__version__", None),
    "Global strategy": alignimg.AlignmentConfig.preset("global_balanced").search_strategy,
    "Refine strategy": alignimg.AlignmentConfig.preset("refine").search_strategy,
}
for name, value in values.items():
    print(f"{name}: {value}")
assert values["AlignImg module"] == "2.2.0"
assert values["AlignImg metadata"] == "2.2.0"
assert values["AlignImg GPU"] == "2.2.0"
assert values["Native CUDA"] == "2.2.0"
assert values["Global strategy"] == "proposal"
assert values["Refine strategy"] == "quadratic_refine"
PY
```

先確認 GPU 沒有其他工作，再執行 quick conformance：

```bash
nvidia-smi

python tools/server_validation.py \
  --suite quick --backend cuda --batch-size 512 \
  --output validation-results/fast-hard/stage-0/dev0-cuda-quick.json

python tools/server_validation.py \
  --suite quick --backend cupy --batch-size 512 \
  --output validation-results/fast-hard/stage-0/dev0-cupy-quick.json
```

接著執行 Stage 0 frozen workloads：

```bash
python tools/fast_hard_validation.py \
  --suite synthetic --backend cuda \
  --batch-size 512 --memory-fraction 0.8 \
  --deterministic-repeats 2 --profile-execution \
  --inputs validation-results/fast-hard/stage-0/frozen-inputs/synthetic.inputs.npz \
  --output validation-results/fast-hard/stage-0/dev0-cuda-synthetic.json

python tools/fast_hard_validation.py \
  --suite synthetic --backend cupy \
  --batch-size 512 --memory-fraction 0.8 \
  --deterministic-repeats 2 --profile-execution \
  --inputs validation-results/fast-hard/stage-0/frozen-inputs/synthetic.inputs.npz \
  --output validation-results/fast-hard/stage-0/dev0-cupy-synthetic.json

python tools/fast_hard_validation.py \
  --suite homogeneous --backend cupy --only fixed_k3 \
  --batch-size 512 --memory-fraction 0.8 \
  --deterministic-repeats 2 --profile-execution \
  --inputs validation-results/fast-hard/stage-0/frozen-inputs/homogeneous.inputs.npz \
  --output validation-results/fast-hard/stage-0/dev0-cupy-homogeneous-fixed-k3.json

python tools/fast_hard_validation.py \
  --suite homogeneous --backend cuda \
  --batch-size 512 --memory-fraction 0.8 \
  --deterministic-repeats 2 --profile-execution \
  --inputs validation-results/fast-hard/stage-0/frozen-inputs/homogeneous.inputs.npz \
  --output validation-results/fast-hard/stage-0/dev0-cuda-homogeneous.json

python tools/fast_hard_validation.py \
  --suite mra1000 --backend cuda \
  --batch-size 512 --memory-fraction 0.8 \
  --deterministic-repeats 2 --profile-execution \
  --inputs validation-results/fast-hard/stage-0/frozen-inputs/mra1000.inputs.npz \
  --output validation-results/fast-hard/stage-0/dev0-cuda-mra1000.json
```

第一輪只需帶回上述七份 JSON。大型 result NPZ 留在 server；只有 JSON 顯示 hash、
數值、backend identity 或 parity 異常時，再帶回指定 result artifact。

## FTP 上傳後的 server 執行順序

在 server 專案根目錄及既有開發環境中執行。先確認版本與完整測試：

```bash
python - <<'PY'
import alignimg
import alignimg_gpu
from alignimg_gpu.backend import _native_module

native = _native_module()
print("AlignImg:", alignimg.__version__, alignimg.__file__)
print("AlignImg GPU:", alignimg_gpu.__version__, alignimg_gpu.__file__)
print("Native CUDA:", None if native is None else native.__version__)
assert alignimg.__version__ == "2.2.0"
assert alignimg_gpu.__version__ == "2.2.0"
assert native is not None and native.__version__ == "2.2.0"
PY

python -m pytest -q
```

接著執行 quick conformance：

```bash
python tools/server_validation.py \
  --suite quick --backend cuda --batch-size 512 \
  --output validation-results/fast-hard/stage-0/dev0-cuda-quick.json

python tools/server_validation.py \
  --suite quick --backend cupy --batch-size 512 \
  --output validation-results/fast-hard/stage-0/dev0-cupy-quick.json
```

再以同一批 frozen inputs 執行 Stage 0 workloads：

```bash
python tools/fast_hard_validation.py \
  --suite synthetic --backend cuda \
  --batch-size 512 --memory-fraction 0.8 \
  --deterministic-repeats 2 --profile-execution \
  --inputs validation-results/fast-hard/stage-0/frozen-inputs/synthetic.inputs.npz \
  --output validation-results/fast-hard/stage-0/dev0-cuda-synthetic.json

python tools/fast_hard_validation.py \
  --suite synthetic --backend cupy \
  --batch-size 512 --memory-fraction 0.8 \
  --deterministic-repeats 2 --profile-execution \
  --inputs validation-results/fast-hard/stage-0/frozen-inputs/synthetic.inputs.npz \
  --output validation-results/fast-hard/stage-0/dev0-cupy-synthetic.json

python tools/fast_hard_validation.py \
  --suite homogeneous --backend cuda \
  --batch-size 512 --memory-fraction 0.8 \
  --deterministic-repeats 2 --profile-execution \
  --inputs validation-results/fast-hard/stage-0/frozen-inputs/homogeneous.inputs.npz \
  --output validation-results/fast-hard/stage-0/dev0-cuda-homogeneous.json

python tools/fast_hard_validation.py \
  --suite homogeneous --backend cupy --only fixed_k3 \
  --batch-size 512 --memory-fraction 0.8 \
  --deterministic-repeats 2 --profile-execution \
  --inputs validation-results/fast-hard/stage-0/frozen-inputs/homogeneous.inputs.npz \
  --output validation-results/fast-hard/stage-0/dev0-cupy-homogeneous-fixed-k3.json

python tools/fast_hard_validation.py \
  --suite mra1000 --backend cuda \
  --batch-size 512 --memory-fraction 0.8 \
  --deterministic-repeats 2 --profile-execution \
  --inputs validation-results/fast-hard/stage-0/frozen-inputs/mra1000.inputs.npz \
  --output validation-results/fast-hard/stage-0/dev0-cuda-mra1000.json
```

執行完成後只需先抓回上述七份 JSON。大型 `*.result.npz` 留在 server；若 JSON
顯示 hash、數值或 VRAM 異常，再取回對應 result artifact。
