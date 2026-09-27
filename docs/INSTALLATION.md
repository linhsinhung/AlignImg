# Installation and upgrades

These instructions describe the AlignImg 2.3 source tree: core/GPU/native
`2.3.0`, optional Workbench `0.7.0`. They do not assume a published PyPI release
or an existing version tag. See the [release status](RELEASE_FREEZE_2_3.md) for
validation still required before formal acceptance.

## Requirements

- Python 3.10+. The core installs NumPy, SciPy, and OpenCV headless.
- MRC/MRCS I/O: optional `mrcfile`, installed by the core `[io]` extra.
- Both GPU backends: Linux x86-64, one NVIDIA GPU, a compatible driver, and
  CuPy 14+. Use CPU on other platforms.
- Native CUDA: CUDA Toolkit 12+ with `nvcc`, a C++17 host compiler, and
  CMake 3.24+. Normal pip builds manage scikit-build-core and pybind11.
- Workbench: a graphical desktop, PyQt6, pyqtgraph, NumPy, and `mrcfile`;
  pip installs its Python dependencies.

Host RAM must hold the input array and working copies. GPU batching bounds
VRAM usage; it does not provide disk streaming or eliminate host RAM needs.

## 1. Install the core

From a clone or extracted source checkout, run commands at the repository root.
The optional-package commands assume the full repository (GitHub archive or
checkout), not the core-only Python sdist.
For a new POSIX environment:

```bash
git clone https://github.com/linhsinhung/AlignImg.git
cd AlignImg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[io]"
```

If an existing Conda/virtual environment is already prepared, activate it and
skip `.venv` creation. Install with `python -m pip install .` if you only need
the NumPy-array API. The core installs neither GPU nor GUI dependencies.

Check the core installation before adding optional packages:

```bash
python - <<'PY'
from importlib.metadata import version
import alignimg as ai

print('Core module:', ai.__version__)
print('Core metadata:', version('alignimg'))
print('Backends:', ai.available_alignment_backends())
assert ai.__version__ == version('alignimg') == '2.3.0'
PY
```

## 2. Optional GPU support

Install the core first. The following commands build from the GPU package's
source directory; they do not download a prebuilt native CUDA wheel.

### Native CUDA

For CUDA 12 with `nvcc` discoverable:

```bash
python -m pip install -v "./packages/alignimg-gpu[cuda12]"
```

For CUDA 13, choose `[cuda13]` instead. Select the extra for the toolkit/runtime
you intend to use, not just the maximum CUDA version reported by `nvidia-smi`.
Install only one CuPy distribution in an environment.

The default native build targets the build host's GPU. To set the compiler and
architecture explicitly, for example on an RTX 3090:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" \
python -m pip install -v "./packages/alignimg-gpu[cuda12]"
```

Adjust these values for your host. Without a CUDA compiler, the package builds
only the CuPy fallback; successful pip installation alone does not establish
that the native backend is available.

### CuPy-only fallback

To explicitly disable native compilation:

```bash
CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=NOTFOUND" \
python -m pip install "./packages/alignimg-gpu[cuda12]"
```

This still requires the NVIDIA driver and working CuPy runtime. The package
build also configures CMake and a host C++ compiler.

### GPU identity and smoke check

For an intended **native** installation:

```bash
python - <<'PY'
from importlib.metadata import version
import alignimg as ai
import alignimg_gpu
from alignimg_gpu.backend import _native_module

native = _native_module()
assert ai.__version__ == version('alignimg') == '2.3.0'
assert alignimg_gpu.__version__ == version('alignimg-gpu') == '2.3.0'
assert getattr(native, '__version__', None) == '2.3.0'
assert ai.available_alignment_backends()['cuda']['available']
print('Core/GPU/native CUDA: 2.3.0')
print('Native binary:', native.__file__)
PY

python tools/server_validation.py \
  --suite quick --backend cuda --batch-size 512 \
  --output validation-results/install/cuda-quick.json
```

For a deliberately CuPy-only installation, check core/GPU versions and require
`available_alignment_backends()['cupy']['available']`; a native stamp is not
expected. Run the smoke command with `--backend cupy` and a separate output
path. Use a new report filename when repeating acceptance runs.

Explicit `backend="cuda"` or `"cupy"` fails if unavailable. `"auto"` may select
CUDA → CuPy → CPU, so do not use it to prove a specific installation. Omitting
`backend` uses CPU. See the [GPU guide](../packages/alignimg-gpu/README.md) for
memory and execution behavior.

## 3. Optional Workbench

From the repository root, after installing the core:

```bash
python -m pip install ./packages/alignimg-gui
python - <<'PY'
from importlib.metadata import version
import alignimg_gui

assert alignimg_gui.__version__ == version('alignimg-gui') == '0.7.0'
print('Workbench: 0.7.0')
PY
alignimg-gui
```

`python -m alignimg_gui` is an alternative entry point. The GUI works without
the GPU package, using CPU. See the [Workbench guide](../packages/alignimg-gui/README.md)
for workflows and artifacts.

## Development and tests

Development installs are explicitly editable, unlike the regular source
installs above:

```bash
python -m pip install -e ".[dev]"
# Only when developing the GUI:
python -m pip install -e ./packages/alignimg-gui
python -m pytest -q
```

The default suite is CPU-only and covers core alignment plus backend emulation.
Real-GPU checks use `python -m pytest -q -m gpu`. See the
[test guide](../tests/README.zh-TW.md) for current-product, GUI, historical, and
complete suites. An editable Python install does not automatically rebuild an
already installed native CUDA binary.

## Upgrade an already prepared server in place

Keep the existing environment and working directory. Upload/extract a verified
source delivery; preserve data and historical validation results. Compare the
delivery's archive checksum and source manifest **before** installing or
running calculations. Do not apply an old delivery's hash to a newer checkout.

FTP overlay does not remove obsolete files. When changing the test layout,
back up and replace the old `tests` directory rather than merging old and new
test paths. No additional Python environment is required.

When runtime dependencies are already installed, reinstall the updated code:

```bash
python -m pip install --force-reinstall --no-deps -e .

CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" \
python -m pip install -v --force-reinstall --no-deps \
  ./packages/alignimg-gpu

# If Workbench is installed:
python -m pip install --force-reinstall --no-deps ./packages/alignimg-gui
```

Adjust CUDA settings for the host. `--no-deps` is for a prepared environment;
it is not a fresh-install recipe. Normal build isolation remains enabled. Use
`--no-build-isolation` only if the environment already contains the required
build tools. For a sealed delivery, use its specified GPU sdist/GUI wheel paths
instead of assuming that archives exist in a fresh Git clone.

Re-run the version checks and explicit-backend quick report after rebuilding.
When only docs/test organization changed, existing frozen scientific results
need not be rerun; formal acceptance must still identify the exact delivered
source/packages/native binary. Stop on checksum, build, or assertion failures.

## Repository hygiene before publishing

Commit source, package metadata, licenses/notices, documentation, and tests.
Generated distributions, environments, caches, data, coverage and validation
outputs are local artifacts excluded by `.gitignore`; ignoring a path does not
remove files already tracked by Git. Review `git status --short` and the staged
diff before publishing, including test moves and new native source files.

Keep release artifacts and their checksums separate from the source tree. Do
not rewrite sealed artifacts or historical scientific gates after a test/doc
cleanup. Formal 2.3 acceptance remains subject to the
[release record](RELEASE_FREEZE_2_3.md), not merely a successful documentation
update or a push to `main`.
