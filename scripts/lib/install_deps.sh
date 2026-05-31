#!/usr/bin/env bash
# Shared dependency installer for MultiGeo-DTA
# Stack: Python 3.8 | PyTorch 2.1 | CUDA 11.8 | DGL | PyG | mamba-ssm (wheels)
set -euo pipefail

ROOT="${MULTIGEODTA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WHEEL_DIR="${ROOT}/.wheels"
mkdir -p "$WHEEL_DIR"

# Use official PyPI for build tools when mirror is flaky
PYPI="${PIP_INDEX_URL:-https://pypi.org/simple}"
TORCH_INDEX="https://download.pytorch.org/whl/cu118"
PYG_FIND="https://data.pyg.org/whl/torch-2.1.0+cu118.html"
DGL_FIND="https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html"

CAUSAL_WHEEL_URL="${CAUSAL_WHEEL_URL:-https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl}"
MAMBA_WHEEL_URL="${MAMBA_WHEEL_URL:-https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl}"
# When github.com is unreachable, set e.g. GITHUB_RELEASE_MIRROR=https://ghproxy.net/
GITHUB_RELEASE_MIRROR="${GITHUB_RELEASE_MIRROR:-}"

ensure_cuda_runtime() {
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  if [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/lib/libcusparse.so.11" ]]; then
    echo "CUDA runtime libs OK: ${CONDA_PREFIX}/lib/libcusparse.so.11"
    return 0
  fi
  if [[ "${SKIP_CUDATOOLKIT:-0}" == "1" ]]; then
    echo "WARN: libcusparse.so.11 not found (SKIP_CUDATOOLKIT=1)"
    return 0
  fi
  echo "Installing cudatoolkit=11.8 (DGL needs libcusparse.so.11) ..."
  if command -v conda &>/dev/null && [[ -n "${CONDA_PREFIX:-}" ]]; then
    conda install -y cudatoolkit=11.8
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  else
    echo "ERROR: libcusparse.so.11 missing; activate a conda env or set LD_LIBRARY_PATH"
    return 1
  fi
  [[ -f "${CONDA_PREFIX}/lib/libcusparse.so.11" ]]
}

pip_retry() {
  local max=3 attempt=1
  until pip "$@"; do
    attempt=$((attempt + 1))
    if (( attempt > max )); then
      echo "ERROR: pip failed after ${max} attempts: pip $*"
      return 1
    fi
    echo "Retry pip ($attempt/$max)..."
    sleep 2
  done
}

download_wheel() {
  local url="$1" dest="$2"
  if [[ -f "$dest" ]] && python - <<PY
import zipfile, sys
try:
    zipfile.ZipFile("${dest}").testzip()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
  then
    echo "Using cached wheel: $dest"
    return 0
  fi

  local urls=("$url")
  if [[ -n "$GITHUB_RELEASE_MIRROR" && "$url" == https://github.com/* ]]; then
    urls+=("${GITHUB_RELEASE_MIRROR%/}/${url}")
  fi

  local attempt=1 max=5
  while (( attempt <= max )); do
    local fetch_url="${urls[$(( (attempt - 1) % ${#urls[@]} ))]}"
    if [[ -f "$dest" ]]; then
      echo "Resuming download (attempt ${attempt}/${max}): $(basename "$dest") ..."
    else
      echo "Downloading (attempt ${attempt}/${max}): $(basename "$dest") ..."
    fi
    if command -v curl &>/dev/null; then
      curl -L --connect-timeout 30 --max-time 7200 \
        --retry 3 --retry-delay 5 -C - --fail -o "$dest" "$fetch_url" || true
    elif command -v wget &>/dev/null; then
      wget -c -O "$dest" "$fetch_url" || true
    else
      python - <<PY
import urllib.request
urllib.request.urlretrieve("${fetch_url}", "${dest}")
PY
    fi
    if python - <<PY
import zipfile
zipfile.ZipFile("${dest}").testzip()
PY
    then
      echo "Verified: $dest"
      return 0
    fi
    echo "WARN: wheel incomplete or corrupt, will retry..."
    attempt=$((attempt + 1))
    sleep 3
  done
  echo "ERROR: failed to download valid wheel: $(basename "$dest")"
  echo "  Place the file manually at: $dest"
  echo "  Or run: pip install $url"
  echo "  If github.com is blocked, try: export GITHUB_RELEASE_MIRROR=https://ghproxy.net/"
  return 1
}

install_torch_stack() {
  echo "=== [1/6] PyTorch 2.1 + CUDA 11.8 ==="
  if python -c "import torch; assert torch.__version__.startswith('2.1')" 2>/dev/null; then
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  else
    pip_retry install torch==2.1.0 --index-url "$TORCH_INDEX"
  fi
  # cudatoolkit provides libcusparse.so.11 required by DGL wheels
  ensure_cuda_runtime
}

install_python_deps() {
  echo "=== [2/6] Base Python packages ==="
  pip_retry install -i "$PYPI" "setuptools>=61" wheel pip
  pip_retry install -r "$ROOT/requirements/base.txt"
  pip_retry install rdkit torch-geometric
}

install_pyg_extensions() {
  echo "=== [3/6] PyTorch Geometric extensions (cu118 / torch2.1) ==="
  pip_retry install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f "$PYG_FIND"
}

install_dgl() {
  echo "=== [4/6] DGL (torch 2.1 / cu118) ==="
  ensure_cuda_runtime
  pip_retry install dgl -f "$DGL_FIND"
}

install_mamba_wheels() {
  echo "=== [5/6] mamba-ssm + causal-conv1d (prebuilt wheels) ==="
  # pip requires PEP 427 wheel names (version-pyabi-platform); keep URL basename
  local causal="$WHEEL_DIR/$(basename "$CAUSAL_WHEEL_URL")"
  local mamba="$WHEEL_DIR/$(basename "$MAMBA_WHEEL_URL")"
  # migrate legacy short names from earlier script versions
  [[ -f "$WHEEL_DIR/causal_conv1d_cu118_torch21_cp38.whl" && ! -f "$causal" ]] && \
    mv "$WHEEL_DIR/causal_conv1d_cu118_torch21_cp38.whl" "$causal"
  [[ -f "$WHEEL_DIR/mamba_ssm_cu118_torch21_cp38.whl" && ! -f "$mamba" ]] && \
    mv "$WHEEL_DIR/mamba_ssm_cu118_torch21_cp38.whl" "$mamba"
  download_wheel "$CAUSAL_WHEEL_URL" "$causal"
  download_wheel "$MAMBA_WHEEL_URL" "$mamba"
  pip_retry install "$causal" "$mamba"
  python -c "from mamba_ssm import Mamba2; print('mamba_ssm OK')"
}

install_package() {
  echo "=== [6/6] Install multigeodta package ==="
  cd "$ROOT"
  # --no-build-isolation avoids re-fetching setuptools in isolated env (mirror DNS issues)
  if pip_retry install -e . --no-build-isolation; then
    echo "Installed editable: multigeodta"
  else
    echo "WARN: pip install -e . failed; falling back to PYTHONPATH=$ROOT/src"
    export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
    grep -q 'PYTHONPATH.*MultiGeoDTA/src' "$HOME/.bashrc" 2>/dev/null || true
  fi
}

verify_stack() {
  echo "=== Verify imports ==="
  ensure_cuda_runtime
  python - <<'PY'
import torch
import torch_geometric
import dgl
from mamba_ssm import Mamba2
import multigeodta
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("multigeodta", multigeodta.__version__)
PY
}

install_torch_stack
install_python_deps
install_pyg_extensions
install_dgl
install_mamba_wheels
install_package
verify_stack
echo "=== All dependencies installed ==="
