#!/usr/bin/env bash
# One-shot environment setup: conda env + full CUDA 11.8 stack
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MULTIGEODTA_ROOT="$ROOT"
cd "$ROOT"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | grep -qE '^multigeodta '; then
  echo "Creating conda env multigeodta (python=3.8 + cudatoolkit)..."
  conda create -n multigeodta python=3.8 cudatoolkit=11.8 -y
fi
conda activate multigeodta
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# shellcheck source=/dev/null
source "$ROOT/scripts/lib/install_deps.sh"

echo ""
echo "Done. Next steps:"
echo "  conda activate multigeodta"
echo "  export MULTIGEODTA_DATA_DIR=$ROOT/data"
echo "  bash scripts/download_data.sh   # if data not present"
