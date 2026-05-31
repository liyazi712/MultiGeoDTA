#!/usr/bin/env bash
# Install dependencies + smoke train (1 epoch, 8 samples per split)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MULTIGEODTA_ROOT="$ROOT"
cd "$ROOT"
LOG="$ROOT/outputs/smoke_install.log"
mkdir -p outputs
exec > >(tee "$LOG") 2>&1

echo "=== MultiGeo-DTA: install + smoke train ==="
echo "Log: $LOG"

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

export MULTIGEODTA_DATA_DIR="${MULTIGEODTA_DATA_DIR:-$ROOT/data}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
# HF: default official hub; set HF_ENDPOINT yourself if you use a resolvable mirror.
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT
fi
TASK_DIR="$MULTIGEODTA_DATA_DIR/pdbbind_v2016"
mkdir -p "$TASK_DIR"

echo "Checking / downloading pdbbind_v2016 assets..."
for f in last_train_2016.csv last_valid_2016.csv last_test_2016.csv \
  pocket_structures_train.pkl.gz pocket_structures_valid.pkl.gz pocket_structures_test.pkl.gz \
  mol_structures_train.pkl.gz mol_structures_valid.pkl.gz mol_structures_test.pkl.gz; do
  if [[ ! -f "$TASK_DIR/$f" ]]; then
    echo "  -> $f"
    huggingface-cli download laddymo/MultiGeoDTA "pdbbind_v2016/$f" \
      --repo-type dataset --local-dir "$MULTIGEODTA_DATA_DIR"
  fi
done

echo "=== Smoke train: 1 epoch, 8 samples, batch_size=4 ==="
python -m multigeodta train \
  --task pdbbind_v2016 \
  --smoke \
  --output_dir outputs/smoke_test \
  --n_epochs 1 \
  --n_ensembles 1 \
  --batch_size 4 \
  --device 0 \
  --eval_freq 1 \
  --patience 1

echo "=== SUCCESS ==="
ls -la outputs/smoke_test/
