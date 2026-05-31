#!/usr/bin/env bash
# Download preprocessed MultiGeo-DTA datasets from Hugging Face
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${MULTIGEODTA_DATA_DIR:-$ROOT/data}"
mkdir -p "$DATA_DIR"

pip install -q -U huggingface_hub

# Default: official hub (https://huggingface.co). Do not default to a mirror — many
# clusters cannot resolve hf-mirror.com and would silently reuse stale local_dir.
# In mainland China you may try: export HF_ENDPOINT=https://hf-mirror.com
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT
  echo "Using HF_ENDPOINT=$HF_ENDPOINT"
else
  unset HF_ENDPOINT 2>/dev/null || true
  echo "Using default Hugging Face hub (set HF_ENDPOINT to use a mirror)."
fi

LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT

# huggingface-cli is deprecated but still widely installed; prefer hf when present.
if command -v hf &>/dev/null; then
  DL=(hf download laddymo/MultiGeoDTA --repo-type dataset --local-dir "$DATA_DIR")
else
  DL=(huggingface-cli download laddymo/MultiGeoDTA --repo-type dataset --local-dir "$DATA_DIR" --local-dir-use-symlinks False)
fi

if "${DL[@]}" 2>&1 | tee "$LOG"; then
  :
else
  echo "ERROR: dataset download command failed." >&2
  exit 1
fi

if grep -qE 'remote repo cannot be accessed|Returning existing local_dir.*cannot be accessed' "$LOG"; then
  echo "" >&2
  echo "ERROR: 未能连接 Hugging Face 数据集仓库，未确认已同步远程文件。" >&2
  echo "  - 检查 DNS / 代理；或尝试官方节点：unset HF_ENDPOINT 后重试。" >&2
  echo "  - 若需镜像：export HF_ENDPOINT=<可解析的镜像地址> 后再运行本脚本。" >&2
  exit 1
fi

echo "Data saved to $DATA_DIR"
echo "export MULTIGEODTA_DATA_DIR=$DATA_DIR"
