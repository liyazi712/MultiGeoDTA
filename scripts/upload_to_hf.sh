#!/usr/bin/env bash
# Replace all files in laddymo/MultiGeoDTA (dataset) with local data/ and outputs/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ID="${HF_REPO_ID:-laddymo/MultiGeoDTA}"
REPO_TYPE="${HF_REPO_TYPE:-dataset}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: 请设置 HF_TOKEN（需 Write 权限）。" >&2
  echo "  export HF_TOKEN=hf_..." >&2
  echo "  在 https://huggingface.co/settings/tokens 创建 token。" >&2
  exit 1
fi

# Official hub often times out on this cluster; mirror is reachable.
if [[ -z "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT="https://hf-mirror.com"
fi
export HF_REPO_ID="$REPO_ID"
echo "Using HF_ENDPOINT=${HF_ENDPOINT}"
echo "Target repo: ${REPO_ID} (type=${REPO_TYPE})"

pip install -q -U huggingface_hub

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

# Paths are relative to data/ (src_root = $ROOT/data).
ZINC_RAW_EXCLUDE=(
  "zinc/code/sdf"
  "zinc/code/smile"
  "zinc/code/structure"
  "zinc/code/zinc_sdf"
)

should_exclude_zinc_raw() {
  local rel="$1"
  if [[ "${INCLUDE_RAW_ZINC:-0}" == "1" ]]; then
    return 1
  fi
  for prefix in "${ZINC_RAW_EXCLUDE[@]}"; do
    if [[ "$rel" == "$prefix" || "$rel" == "$prefix"/* ]]; then
      return 0
    fi
  done
  return 1
}

stage_tree() {
  local src_root="$1"
  local dst_root="$2"
  local filter_zinc="${3:-0}"

  mkdir -p "$dst_root"
  while IFS= read -r -d '' file; do
    rel="${file#"$src_root"/}"
    base="$(basename "$file")"
    if [[ "$base" == ".DS_Store" || "$base" == ".gitkeep" ]]; then
      continue
    fi
    if [[ "$rel" == *"/__pycache__/"* || "$rel" == "__pycache__/"* ]]; then
      continue
    fi
    if [[ "$filter_zinc" == "1" ]] && should_exclude_zinc_raw "$rel"; then
      continue
    fi
    mkdir -p "$dst_root/$(dirname "$rel")"
    cp -a "$file" "$dst_root/$rel"
  done < <(find "$src_root" -type f -print0)
}

echo "Staging data/ and outputs/ ..."
stage_tree "$ROOT/data" "$STAGING/data" 1
stage_tree "$ROOT/outputs" "$STAGING/outputs" 0

file_count="$(find "$STAGING" -type f | wc -l | tr -d ' ')"
total_bytes="$(find "$STAGING" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')"
echo "Staged ${file_count} files ($(awk -v b="$total_bytes" 'BEGIN {printf "%.2f GiB", b/1024/1024/1024}'))"
if [[ "${INCLUDE_RAW_ZINC:-0}" != "1" ]]; then
  echo "Excluded raw ZINC intermediates under data/zinc/code/{sdf,smile,structure,zinc_sdf}."
  echo "Set INCLUDE_RAW_ZINC=1 to include them (~594k extra files)."
fi

echo "Deleting all existing files in ${REPO_ID} ..."
HF_ENDPOINT="$HF_ENDPOINT" HF_TOKEN="$HF_TOKEN" HF_REPO_ID="$REPO_ID" HF_REPO_TYPE="$REPO_TYPE" python3 - <<'PY'
import os
from huggingface_hub import HfApi, list_repo_files

repo_id = os.environ["HF_REPO_ID"]
repo_type = os.environ.get("HF_REPO_TYPE", "dataset")
token = os.environ["HF_TOKEN"]
endpoint = os.environ.get("HF_ENDPOINT")
api = HfApi(endpoint=endpoint, token=token)
files = list_repo_files(repo_id, repo_type=repo_type, token=token)
if files:
    api.delete_files(
        repo_id,
        files,
        repo_type=repo_type,
        commit_message="Remove outdated dataset and model files",
        token=token,
    )
print(f"Deleted {len(files)} remote files.")
PY

echo "Uploading staged content (resumable) ..."
STAGING="$STAGING" HF_ENDPOINT="$HF_ENDPOINT" HF_TOKEN="$HF_TOKEN" HF_REPO_ID="$REPO_ID" HF_REPO_TYPE="$REPO_TYPE" python3 - <<'PY'
import os
from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO_ID"]
repo_type = os.environ.get("HF_REPO_TYPE", "dataset")
token = os.environ["HF_TOKEN"]
endpoint = os.environ.get("HF_ENDPOINT")
staging = os.environ["STAGING"]

api = HfApi(endpoint=endpoint, token=token)
api.upload_large_folder(
    repo_id=repo_id,
    folder_path=staging,
    repo_type=repo_type,
    num_workers=4,
)
print("Upload finished.")
PY

echo "Done. Verify: hf download ${REPO_ID} --repo-type ${REPO_TYPE} --local-dir /tmp/hf_check"
