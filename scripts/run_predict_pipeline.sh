#!/usr/bin/env bash
# End-to-end drug-target affinity prediction for unknown structures:
#   1. ESMFold2 (esmfold2 env)  → full protein PDB
#   2. DoGSite3 REST API        → binding pocket PDB
#   3. MultiGeo-DTA (multigeodta env) → ensemble affinity prediction
#
# Usage (from repo root):
#   bash scripts/run_predict_pipeline.sh -p "MKT..." -s "CCO"
#
# Environment variables:
#   ESMFOLD_ENV, MULTIGEODTA_ENV  conda env names (defaults: esmfold2, multigeodta)
#   DEVICE                        CUDA device id (default: 0)
#   FORCE_REPREDICT               set to 1 to ignore structure cache
#
# Extra flags after "--" are forwarded to predict_affinity.py.
# Results are saved under outputs/user_request_results/<YYYYMMDD_HHMMSS>/.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIPE="$ROOT/scripts/predict_affinity_from_sequence"
ESMFOLD_ENV="${ESMFOLD_ENV:-esmfold2}"
MULTIGEODTA_ENV="${MULTIGEODTA_ENV:-multigeodta}"
DEVICE="${DEVICE:-0}"
FORCE_REPREDICT="${FORCE_REPREDICT:-0}"

PROTEIN_SEQUENCE=""
SMILES=""
REQUEST_RESULTS_DIR=""
EXTRA_AFFINITY_ARGS=()

log() { echo "[predict_pipeline] $*"; }

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_predict_pipeline.sh -p SEQ -s SMILES [options] [-- extra predict_affinity flags]

Options:
  -p, --protein_sequence     Target amino-acid sequence (required)
  -s, --smiles               Ligand SMILES (required)
  --request_results_dir      Output dir for this request (default: outputs/user_request_results/<time>)
  --structure_cache_dir      Alias for --request_results_dir (deprecated)
  --force_repredict          Re-run structure prediction even if cached
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--protein_sequence) PROTEIN_SEQUENCE="$2"; shift 2 ;;
    -s|--smiles)           SMILES="$2"; shift 2 ;;
    --request_results_dir|--structure_cache_dir) REQUEST_RESULTS_DIR="$2"; shift 2 ;;
    --force_repredict)     FORCE_REPREDICT=1; shift ;;
    -h|--help)             usage; exit 0 ;;
    --)                    shift; EXTRA_AFFINITY_ARGS=("$@"); break ;;
    *)                     echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

[[ -n "$PROTEIN_SEQUENCE" && -n "$SMILES" ]] || {
  echo "ERROR: --protein_sequence and --smiles are required." >&2
  usage >&2
  exit 1
}

if [[ -z "$REQUEST_RESULTS_DIR" ]]; then
  REQUEST_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  REQUEST_RESULTS_DIR="$ROOT/outputs/user_request_results/$REQUEST_TIMESTAMP"
  suffix=1
  while [[ -e "$REQUEST_RESULTS_DIR" ]]; do
    REQUEST_RESULTS_DIR="$ROOT/outputs/user_request_results/${REQUEST_TIMESTAMP}_${suffix}"
    suffix=$((suffix + 1))
  done
fi
mkdir -p "$REQUEST_RESULTS_DIR"
PIPELINE_LOG="$REQUEST_RESULTS_DIR/predict_pipeline.log"

log() {
  echo "[predict_pipeline] $*" | tee -a "$PIPELINE_LOG"
}

command -v conda >/dev/null || { echo "ERROR: conda not found in PATH." >&2; exit 1; }
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

for env in "$ESMFOLD_ENV" "$MULTIGEODTA_ENV"; do
  conda env list | awk '{print $1}' | grep -qx "$env" || {
    echo "ERROR: conda env '$env' not found." >&2
    exit 1
  }
done

has_output_flag=0
for arg in "${EXTRA_AFFINITY_ARGS[@]}"; do
  case "$arg" in
    -o|--output) has_output_flag=1; break ;;
  esac
done
if [[ "$has_output_flag" -eq 0 ]]; then
  EXTRA_AFFINITY_ARGS+=(-o "$REQUEST_RESULTS_DIR/prediction.json")
fi

STRUCT_ARGS=(
  --protein_sequence "$PROTEIN_SEQUENCE"
  --device "$DEVICE"
  --json
  --request_results_dir "$REQUEST_RESULTS_DIR"
  --output "$REQUEST_RESULTS_DIR/structure.json"
)
[[ "$FORCE_REPREDICT" == "1" ]] && STRUCT_ARGS+=(--force_repredict)

log "Results directory: $REQUEST_RESULTS_DIR"
log "Step 1/2: ESMFold2 + DoGSite3 (env=$ESMFOLD_ENV)"
{
  conda run -n "$ESMFOLD_ENV" python "$PIPE/predict_structure.py" "${STRUCT_ARGS[@]}"
} 2>&1 | tee -a "$PIPELINE_LOG"

STRUCT_JSON="$(<"$REQUEST_RESULTS_DIR/structure.json")"

read -r PROTEIN_PDB POCKET_PDB < <(
  python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['protein_pdb'], d['pocket_pdb'])" "$STRUCT_JSON"
)
log "  protein_pdb=$PROTEIN_PDB"
log "  pocket_pdb=$POCKET_PDB"

log "Step 2/2: MultiGeo-DTA affinity prediction (env=$MULTIGEODTA_ENV)"
{
  conda run -n "$MULTIGEODTA_ENV" python "$PIPE/predict_affinity.py" \
    --protein_sequence "$PROTEIN_SEQUENCE" \
    --smiles "$SMILES" \
    --protein_pdb "$PROTEIN_PDB" \
    --pocket_pdb "$POCKET_PDB" \
    --device "$DEVICE" \
    "${EXTRA_AFFINITY_ARGS[@]}"
} 2>&1 | tee -a "$PIPELINE_LOG"

log "Done. All outputs saved under $REQUEST_RESULTS_DIR"
