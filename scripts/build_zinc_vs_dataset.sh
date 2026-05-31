#!/usr/bin/env bash
# Build the ZINC virtual-screening dataset from raw downloads to model-ready inputs.
#
# Produces under data/zinc/:
#   mol_structures.pkl.gz          ligand 3D graphs
#   pocket_structures.pkl.gz       target pocket graph (key: "target")
#   processed_zinc_<TARGET>.csv    compound list (zinc_id, SMILES)
#   target_snippet_<TARGET>.yaml   copy into configs/tasks/zinc_vs.yaml
#
# Usage (from repo root):
#   bash scripts/build_zinc_vs_dataset.sh
#
# Environment variables:
#   MULTIGEODTA_DATA_DIR   data root (default: <repo>/data)
#   TARGET_NAME            target subfolder name (default: CB1R)
#   PROTEIN_PDB            override full structure PDB path
#   POCKET_PDB             override DoGSite3 pocket PDB path
#   SKIP_DOWNLOAD=1        skip wget if sdf/ and smile/ already populated
#   SKIP_MOL_PKL=1         reuse existing mol_structures.pkl.gz (slow step)
#   UPDATE_ZINC_CONFIG=1   merge target snippet into configs/tasks/zinc_vs.yaml
#
# Prerequisites:
#   - conda env multigeodta (rdkit, biopython, pandas, pyyaml)
#   - Target PDB files under data/zinc/Target_information/<TARGET_NAME>/
#     (prepare with AlphaFold2 / PDB + DoGSite3; see data/zinc/README.md)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${MULTIGEODTA_DATA_DIR:-$ROOT/data}"
ZINC_DIR="$DATA_DIR/zinc"
CODE_DIR="$ZINC_DIR/code"

TARGET_NAME="${TARGET_NAME:-CB1R}"
TARGET_DIR="$ZINC_DIR/Target_information/$TARGET_NAME"
PROTEIN_PDB="${PROTEIN_PDB:-$TARGET_DIR/alphafold_protein.pdb}"
POCKET_PDB="${POCKET_PDB:-$TARGET_DIR/alphafold_DoGsite3_pocket.pdb}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_MOL_PKL="${SKIP_MOL_PKL:-0}"
SKIP_COMPOUNDS="${SKIP_COMPOUNDS:-0}"
UPDATE_ZINC_CONFIG="${UPDATE_ZINC_CONFIG:-0}"

log() { echo "[build_zinc_vs] $*"; }

if [[ ! -d "$CODE_DIR" ]]; then
  echo "ERROR: $CODE_DIR not found. Set MULTIGEODTA_DATA_DIR correctly." >&2
  exit 1
fi

for pdb in "$PROTEIN_PDB" "$POCKET_PDB"; do
  if [[ ! -f "$pdb" ]]; then
    echo "ERROR: Target structure missing: $pdb" >&2
    echo "  Place AlphaFold/PDB + DoGSite3 files under $TARGET_DIR" >&2
    echo "  See data/zinc/README.md (Step 0 — Prepare the protein target)." >&2
    exit 1
  fi
done

cd "$CODE_DIR"
mkdir -p sdf smile structure zinc_sdf

if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
  if [[ ! -f ZINC-downloader-3D-sdf.gz.uri || ! -f ZINC-downloader-3D-smi.uri ]]; then
    echo "ERROR: ZINC downloader URI lists missing in $CODE_DIR" >&2
    exit 1
  fi
  log "Downloading ZINC 3D SMILES and SDF archives (this may take a while)..."
  wget -i ZINC-downloader-3D-smi.uri   -P ./smile -nc
  wget -i ZINC-downloader-3D-sdf.gz.uri -P ./sdf   -nc
else
  log "SKIP_DOWNLOAD=1 — using existing ./smile and ./sdf"
fi

log "Step 1/4 — extract SMILES table"
python 01_extract_smiles.py

log "Step 2/4 — decompress SDF archives"
python 02_unzip_srtu_files.py

log "Step 3/4 — split multi-compound SDF files"
python 03_split_sdf_file.py

BUILD_ARGS=(
  --zinc-dir "$ZINC_DIR"
  --target-name "$TARGET_NAME"
  --protein-pdb "$PROTEIN_PDB"
  --pocket-pdb "$POCKET_PDB"
)
if [[ "$SKIP_MOL_PKL" == "1" ]]; then
  BUILD_ARGS+=(--skip-mol-pkl)
fi
if [[ "$SKIP_COMPOUNDS" == "1" ]]; then
  BUILD_ARGS+=(--skip-compounds)
fi
if [[ "$UPDATE_ZINC_CONFIG" == "1" ]]; then
  BUILD_ARGS+=(--update-zinc-config)
fi

log "Steps 4-8 — mol/pocket pkl, compound CSV, target snippet, validation"
python build_vs_dataset.py "${BUILD_ARGS[@]}"

log "Done. Model-ready files are in $ZINC_DIR"
log "Run virtual screening:"
log "  export MULTIGEODTA_DATA_DIR=$DATA_DIR"
if [[ "$UPDATE_ZINC_CONFIG" != "1" ]]; then
  log "  # Merge target block from $ZINC_DIR/target_snippet_${TARGET_NAME}.yaml into configs/tasks/zinc_vs.yaml"
fi
log "  python -m multigeodta screen --config configs/tasks/zinc_vs.yaml \\"
log "    --model_file pdbbind_v2020 --output_dir outputs/zinc --device 0"
