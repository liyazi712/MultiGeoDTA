#!/usr/bin/env python3
"""
Predict drug-target binding affinity for a single protein–ligand pair.

Uses the five ensemble checkpoints trained on PDBBind v2020 and reports
the mean prediction (pKd / pKi scale, same as training labels).

Required inputs:
  - Protein amino-acid sequence (--protein_sequence)
  - Ligand SMILES (--smiles)

Structure inputs (choose one mode):

  A) Known structures (recommended when available):
     - --pocket_pdb + --protein_pdb
     - --pocket_pdb + --pocket_positions
     - --protein_pdb + --pocket_positions

  B) Unknown structures (--predict_structure):
     1. ESMFold2 folds the sequence to a full protein structure
     2. DoGSite3 (ProteinsPlus REST API) predicts the binding pocket
     3. MultiGeo-DTA runs on the generated structures

Example (known structures):
  python scripts/predict_affinity.py \\
    --protein_sequence "MKT..." \\
    --smiles "CCO" \\
    --pocket_pdb data/zinc/Target_information/CB1R/alphafold_DoGsite3_pocket.pdb \\
    --protein_pdb data/zinc/Target_information/CB1R/alphafold_protein.pdb

Example (structure prediction pipeline):
  python scripts/predict_affinity.py \\
    --protein_sequence "MKT..." \\
    --smiles "CCO" \\
    --predict_structure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from multigeodta.data import featurizers  # noqa: E402
from multigeodta.data.dataset import DTADataset  # noqa: E402
from multigeodta.data.tasks.base import DTATask, parse_positions  # noqa: E402
from multigeodta.models.dta_model import DTAModel  # noqa: E402
from multigeodta.utils.paths import resolve_checkpoint_dir  # noqa: E402
from structure_pipeline import DEFAULT_STRUCTURE_CACHE_DIR, ensure_structure_files  # noqa: E402

AA_MAP = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y", "SEC": "U", "PYL": "O", "HYP": "B", "SEP": "Z", "TPO": "J",
}

DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "outputs" / "pdbbind_v2020"
N_ENSEMBLES = 5


def _parse_pocket_positions(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or not str(raw).strip():
        return None
    return parse_positions(raw)


def _read_pdb_sequence(path: Path) -> Tuple[str, Dict[Tuple[str, str], int]]:
    """Return one-letter sequence and (chain, resseq) -> 1-based index mapping."""
    seq = ""
    order: Dict[Tuple[str, str], int] = {}
    idx = 0
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        columns = line.split()
        chain = columns[4]
        resseq = columns[5]
        if len(columns[4]) > 1:
            chain = columns[4][0]
            resseq = columns[4][1:]
        key = (chain, resseq)
        if key in order:
            continue
        idx += 1
        order[key] = idx
        seq += AA_MAP.get(columns[3], "X")
    return seq, order


def infer_pocket_positions(
    protein_sequence: str,
    pocket_pdb: Path,
    protein_pdb: Optional[Path] = None,
) -> List[int]:
    """Map pocket residues to 1-based indices in the full protein sequence."""
    if protein_pdb is not None:
        pdb_seq, order = _read_pdb_sequence(protein_pdb)
        if pdb_seq and pdb_seq != protein_sequence:
            print(
                f"Warning: --protein_sequence length={len(protein_sequence)} "
                f"differs from --protein_pdb sequence length={len(pdb_seq)}; "
                "using PDB numbering for pocket_positions.",
                file=sys.stderr,
            )
        positions: List[int] = []
        last_resseq = ""
        for line in pocket_pdb.read_text().splitlines():
            if not line.startswith("ATOM"):
                continue
            columns = line.split()
            chain = columns[4]
            resseq = columns[5]
            if len(columns[4]) > 1:
                chain = columns[4][0]
                resseq = columns[4][1:]
            if resseq == last_resseq:
                continue
            last_resseq = resseq
            positions.append(order[(chain, resseq)])
        return positions

    pocket_seq = ""
    last_resseq = ""
    for line in pocket_pdb.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        columns = line.split()
        resseq = columns[5]
        if len(columns[4]) > 1:
            resseq = columns[4][1:]
        if resseq == last_resseq:
            continue
        last_resseq = resseq
        pocket_seq += AA_MAP.get(columns[3], "X")

    start = protein_sequence.find(pocket_seq)
    if start < 0:
        raise ValueError(
            "Cannot align pocket PDB residues to --protein_sequence. "
            "Provide --protein_pdb or explicit --pocket_positions."
        )
    return list(range(start + 1, start + len(pocket_seq) + 1))


def pocket_pdb_to_entry(pocket_pdb: Path) -> Dict:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pocket", str(pocket_pdb))
    protein_sequence: List[str] = []
    coords = {"N": [], "CA": [], "C": [], "O": []}
    for residue in structure.get_residues():
        if not residue.has_id("CA"):
            continue
        aa_one = AA_MAP.get(residue.get_resname())
        if aa_one is None:
            continue
        protein_sequence.append(aa_one)
        for atom in residue:
            name = atom.name.strip()
            if name in coords:
                coords[name].append([round(float(x), 3) for x in atom.coord])
    return {"seq": "".join(protein_sequence), "coords": coords}


def extract_pocket_from_protein_pdb(
    protein_pdb: Path,
    pocket_positions: List[int],
) -> Dict:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(protein_pdb))
    idx_to_residue = {}
    idx = 0
    seen = set()
    for model in structure:
        for chain in model:
            for residue in chain:
                if not residue.has_id("CA"):
                    continue
                resseq = str(residue.id[1])
                chain_id = chain.id
                key = (chain_id, resseq)
                if key in seen:
                    continue
                seen.add(key)
                idx += 1
                idx_to_residue[idx] = residue

    coords = {"N": [], "CA": [], "C": [], "O": []}
    pocket_seq: List[str] = []
    for pos in pocket_positions:
        if pos not in idx_to_residue:
            raise ValueError(f"Pocket position {pos} not found in {protein_pdb}")
        residue = idx_to_residue[pos]
        aa_one = AA_MAP.get(residue.get_resname())
        if aa_one is None:
            raise ValueError(f"Unsupported residue at position {pos}: {residue.get_resname()}")
        pocket_seq.append(aa_one)
        for atom in residue:
            name = atom.name.strip()
            if name in coords:
                coords[name].append([round(float(x), 3) for x in atom.coord])
    return {"seq": "".join(pocket_seq), "coords": coords}


def smiles_to_mol_entry(smiles: str, compound_id: str = "query") -> Dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, randomSeed=42)
    if status != 0:
        status = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
    if status != 0:
        raise ValueError(f"Failed to generate 3D conformer for SMILES: {smiles}")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    conf = mol.GetConformer()
    coords = conf.GetPositions().tolist()
    atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
    if len(atom_types) != len(coords):
        raise ValueError("Atom count mismatch after 3D embedding.")
    return {
        "mol": mol,
        "coords": coords,
        "atom_types": atom_types,
        "id": compound_id,
    }


def build_sample(
    protein_sequence: str,
    smiles: str,
    pocket_positions: List[int],
    pocket_struct: Dict,
    featurize_params: Dict,
) -> Dict:
    task = DTATask(
        task_name="inference",
        max_seq_len=1024,
        max_smi_len=128,
        num_pos_emb=featurize_params["num_pos_emb"],
        num_rbf=featurize_params["num_rbf"],
        contact_cutoff=featurize_params["contact_cutoff"],
    )
    mol_entry = smiles_to_mol_entry(smiles)
    mol_graph = featurizers.sdf_to_graphs(
        {mol_entry["id"]: {
            "mol": mol_entry["mol"],
            "coords": mol_entry["coords"],
            "atom_types": mol_entry["atom_types"],
        }}
    )[mol_entry["id"]]
    pocket_entry = task._format_pdb_entry(pocket_struct)
    protein_graph = featurizers.pdb_to_graphs(
        {"target": pocket_entry},
        featurize_params,
    )["target"]
    pocket_masked = task.mask_pocket(protein_sequence, pocket_positions)
    return {
        "drug_graph": mol_graph,
        "protein_graph": protein_graph,
        "full_sequence": task.encode_protein(protein_sequence),
        "pocket_sequence": task.encode_protein(pocket_masked),
        "smile_sequence": task.encode_smiles(smiles),
        "smile": smiles,
    }


def resolve_pocket_inputs(
    protein_sequence: str,
    pocket_pdb: Optional[Path],
    protein_pdb: Optional[Path],
    pocket_positions: Optional[List[int]],
) -> Tuple[List[int], Dict]:
    if pocket_pdb is not None:
        pocket_struct = pocket_pdb_to_entry(pocket_pdb)
        if pocket_positions is None:
            pocket_positions = infer_pocket_positions(
                protein_sequence, pocket_pdb, protein_pdb
            )
        return pocket_positions, pocket_struct

    if protein_pdb is not None and pocket_positions is not None:
        return pocket_positions, extract_pocket_from_protein_pdb(protein_pdb, pocket_positions)

    raise ValueError(
        "3D pocket structure is required. Provide one of:\n"
        "  • --pocket_pdb (optionally with --protein_pdb for automatic pocket_positions)\n"
        "  • --protein_pdb together with --pocket_positions\n"
        "  • --predict_structure to run ESMFold2 + DoGSite3 automatically"
    )


@torch.no_grad()
def predict_ensemble(
    sample: Dict,
    checkpoint_dir: Path,
    device_id: int = 0,
    n_ensembles: int = N_ENSEMBLES,
    mlp_dims: Optional[List[int]] = None,
    mlp_dropout: float = 0.25,
) -> Dict:
    dataset = DTADataset([sample])
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=dataset.collate,
        shuffle=False,
    )
    batch = next(iter(loader))
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    model_config = {
        "mlp_dims": mlp_dims or [1024, 512],
        "mlp_dropout": mlp_dropout,
    }

    predictions: List[float] = []
    for midx in range(n_ensembles):
        ckpt_path = checkpoint_dir / f"checkpoint_{midx + 1}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        model = DTAModel(**model_config).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        xd = batch["drug"].to(device)
        xp = batch["protein"].to(device)
        protein_seq = batch["full_seq"].to(device)
        pocket_seq = batch["poc_seq"].to(device)
        smile_seq = batch["smile_seq"].to(device)
        yh, _, _, _, _ = model(xd, xp, protein_seq, pocket_seq, smile_seq)
        predictions.append(float(yh.item()))

    y_pred_avg = float(np.mean(predictions))
    return {
        "y_pred_avg": y_pred_avg,
        **{f"y_pred_{i + 1}": predictions[i] for i in range(n_ensembles)},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict drug-target binding affinity with MultiGeo-DTA (PDBBind v2020 ensemble).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--protein_sequence", "-p",
        required=True,
        help="Full target amino-acid sequence (one-letter code).",
    )
    parser.add_argument(
        "--smiles", "-s",
        required=True,
        help="Ligand SMILES string.",
    )
    parser.add_argument(
        "--pocket_pdb",
        type=Path,
        default=None,
        help="Pocket structure PDB file (optional but recommended).",
    )
    parser.add_argument(
        "--protein_pdb",
        type=Path,
        default=None,
        help="Full protein structure PDB (used to infer pocket_positions).",
    )
    parser.add_argument(
        "--pocket_positions",
        default=None,
        help="1-based pocket residue indices, e.g. '10,12,72' or '[10, 12, 72]'.",
    )
    parser.add_argument(
        "--predict_structure",
        action="store_true",
        help="When structure files are missing, run ESMFold2 (fold) and DoGSite3 (pocket).",
    )
    parser.add_argument(
        "--structure_cache_dir",
        type=Path,
        default=None,
        help="Directory to cache ESMFold2 / DoGSite3 outputs (default: outputs/structure_cache/<hash>).",
    )
    parser.add_argument(
        "--force_repredict",
        action="store_true",
        help="Ignore cached structure files and re-run ESMFold2 / DoGSite3.",
    )
    parser.add_argument(
        "--esmfold_num_loops",
        type=int,
        default=3,
        help="ESMFold2 diffusion loops (default: 3).",
    )
    parser.add_argument(
        "--esmfold_num_sampling_steps",
        type=int,
        default=50,
        help="ESMFold2 sampling steps per loop (default: 50).",
    )
    parser.add_argument(
        "--esmfold_seed",
        type=int,
        default=0,
        help="Random seed for ESMFold2 (default: 0).",
    )
    parser.add_argument(
        "--dogsite_chain",
        default="",
        help="DoGSite3 chain filter (empty = all chains).",
    )
    parser.add_argument(
        "--dogsite_ligand",
        default="",
        help='DoGSite3 reference ligand, e.g. "JE2_A_701" (optional).',
    )
    parser.add_argument(
        "--dogsite_ligand_bias",
        action="store_true",
        help="Bias DoGSite3 grid toward --dogsite_ligand.",
    )
    parser.add_argument(
        "--dogsite_pocket_id",
        default=None,
        help='Use a specific DoGSite3 pocket id (e.g. "P_1") instead of auto-ranking.',
    )
    parser.add_argument(
        "--dogsite_poll_interval",
        type=float,
        default=10.0,
        help="Seconds between DoGSite3 status polls (default: 10).",
    )
    parser.add_argument(
        "--dogsite_max_attempts",
        type=int,
        default=60,
        help="Maximum DoGSite3 polling attempts (default: 60).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Directory with checkpoint_1.pt … checkpoint_{N_ENSEMBLES}.pt "
             f"(default: {DEFAULT_CHECKPOINT_DIR}).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device id (default: 0). Uses CPU when CUDA is unavailable.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Optional JSON file to save the prediction summary.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON to stdout.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    checkpoint_dir = resolve_checkpoint_dir(model_file=str(args.checkpoint_dir))
    if not (checkpoint_dir / "checkpoint_1.pt").exists():
        raise FileNotFoundError(
            f"No checkpoints found under {checkpoint_dir}. "
            "Download weights or set --checkpoint_dir."
        )

    structure_metadata: Dict = {}
    protein_pdb = args.protein_pdb
    pocket_pdb = args.pocket_pdb
    needs_protein = protein_pdb is None
    needs_pocket = pocket_pdb is None and args.pocket_positions is None

    if needs_protein or needs_pocket:
        if not args.predict_structure:
            missing = []
            if needs_protein:
                missing.append("--protein_pdb")
            if needs_pocket:
                missing.append("--pocket_pdb (or --pocket_positions)")
            raise ValueError(
                "Missing structure input(s): "
                + ", ".join(missing)
                + ". Provide them explicitly or add --predict_structure to run "
                "ESMFold2 + DoGSite3 automatically."
            )
        protein_pdb, pocket_pdb, structure_metadata = ensure_structure_files(
            protein_sequence=args.protein_sequence,
            protein_pdb=protein_pdb,
            pocket_pdb=pocket_pdb,
            cache_dir=args.structure_cache_dir,
            device_id=args.device,
            force_repredict=args.force_repredict,
            esmfold_num_loops=args.esmfold_num_loops,
            esmfold_num_sampling_steps=args.esmfold_num_sampling_steps,
            esmfold_seed=args.esmfold_seed,
            dogsite_chain=args.dogsite_chain,
            dogsite_ligand=args.dogsite_ligand,
            dogsite_ligand_bias=args.dogsite_ligand_bias,
            dogsite_pocket_id=args.dogsite_pocket_id,
            dogsite_poll_interval=args.dogsite_poll_interval,
            dogsite_max_attempts=args.dogsite_max_attempts,
        )

    pocket_positions_arg = _parse_pocket_positions(args.pocket_positions)
    pocket_positions, pocket_struct = resolve_pocket_inputs(
        protein_sequence=args.protein_sequence,
        pocket_pdb=pocket_pdb,
        protein_pdb=protein_pdb,
        pocket_positions=pocket_positions_arg,
    )

    featurize_params = {
        "num_pos_emb": 16,
        "num_rbf": 16,
        "contact_cutoff": 8.0,
    }
    sample = build_sample(
        protein_sequence=args.protein_sequence,
        smiles=args.smiles,
        pocket_positions=pocket_positions,
        pocket_struct=pocket_struct,
        featurize_params=featurize_params,
    )

    result = predict_ensemble(
        sample=sample,
        checkpoint_dir=checkpoint_dir,
        device_id=args.device,
    )

    summary = {
        "protein_sequence_length": len(args.protein_sequence),
        "smiles": args.smiles,
        "pocket_positions": pocket_positions,
        "checkpoint_dir": str(checkpoint_dir),
        **structure_metadata,
        **result,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved prediction to {args.output}")

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("MultiGeo-DTA binding affinity prediction (PDBBind v2020 ensemble)")
        print(f"  SMILES:              {args.smiles}")
        print(f"  Protein length:      {len(args.protein_sequence)}")
        print(f"  Pocket residues:     {len(pocket_positions)}")
        if structure_metadata:
            if "protein_pdb" in structure_metadata:
                print(f"  Protein structure:   {structure_metadata['protein_pdb']}")
            if "pocket_pdb" in structure_metadata:
                print(f"  Pocket structure:    {structure_metadata['pocket_pdb']}")
            if "dogsite_pocket_id" in structure_metadata:
                print(f"  DoGSite3 pocket:     {structure_metadata['dogsite_pocket_id']}")
        print(f"  Per-model predictions:")
        for i in range(N_ENSEMBLES):
            print(f"    Model {i + 1}: {result[f'y_pred_{i + 1}']:.4f}")
        print(f"  Ensemble average:    {result['y_pred_avg']:.4f}  (pKd/pKi scale)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
