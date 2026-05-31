#!/usr/bin/env python3
"""Run ESMFold2 + DoGSite3 structure prediction (for the esmfold2 conda env)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from structure_pipeline import DEFAULT_STRUCTURE_CACHE_DIR, ensure_structure_files


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict protein + pocket structures with ESMFold2 and DoGSite3.",
    )
    parser.add_argument(
        "--protein_sequence", "-p",
        required=True,
        help="Full target amino-acid sequence (one-letter code).",
    )
    parser.add_argument(
        "--protein_pdb",
        type=Path,
        default=None,
        help="Existing full protein PDB (skip ESMFold2 if set).",
    )
    parser.add_argument(
        "--pocket_pdb",
        type=Path,
        default=None,
        help="Existing pocket PDB (skip DoGSite3 if set).",
    )
    parser.add_argument(
        "--structure_cache_dir",
        type=Path,
        default=None,
        help=f"Cache directory (default: {DEFAULT_STRUCTURE_CACHE_DIR}/<hash>).",
    )
    parser.add_argument(
        "--force_repredict",
        action="store_true",
        help="Ignore cached structure files and re-run ESMFold2 / DoGSite3.",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id (default: 0).")
    parser.add_argument("--esmfold_num_loops", type=int, default=3)
    parser.add_argument("--esmfold_num_sampling_steps", type=int, default=50)
    parser.add_argument("--esmfold_seed", type=int, default=0)
    parser.add_argument("--dogsite_chain", default="")
    parser.add_argument("--dogsite_ligand", default="")
    parser.add_argument("--dogsite_ligand_bias", action="store_true")
    parser.add_argument("--dogsite_pocket_id", default=None)
    parser.add_argument("--dogsite_poll_interval", type=float, default=10.0)
    parser.add_argument("--dogsite_max_attempts", type=int, default=60)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result paths as JSON to stdout.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    protein_pdb, pocket_pdb, metadata = ensure_structure_files(
        protein_sequence=args.protein_sequence,
        protein_pdb=args.protein_pdb,
        pocket_pdb=args.pocket_pdb,
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
    summary = {
        "protein_pdb": str(protein_pdb),
        "pocket_pdb": str(pocket_pdb),
        **metadata,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Protein structure: {protein_pdb}")
        print(f"Pocket structure:  {pocket_pdb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
