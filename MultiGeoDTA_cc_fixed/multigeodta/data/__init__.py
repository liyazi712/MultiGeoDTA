"""
Data Processing Module for MultiGeoDTA

This module provides:
- DTADataset: Base dataset class for drug-target affinity data
- Protein graph construction utilities
- Molecule graph construction utilities
- Constants and vocabularies
"""

from multigeodta.data.dataset import DTADataset, DTADataLoader
from multigeodta.data.protein_graph import pdb_to_graphs, featurize_protein_graph
from multigeodta.data.mol_graph import sdf_to_graphs, featurize_drug
from multigeodta.data.constants import (
    AA_MAP, LETTER_TO_NUM, NUM_TO_LETTER, ATOM_VOCAB,
    PROTEIN_CHAR, SMILES_CHAR
)

__all__ = [
    "DTADataset",
    "DTADataLoader",
    "pdb_to_graphs",
    "featurize_protein_graph",
    "sdf_to_graphs",
    "featurize_drug",
    "AA_MAP",
    "LETTER_TO_NUM",
    "NUM_TO_LETTER",
    "ATOM_VOCAB",
    "PROTEIN_CHAR",
    "SMILES_CHAR",
]
